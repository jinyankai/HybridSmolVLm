import math

from dataset.cauldron import CauldronDataset,get_dataloader
from config.train_config import TrainConfig
from model.mamba2.hybrid_mamba_config import PhiMambaConfig
from model.model_wrapper import HybridSmolVLMForConditionalGeneration,HybridSmolVLMWrapper
from transformers import AutoProcessor, AutoTokenizer, logging as hf_logging, get_scheduler
from utils.loss import VLMDitillationLoss,DistillationLossConfig
import logging
logger = logging.getLogger(__name__)
from accelerate import Accelerator
from accelerate.utils import set_seed
import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F
from pathlib import Path

def build_model(cfg:TrainConfig):
    # Prepare Teacher Model
    teacher_model = HybridSmolVLMForConditionalGeneration.from_pretrained(cfg.teacher_name,torch_dtype=cfg.dtype)
    for param in teacher_model.parameters():
        param.requires_grad = False
    logger.log(logging.INFO, f"Teacher model {cfg.teacher_name} loaded successfully.")
    mamba_cfg = PhiMambaConfig(
        d_model=2048,
        ssm_cfg={"expand": 1, "ngroups":32, "d_state": 64,"d_conv": 4},
        rms_norm_eps=1e-05,
        d_inner=2048,
        d_xb=2048,
        intermediate_size=8192,
        hidden_act="silu",
        n_layer=24,
        attn_layers=[0,1,2,3,5,6,7,9,10,11,13,14,15,16,17,18,19,21,22,23],
        resid_pdrop=0.1,
        bidirectional=False,
        is_bias=False
        )
    student_wrapper = HybridSmolVLMWrapper.init_distillation(checkpoint_path=None,
                                                             tranformer_name=cfg.teacher_name,
                                                             mamba_config=mamba_cfg,
                                                             attn_layers=mamba_cfg.attn_layers,
                                                             dtype=cfg.dtype)
    student_model = student_wrapper.model
    student_model.gradient_checkpointing_enable()
    for name, param in student_model.named_parameters():
        if "mamba" in name:
            param.requires_grad = True
            print("name of the trainable param",name)
        else:
            param.requires_grad = False

    logger.info("Successfully load Student, Building tok")

    tok = AutoTokenizer.from_pretrained(cfg.teacher_name, local_files_only=True, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    logger.info("Processor")
    processor = AutoProcessor.from_pretrained(cfg.teacher_name, local_files=True)

    return teacher_model,student_wrapper,student_model,tok, processor

def evaluate(model, loader, accel: Accelerator):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader or []:
            batch = {k: v.to(accel.device) for k, v in batch.items() if torch.is_tensor(v)}
            out = model(**batch)
            total += out.loss.float().item()
            n += 1
    model.train()
    return total / max(n, 1)

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    for f in TrainConfig.__dataclass_fields__.values():
        name = f"--{f.name}"
        p.add_argument(name, type=type(f.default) if f.default is not None else str, default=f.default, nargs="?",
                       help=f.metadata.get("help", ""))
    args = p.parse_args()
    return TrainConfig(**vars(args))

def main():
    cfg = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    hf_logging.set_verbosity_error()

    accelerator = Accelerator(log_with="tensorboard", project_dir=cfg.logging_dir)
    set_seed(cfg.seed)

    accelerator.wait_for_everyone()
    teacher_model, student_wrapper,student_model,tok,processor = build_model(cfg)
    if accelerator.is_main_process:
        print("teacher_model:", teacher_model)
        total_params = sum(p.numel() for p in teacher_model.parameters())
        total_trainable_params = sum(
            p.numel() for p in teacher_model.parameters() if p.requires_grad)
        print(f"number of total params:{total_params}")
        print(f"number of total trainable params:{total_trainable_params}")

        print("student_model:", student_model)
        total_params = sum(p.numel() for p in student_model.parameters())
        total_trainable_params = sum(
            p.numel() for p in student_model.parameters() if p.requires_grad)
        print(f"number of total params:{total_params}")
        print(f"number of total trainable params:{total_trainable_params}")
        for name, param in student_model.named_parameters():
            if param.requires_grad:
                print(f"Parameter: {name}, requires_grad: {param.requires_grad}")
        student_wrapper.save_config(cfg.output)
    logger.info("Loading data")

    train_loader = get_dataloader(cfg,processor)

    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum)
    max_steps = cfg.max_steps if cfg.max_steps > 0 else steps_per_epoch * cfg.num_epochs

    loss_config = DistillationLossConfig(
        kl_weight=1.0,
        l2_weight=0.5,
        ce_weight=0.0,
        temperature=1.0,
        l2_loss_layers=[4, 8, 12, 20]
    )
    distil_loss = VLMDitillationLoss(config=loss_config)
    # Optimiser & scheduler
    logger.info("Optimizer & Scheduler Setup")
    no_decay = {"bias", "LayerNorm.weight"}
    optim_groups = [
        {"params": [p for n, p in student_model.named_parameters() if
                    p.requires_grad and not any(nd in n for nd in no_decay)],
         "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in student_model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2))
    scheduler = get_scheduler("cosine", optimizer,
                              num_warmup_steps=cfg.warmup_steps,
                              num_training_steps=max_steps)
    # prepare
    logger.info("Prepare")
    teacher_model, student, optimizer, train_loader, scheduler = accelerator.prepare(
        teacher_model, student_model, optimizer, train_loader, scheduler)
    teacher_model.eval()

    if cfg.check_point_path is not None and cfg.check_point_path != "" and cfg.resume_from_checkpoint:
        accelerator.print(f"resume from checkpoint '{cfg.check_point_path}' 恢复训练...")
        accelerator.load_state(cfg.check_point_path)
        # 加载后，你注册的 train_state 会被自动更新
        accelerator.print("success in resuming！")
    else:
        accelerator.print("未找到 checkpoint, 从头开始训练。")

    if accelerator.is_main_process:
        from torch.utils.tensorboard import SummaryWriter
        tbwriter = SummaryWriter(log_dir=cfg.logging_dir)

    logger.info("Train")

    global_step = 0

    for epoch in range(cfg.num_epochs):
        student.train()
        from tqdm.auto import tqdm
        import collections
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{cfg.num_epochs}",
            disable=not accelerator.is_main_process
        )
        running_losses = collections.defaultdict(float)

        for step, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(accelerator.device, non_blocking=True)
            attn = batch["attention_mask"].to(accelerator.device, non_blocking=True)
            labels = batch["labels"].to(accelerator.device, non_blocking=True)
            images = batch["pixel_values"].to(
                accelerator.device,
                dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
                non_blocking=True,
            )

            batch = {
                "input_ids": ids,  # still int64
                "attention_mask": attn,  # int64
                "labels": labels,  # int64
                "pixel_values": images,  # bf16 / fp16
            }
            if loss_config.l2_weight > 0:
                with torch.no_grad():
                    t_out = teacher_model(**batch, output_hidden_states=True)
                    t_out_h = t_out.hidden_states
                s_out = student(**batch, output_hidden_states=True, teacher_outputs=t_out_h)
                student_hidden_states= s_out.hidden_states
                total_loss, loss_details = distil_loss(
                    student_logits=s_out.logits,
                    teacher_logits=t_out.logits,
                    student_ce_loss=s_out.loss,
                    teacher_hidden_states=t_out_h,
                    student_hidden_states=student_hidden_states
                )

            else:
                with torch.no_grad():
                    t_out = teacher_model(**batch, output_hidden_states=True)
                s_out = student(**batch, output_hidden_states=True)
                total_loss, loss_details = distil_loss(
                    student_logits=s_out.logits,
                    teacher_logits=t_out.logits,
                    student_ce_loss=s_out.loss,
                    student_hidden_states=s_out.hidden_states,
                    teacher_hidden_states=t_out.hidden_states
                )

            # 应用梯度累积
            loss_for_backward = total_loss / cfg.grad_accum
            accelerator.backward(loss_for_backward)

            for k, v in loss_details.items():
                running_losses[k] += v.item() / cfg.grad_accum

            if (step + 1) % cfg.grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if accelerator.is_main_process:
                    # 3. 更新进度条的后缀信息，动态显示损失和学习率
                    progress_bar.set_postfix(
                        loss=running_losses["total_loss"] / global_step,
                        kl=running_losses['kl_loss'] / global_step,
                        l2=running_losses['l2_loss'] / global_step,
                        ce=running_losses['ce_loss'] / global_step,
                        lr=scheduler.get_last_lr()[0]
                    )

                    # 4. 按 global_step 记录到TensorBoard
                    if global_step % 2 == 0:  # 您可以调整记录频率
                        tbwriter.add_scalar('train/total_loss', running_losses['total_loss'] / global_step, global_step)
                        tbwriter.add_scalar('train/kl_loss', running_losses['kl_loss'] / global_step, global_step)
                        tbwriter.add_scalar('train/l2_loss', running_losses['l2_loss'] / global_step, global_step)
                        tbwriter.add_scalar('train/ce_loss', running_losses['ce_loss'] / global_step, global_step)
                        tbwriter.add_scalar('learning_rate', scheduler.get_last_lr()[0], global_step)

                # 保存模型检查点
                if cfg.save_steps and global_step % cfg.save_steps == 0:
                    path = Path(cfg.output) / f"step_{global_step}"
                    accelerator.save_state(path)

                if cfg.eval_steps and global_step % cfg.eval_steps == 0:
                    eval_loss = evaluate(student, None, accelerator)  # no val loader stub
                    if accelerator.is_main_process:
                        tbwriter.add_scalar('loss', eval_loss, global_step)

            if global_step >= max_steps:
                break
        if global_step >= max_steps:
            break

    # final save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        out_dir = Path(cfg.output) / "final"
        #if path doesn't exist create one
        out_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(student)  # <-- this is the right "unwrap"
        if hasattr(unwrapped, "save_pretrained"):
            unwrapped.save_pretrained(out_dir)
        else:
            accelerator.save(unwrapped.state_dict(), out_dir / "pytorch_model.bin")
        tok.save_pretrained(out_dir)
    accelerator.end_training()

if __name__ == "__main__":
    # accelerate
    # launch - -num_processes
    # 4 - -mixed_precision
    # bf16 - -deepspeed_config_file
    # deepspeed_config.json
    # train.py
    main()
