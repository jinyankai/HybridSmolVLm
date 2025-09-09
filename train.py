import math
import os
os.environ['HF_ENDPOINT']="https://hf-mirror.com"
from dataset.cauldron import get_dataloader
from dataset.sharegpt4v import make_supervised_data_module,DataArguments
from config.train_config import TrainConfig
from model.decoder_layer import HybridDecoderLayers
from model.mamba2.hybrid_mamba_config import PhiMambaConfig
from model.model_wrapper import HybridSmolVLMForConditionalGeneration,HybridSmolVLMWrapper
from transformers import AutoProcessor, AutoTokenizer, logging as hf_logging, get_scheduler
from utils.loss import VLMDitillationLoss,DistillationLossConfig
import logging
logger = logging.getLogger(__name__)
from accelerate import Accelerator
from accelerate.utils import set_seed
import torch
from torch.utils.data import DataLoader

import argparse
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
        attn_layers=[0,1,2,3,5,6,7,9,10,11,12,13,14,15,17,18,19,21,22,23],
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
    for param in student_model.parameters():
        param.requires_grad = False

    # 2. 然后，通过遍历模块来精确地解冻需要的子模块
    for module_name, module in student_model.named_modules():
        # 检查当前模块的类型是否是我们的目标自定义层
        if isinstance(module, HybridDecoderLayers):

            print(f"✅ Found Hybrid Layer: '{module_name}'. Unfreezing its components.")

            # 将该模块内的 mamba 和 mlp 的参数设为可训练
            for sub_module_name, sub_module in module.named_children():
                if sub_module_name == "self_attn_mamba" or sub_module_name == "mlp" or sub_module_name == "post_attention_layernorm" or sub_module_name == "input_layernorm":
                    print(f"  -- Unfreezing sub-module: '{sub_module_name}'")
                    for param in sub_module.parameters():
                        param.requires_grad = True


    logger.info("Successfully load Student, Building tok")

    tok = AutoTokenizer.from_pretrained(cfg.teacher_name, local_files_only=True, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    logger.info("Processor")
    processor = AutoProcessor.from_pretrained(cfg.teacher_name, local_files_only=True)

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
            if n > 10:
                break
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
    data_args = DataArguments(
        data_path="/data/jyk_data/data/sharegpt4v/cleaned_sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json",
        image_folder="/data/jyk_data/data/"
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    hf_logging.set_verbosity_error()

    accelerator = Accelerator(log_with="tensorboard", project_dir=cfg.logging_dir,gradient_accumulation_steps=cfg.grad_accum)
    set_seed(cfg.seed)

    accelerator.wait_for_everyone()
    teacher_model, student_wrapper,student_model,tok,processor = build_model(cfg)
    if accelerator.is_main_process:
        report_path = "params_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:

            # teacher
            print("teacher_model:", teacher_model)
            f.write(f"teacher_model: {teacher_model}\n")

            total_params = sum(p.numel() for p in teacher_model.parameters())
            total_trainable_params = sum(
                p.numel() for p in teacher_model.parameters() if p.requires_grad)

            print(f"number of total params:{total_params}")
            print(f"number of total trainable params:{total_trainable_params}")
            f.write(f"number of total params: {total_params}\n")
            f.write(f"number of total trainable params: {total_trainable_params}\n\n")

            # student
            print("student_model:", student_model)
            f.write(f"student_model: {student_model}\n")

            total_params = sum(p.numel() for p in student_model.parameters())
            total_trainable_params = sum(
                p.numel() for p in student_model.parameters() if p.requires_grad)

            print(f"number of total params:{total_params}")
            print(f"number of total trainable params:{total_trainable_params}")
            f.write(f"number of total params: {total_params}\n")
            f.write(f"number of total trainable params: {total_trainable_params}\n\n")

            # 每个参数
            for name, param in student_model.named_parameters():
                if param.requires_grad:
                    print(f"Parameter: {name}, requires_grad: {param.requires_grad}")
                    f.write(f"Parameter: {name}, requires_grad: {param.requires_grad}\n")

        print(f"参数信息已保存到 {report_path}")

        student_wrapper.save_config(cfg.output)
    logger.info("Loading data")

    # train_loader= get_dataloader(cfg,processor)
    # eval_loader= get_dataloader(cfg,processor)
    train_dict = make_supervised_data_module(processor=processor,
                                                               data_args=data_args)
    train_loader = DataLoader(dataset=train_dict['train_dataset'],batch_size=cfg.batch_size,
                              shuffle=True,num_workers=cfg.num_workers,collate_fn=train_dict['data_collator'],pin_memory=True )
    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum)
    max_steps = cfg.max_steps if cfg.max_steps > 0 else steps_per_epoch * cfg.num_epochs

    loss_config = DistillationLossConfig(
        kl_weight=1.,
        l2_weight=1.0,
        ce_weight=1.0,
        temperature=1.0,
        l2_loss_layers=[4, 8, 16, 20]
    )
    distil_loss = VLMDitillationLoss(config=loss_config, use_topk=True)
    # Optimiser & scheduler
    logger.info("Optimizer & Scheduler Setup")
    # no_decay = []
    decay, no_decay = [], []
    for n, p in student_model.named_parameters():
        if not p.requires_grad:
            continue
        is_norm = ("norm" in n.lower()) or n.endswith(".bias")
        (no_decay if is_norm else decay).append(p)

    optim_groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=cfg.lr, betas=(cfg.adam_beta1, cfg.adam_beta2))
    scheduler = get_scheduler("cosine", optimizer,
                              num_warmup_steps=cfg.warmup_steps,
                              num_training_steps=max_steps // cfg.num_epochs)
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
            batch = {k: (v.to(accelerator.device, non_blocking=True) if hasattr(v, "to") else v)
                     for k, v in batch.items()}


            kd_mask = (batch['labels'] != -100).to(torch.long)
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
                    student_hidden_states=student_hidden_states,
                    attention_mask=kd_mask,
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
                    teacher_hidden_states=t_out.hidden_states,
                    attention_mask=kd_mask,
                )

            # 应用梯度累积
            loss_for_backward = total_loss / cfg.grad_accum
            accelerator.backward(loss_for_backward)

            for k, v in loss_details.items():
                running_losses[k] += v.item() / cfg.grad_accum

            if (step + 1) % cfg.grad_accum == 0:
                accelerator.clip_grad_norm_(student.parameters(), 1.0)
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
                        # z_loss=running_losses['z_loss']
                        lr=scheduler.get_last_lr()[0],
                        step= global_step
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
                    path_1 = Path(cfg.output) / f"step_{global_step}_1"
                    accelerator.save_state(path)
                    accelerator.unwrap_model(student).save_pretrained(path_1)

                # if cfg.eval_steps and global_step % cfg.eval_steps == 0:
                #     eval_loss = evaluate(student, eval_loader, accelerator)  # no val loader stub
                #     if accelerator.is_main_process:
                #         tbwriter.add_scalar('loss', eval_loss, global_step)

            if global_step >= max_steps:
                break
        if global_step >= max_steps:
            break

    # final save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        out_dir = Path(cfg.output) / "final"
        acc_out = Path(cfg.output)/ "last"
        #if path doesn't exist create one
        out_dir.mkdir(parents=True, exist_ok=True)
        accelerator.save_state(output_dir=acc_out)
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
