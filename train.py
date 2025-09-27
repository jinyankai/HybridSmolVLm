import math
import os
os.environ['HF_ENDPOINT']="https://hf-mirror.com"
from dataset.sharegpt4v import make_supervised_data_module,DataArguments
from config.train_config import TrainConfig
from model.decoder_layer import HybridDecoderLayers
from model.mamba2.hybrid_mamba_config import PhiMambaConfig
from model.model_wrapper import HybridSmolVLMForConditionalGeneration,HybridSmolVLMWrapper
from model.load_sftensor import load_safetensors_to_dict,construct_language_layer_dict
from transformers import AutoProcessor, AutoTokenizer, logging as hf_logging, get_scheduler
from utils.loss import VLMDitillationLoss,DistillationLossConfig
import logging
logger = logging.getLogger(__name__)
from accelerate import Accelerator,DistributedDataParallelKwargs
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
        d_model=576,
        ssm_cfg={"expand": 1, "ngroups": 9, "d_state": 64, "d_conv": 4},
        rms_norm_eps=1e-05,
        d_inner=576,
        d_xb=192,
        intermediate_size=1536,
        hidden_act="silu",
        n_layer=30,
        attn_layers=[1,2,4,5, 7, 8, 10, 11,12, 13, 14, 15, 16, 17,18, 19, 20, 21, 22, 23,24, 25, 26, 27, 28, 29],
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
    # load weight
    if cfg.prev_model_path is not None and cfg.resume_model:
        # load from a local directory
        logger.info(f"Loading previous model weights from {cfg.prev_model_path}")
        if os.path.exists(f"{cfg.prev_model_path}/pytorch_model.bin"):
            # support save from bin file
            student_model.load_state_dict(
            torch.load(f"{cfg.prev_model_path}/pytorch_model.bin", map_location=torch.device("cpu")))

        else:
            prev_ckp = load_safetensors_to_dict(cfg.prev_model_path)
            prev_checkpoint_layers, is_mamba_layer = construct_language_layer_dict(prev_ckp, mamba_cfg.n_layer)
            print(is_mamba_layer)
            for (layer_id, layer_ckp) in prev_checkpoint_layers.items():
                if is_mamba_layer[layer_id]:
                    student_model.model.text_model.layers[layer_id].load_state_dict(layer_ckp)


    for param in student_model.parameters():
        param.requires_grad = False

    # 2. 然后，通过遍历模块来精确地解冻需要的子模块
    for module_name, module in student_model.named_modules():
        # 检查当前模块的类型是否是我们的目标自定义层
        if isinstance(module, HybridDecoderLayers):

            print(f"✅ Found Hybrid Layer: '{module_name}'. Unfreezing its components.")
            module.gradient_checkpointing = True
            # 将该模块内的 mamba 和 mlp 的参数设为可训练
            for sub_module_name, sub_module in module.named_children():
                print(f"  -- Unfreezing sub-module: '{sub_module_name}'")
                for param in sub_module.parameters():
                    param.requires_grad = True



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
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with="tensorboard", project_dir=cfg.logging_dir
                              ,gradient_accumulation_steps=cfg.grad_accum,
                             )
    set_seed(cfg.seed)

    accelerator.wait_for_everyone()
    teacher_model, student_wrapper,student_model,tok,processor = build_model(cfg)
    if accelerator.is_main_process:
        report_path = "params_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            # teacher
            f.write(f"teacher_model: {teacher_model}\n")

            total_params = sum(p.numel() for p in teacher_model.parameters())
            total_trainable_params = sum(
                p.numel() for p in teacher_model.parameters() if p.requires_grad)

            f.write(f"number of total params: {total_params}\n")
            f.write(f"number of total trainable params: {total_trainable_params}\n\n")

            # student
            f.write(f"student_model: {student_model}\n")

            total_params = sum(p.numel() for p in student_model.parameters())
            total_trainable_params = sum(
                p.numel() for p in student_model.parameters() if p.requires_grad)

            f.write(f"number of total params: {total_params}\n")
            f.write(f"number of total trainable params: {total_trainable_params}\n\n")

            # 每个参数
            for name, param in student_model.named_parameters():
                if param.requires_grad:
                    f.write(f"Parameter: {name}, requires_grad: {param.requires_grad}\n")

        print(f"参数信息已保存到 {report_path}")

        student_wrapper.save_config(cfg.output)
    logger.info("Loading data")

    train_dict = make_supervised_data_module(processor=processor,
                                                               data_args=data_args)
    train_loader = DataLoader(dataset=train_dict['train_dataset'],batch_size=cfg.batch_size,
                              shuffle=True,num_workers=cfg.num_workers,collate_fn=train_dict['data_collator'],pin_memory=True )

    cfg.grad_accum = cfg.grad_accum

    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum)
    total_update_steps = steps_per_epoch * cfg.num_epochs

    loss_config = DistillationLossConfig(
        kl_weight=1.,
        l2_weight=1.,
        ce_weight=0.0,
        temperature=1.0,
        l2_loss_layers=[0,3,6,9]
    )
    distil_loss = VLMDitillationLoss(config=loss_config, use_topk=True)
    # Optimiser & scheduler
    logger.info("Optimizer & Scheduler Setup")

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
    scheduler = get_scheduler("warmup_stable_decay", optimizer,
                              num_warmup_steps=cfg.warmup_steps,
                              num_training_steps=cfg.max_steps,
                              scheduler_specific_kwargs={"num_stable_steps": cfg.stable_steps, "num_decay_steps": cfg.decay_steps})
    # prepare
    logger.info("Prepare")
    teacher_model, student, optimizer,train_loader= accelerator.prepare(
        teacher_model, student_model, optimizer,train_loader)
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
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dict['train_dataset'])}")
    logger.info(f"  Num Epochs = {cfg.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {cfg.batch_size*accelerator.num_processes}")
    logger.info(f"  Gradient Accumulation steps = {cfg.grad_accum}")
    logger.info(f"  Total optimization steps = {total_update_steps}")
    accelerator.wait_for_everyone()

    global_step = 0
    try :

        for epoch in range(cfg.num_epochs):
            student.train()
            from tqdm.auto import tqdm
            import collections
            progress_bar = tqdm(
                total = total_update_steps,
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
                        t_out = teacher_model(**batch, output_hidden_states=True,)
                        t_out_h = t_out.hidden_states
                    s_out = student(**batch, output_hidden_states=True,teacher_outputs=t_out_h,distillation_layers=loss_config.l2_loss_layers,use_cache=False)
                    total_loss, loss_details = distil_loss(
                        student_logits=s_out.logits,
                        teacher_logits=t_out.logits,
                        student_ce_loss=s_out.loss,
                        distillation_alignment_outputs = s_out.distillation_alignment_outputs,
                        attention_mask=kd_mask,
                        step=global_step
                    )

                else:
                    with torch.no_grad():
                        t_out = teacher_model(**batch, output_hidden_states=True)
                    s_out = student(**batch, output_hidden_states=True)
                    total_loss, loss_details = distil_loss(
                        student_logits=s_out.logits,
                        teacher_logits=t_out.logits,
                        student_ce_loss=s_out.loss,
                        distillation_alignment_outputs = None,
                        attention_mask=kd_mask,
                        step=None
                    )

                # 应用梯度累积
                loss_for_backward = total_loss
                accelerator.backward(loss_for_backward)

                for k, v in loss_details.items():
                    if k == "kl_weight":
                        running_losses[k] = v
                    else:
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
                            loss=running_losses["total_loss"] ,
                            kl=running_losses['kl_loss'],
                            l2=running_losses['l2_loss'],
                            ce=running_losses['ce_loss'] ,
                            te_loss=t_out.loss.item(),
                            lr=scheduler.get_last_lr()[0],
                            kl_weight = running_losses['kl_weight'],
                            step= global_step
                        )
                        progress_bar.update(1)

                        # 4. 按 global_step 记录到TensorBoard
                        if global_step % 2 == 0:  # 您可以调整记录频率
                            tbwriter.add_scalar('train/total_loss', running_losses['total_loss'], global_step)
                            tbwriter.add_scalar('train/kl_loss', running_losses['kl_loss'] , global_step)
                            tbwriter.add_scalar('train/l2_loss', running_losses['l2_loss'] , global_step)
                            tbwriter.add_scalar('train/ce_loss', running_losses['ce_loss'] , global_step)
                            tbwriter.add_scalar('learning_rate', scheduler.get_last_lr()[0], global_step)
                    # 重置运行损失
                    running_losses = collections.defaultdict(float)

                    # 保存模型检查点
                    if cfg.save_steps and global_step % cfg.save_steps == 0:
                        path = Path(cfg.output) / f"step_{global_step}"
                        path_1 = Path(cfg.output) / f"step_{global_step}_1"
                        accelerator.save_state(path)

                    # if cfg.eval_steps and global_step % cfg.eval_steps == 0:
                    #     eval_loss = evaluate(student, eval_loader, accelerator)  # no val loader stub
                    #     if accelerator.is_main_process:
                    #         tbwriter.add_scalar('loss', eval_loss, global_step)

                if global_step >= total_update_steps:
                    break
            if global_step >= total_update_steps:
                break
    except Exception as e :
        logger.error(f"Training failed with exception: {e}")
        #save model
        path2 = Path(cfg.output) / f"step_{global_step}_error"
        accelerator.save_state(path2)
        raise e

    # =================================================================
    # --- 开始：修改后的存储逻辑 ---
    # =================================================================
    accelerator.wait_for_everyone()

    # 1. 保存完整的Accelerator状态，用于未来可能的“继续训练”
    # 这个状态包括了模型（可能是包装后的）、优化器、学习率调度器等。
    # 这是恢复训练的最佳方式。我们将其保存在一个明确的目录中。
    # 使用 try-except 来捕获由 shared_tensor 等问题引发的潜在存储错误。
    try:
        final_checkpoint_dir = Path(cfg.output) / "final_checkpoint"
        final_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        accelerator.print(f"Saving final accelerator state to {final_checkpoint_dir}...")
        accelerator.save_state(output_dir=final_checkpoint_dir)
        accelerator.save_model(student, output_dir=final_checkpoint_dir / "model")
        accelerator.print("Successfully saved accelerator state for resuming training.")
    except Exception as e:
        accelerator.print(f"Warning: Could not save the full accelerator state due to an error: {e}")
        accelerator.print("This may affect your ability to resume training from this point.")
        accelerator.print("Proceeding to save the final unwrapped model weights.")

    # 2. 保存最终的、可用于推理的模型
    # 这是与他人分享或部署模型的标准方式。
    # 使用 `unwrap_model` 来获取底层的、原始的Hugging Face模型。
    unwrapped_model = accelerator.unwrap_model(student)

    # 仅在主进程上执行保存操作，以避免多进程并发写入导致文件损坏
    if accelerator.is_main_process:
        final_model_dir = Path(cfg.output) / "final_model"
        final_model_dir.mkdir(parents=True, exist_ok=True)
        accelerator.print(f"Saving final unwrapped model for inference to {final_model_dir}...")

        # 使用 `save_pretrained` 是Hugging Face推荐的最稳健的保存方式。
        # 它能正确处理权重共享（tied weights / shared tensors）等复杂情况。
        # safe_serialization=True 会保存为 .safetensors 格式，更安全、加载更快。
        if hasattr(unwrapped_model, "save_pretrained"):
            unwrapped_model.save_pretrained(
                final_model_dir,
                safe_serialization=False
            )

        else:
            # 如果模型没有 save_pretrained 方法（非HF模型），则回退到保存 state_dict
            torch.save(unwrapped_model.state_dict(), final_model_dir / "pytorch_model.bin")

        # 保存 tokenizer，以便后续可以轻松加载模型进行推理
        tok.save_pretrained(final_model_dir)
        accelerator.print(f"Successfully saved final model and tokenizer to {final_model_dir}.")

    # =================================================================
    # --- 结束：修改后的存储逻辑 ---
    # =================================================================

    accelerator.end_training()

if __name__ == "__main__":
    # accelerate
    # launch - -num_processes
    # 4 - -mixed_precision
    # bf16 - -deepspeed_config_file
    # deepspeed_config.json
    # train.py
    main()
