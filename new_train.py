import math
import os
os.environ['HF_ENDPOINT']="https://hf-mirror.com"
from dataset.sharegpt4v import make_supervised_data_module,DataArguments
from config.train_config import TrainConfig
from model.decoder_layer import HybridDecoderLayers
from model.mamba2.hybrid_mamba_config import PhiMambaConfig
from model.model_wrapper import HybridSmolVLMWrapper
from transformers import AutoProcessor, AutoTokenizer, logging as hf_logging, get_scheduler
from utils.loss import VLMDitillationLoss,DistillationLossConfig
import logging
logger = logging.getLogger(__name__)
from accelerate import Accelerator,DistributedDataParallelKwargs
from accelerate.utils import set_seed
import torch
from torch.utils.data import DataLoader

import argparse

def build_model(cfg:TrainConfig):
    # Prepare Teacher Model
    # teacher_model = HybridSmolVLMForConditionalGeneration.from_pretrained(cfg.teacher_name,torch_dtype=cfg.dtype)
    # for param in teacher_model.parameters():
    #     param.requires_grad = False
    # logger.log(logging.INFO, f"Teacher model {cfg.teacher_name} loaded successfully.")
    mamba_cfg = PhiMambaConfig(
        d_model=576,
        ssm_cfg={"expand": 1, "ngroups":9, "d_state": 64,"d_conv": 4},
        rms_norm_eps=1e-05,
        d_inner=576,
        d_xb=192,
        intermediate_size=1536,
        hidden_act="silu",
        n_layer=30,
        attn_layers=[1,2,3,4,5,7,8,9,10,11,13,14,15,16,17,19,20,21,22,23,24,25,26,27,28,29],
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
    # if cfg.resume_from_checkpoint:
    #     logger.info(f"resume model weights from : {cfg.check_point_path}")
    #     student_model.from_pretrained(cfg.check_point_path)
    #     logger.info("success")

    for param in student_model.parameters():
        param.requires_grad = False

    # 2. 然后，通过遍历模块来精确地解冻需要的子模块
    for module_name, module in student_model.named_modules():
        # 检查当前模块的类型是否是我们的目标自定义层
        if isinstance(module, HybridDecoderLayers):
            module.gradient_checkpointing = True

            print(f"✅ Found Hybrid Layer: '{module_name}'. Unfreezing its components.")
            for sub_module_name, sub_module in module.named_children():
                if sub_module_name == "self_attn_mamba" or sub_module_name == "mlp" or sub_module_name == "input_layernorm" or sub_module_name == "post_attention_layernorm":
                    print(f"  -- Unfreezing sub-module: '{sub_module_name}'")
                    for param in sub_module.parameters():
                        param.requires_grad = True
    student_model.gradient_checkpointing_enable()
    logger.info("Successfully load Student, Building tok")

    tok = AutoTokenizer.from_pretrained(cfg.teacher_name, local_files_only=True, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    logger.info("Processor")
    processor = AutoProcessor.from_pretrained(cfg.teacher_name, local_files_only=True)

    return student_wrapper,student_model,tok, processor

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
    accelerator = Accelerator(log_with="tensorboard",
                              project_dir=cfg.logging_dir,
                              gradient_accumulation_steps=cfg.grad_accum,
                              # kwargs_handlers=[ddp_kwargs]
                              )
    set_seed(cfg.seed)

    accelerator.wait_for_everyone()
    student_wrapper,student_model,tok,processor = build_model(cfg)
    if accelerator.is_main_process:
        report_path = "params_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:

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

    cfg.grad_accum = cfg.grad_accum // accelerator.num_processes

    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum)
    total_update_steps = steps_per_epoch * cfg.num_epochs

    loss_config = DistillationLossConfig(
        kl_weight=1.,
        l2_weight=1.0,
        ce_weight=0.0,
        temperature=1.0,
        l2_loss_layers=[0, 6, 12, 18]
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
    scheduler = get_scheduler("cosine", optimizer,
                              num_warmup_steps=cfg.warmup_steps,
                              num_training_steps=total_update_steps // cfg.num_epochs)
    # prepare
    logger.info("Prepare")
    student, train_loader,optimizer= accelerator.prepare(
        student_model, train_loader,optimizer)
    if cfg.resume_from_checkpoint:
        accelerator.load_state(cfg.check_point_path)
        logger.info("load state")

    tbwriter = None
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dict['train_dataset'])}")
    logger.info(f"  Num Epochs = {cfg.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {cfg.batch_size*accelerator.num_processes}")
    logger.info(f"  Gradient Accumulation steps = {cfg.grad_accum}")
    logger.info(f"  Total optimization steps = {total_update_steps}")

    global_step = 0
    import traceback
    from pathlib import Path
    from tqdm.auto import tqdm
    import collections
    try:
        if accelerator.is_main_process:
            from torch.utils.tensorboard import SummaryWriter
            tbwriter = SummaryWriter(log_dir=cfg.logging_dir)
        for epoch in range(cfg.num_epochs):
            student.train()
            progress_bar = tqdm(
                total=total_update_steps,
                desc=f"Epoch {epoch + 1}/{cfg.num_epochs}",
                disable=not accelerator.is_main_process,
                initial=global_step  # Make sure the progress bar starts from the correct step when resuming
            )
            running_loss = 0.0

            for step, batch in enumerate(train_loader):
                # The main training logic for a single batch goes here
                # ... (Your existing code for forward pass, loss calculation, etc.)
                batch = {k: (v.to(accelerator.device, non_blocking=True) if hasattr(v, "to") else v)
                         for k, v in batch.items()}
                # input_ids = batch["input_ids"]
                # labels = batch["labels"]
                # attention_mask = batch["attention_mask"]
                # images = batch["pixel_values"].to(cfg.dtype)
                #
                # kd_mask = (batch['labels'] != -100).to(torch.long)
                s_out = student(**batch, output_hidden_states=True)
                loss = s_out.loss
                loss_for_backward = loss / cfg.grad_accum
                accelerator.backward(loss_for_backward)

                running_loss+=loss.item()

                if (step + 1) % cfg.grad_accum == 0:
                    accelerator.clip_grad_norm_(student.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    progress_bar.update(1)  # Update progress bar on global step

                    # Logging, progress bar updates, etc.
                    if accelerator.is_main_process:

                    # 3. 更新进度条的后缀信息，动态显示损失和学习率

                        progress_bar.set_postfix(

                            ce=running_loss / global_step,

                            # z_loss=running_losses['z_loss']

                            lr=scheduler.get_last_lr()[0],

                            step=global_step

                        )

                    # 4. 按 global_step 记录到TensorBoard

                        if global_step % 2 == 0:  # 您可以调整记录频率


                            tbwriter.add_scalar('train/ce_loss',running_loss / global_step, global_step)

                            tbwriter.add_scalar('learning_rate', scheduler.get_last_lr()[0], global_step)

                    # Regular checkpoint saving
                    if cfg.save_steps and global_step % cfg.save_steps == 0:
                        path = Path(cfg.output) / f"step_{global_step}"
                        accelerator.save_state(path)
                        # Optional: save pretrained version as well
                        unwrapped_model = accelerator.unwrap_model(student)
                        if accelerator.is_main_process:
                            unwrapped_model.save_pretrained(path)

                if global_step >= total_update_steps:
                    break
            if global_step >= total_update_steps:
                break

    except Exception as e:
        # This block will execute if any error occurs inside the `try` block
        if accelerator.is_main_process:
            # 1. Log the full error traceback for easier debugging
            logger.error("An unexpected error occurred during training:")
            logger.error(traceback.format_exc())

            # 2. Create a path for the emergency checkpoint
            error_path = Path(cfg.output) / f"error_checkpoint_step_{global_step}"
            logger.info(f"Attempting to save an emergency checkpoint to: {error_path}")

            # 3. Use accelerator.save_state for a robust save
            # This saves the model, optimizer, scheduler, and RNG states
            accelerator.save_state(error_path)

            logger.info("Emergency checkpoint saved successfully.")

        # 4. Re-raise the exception to ensure the script stops and reports the failure
        raise

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
