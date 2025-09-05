import math

from dataset.cauldron import CauldronDataset,get_dataloader
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    hf_logging.set_verbosity_error()

    accelerator = Accelerator(log_with="tensorboard", project_dir=cfg.logging_dir)
    set_seed(cfg.seed)

    accelerator.wait_for_everyone()
    teacher_model, student_wrapper,student_model,tok,processor = build_model(cfg)


    train_loader = get_dataloader(cfg,processor)
    eval_loader = get_dataloader(cfg,processor)

    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum)
    max_steps = cfg.max_steps if cfg.max_steps > 0 else steps_per_epoch * cfg.num_epochs


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
                              num_training_steps=max_steps)
    # prepare
    logger.info("Prepare")
    teacher_model, student_model, optimizer, train_loader, scheduler = accelerator.prepare(
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

    logger.info("Generate")

    global_step = 0

    for step , batch in enumerate(train_loader):
        if global_step > 5:
            break
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
        print(f"__"*10, f"Step {step}", "__"*10)
        teacher_outputs = teacher_model.generate(**batch["input_ids"])
        decoded_text = processor.batch_decode(teacher_outputs, skip_special_tokens=True)[0]
        print(f"Teacher generated text: \"{decoded_text.strip()}\"")

        student_outputs = student_model.generate(**batch["input_ids"])
        decoded_text = processor.batch_decode(student_outputs, skip_special_tokens=True)[0]
        print(f"Student generated text: \"{decoded_text.strip()}\"")
        print("__"*10)
        global_step = global_step+1



if __name__ == "__main__":
    # accelerate
    # launch - -num_processes
    # 4 - -mixed_precision
    # bf16 - -deepspeed_config_file
    # deepspeed_config.json
    # train.py
    main()
