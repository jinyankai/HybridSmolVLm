"""train_sVLM.py
A lightweight, production‑ready training script that distills knowledge from
`HuggingFaceTB/SmolVLM-Instruct` into a custom student SmolVLM wrapped with
Mamba‑2 layers. Designed for 4 × RTX 3090 GPUs using 🤗 Accelerate.

Key features
------------
* 🔧  Dataclass `TrainConfig` for all tunables (CLI via `argparse`).
* 🧠  `build_models()` loads teacher & creates student with `Smollm_custom`.
* 📦  `get_dataloaders()` supports JSONL & WebDataset image‑text pairs.
* 🏋️  Mixed‑precision (bf16), gradient accumulation, gradient checkpointing.
* 🔥  Distillation loss: CE + KL (teacher logits) + optional ITC & hidden MSE.
* 📈  TensorBoard logging, tqdm progress bars, deterministic seeding.
* 💾  Auto checkpoint save/resume & periodic evaluation.

Run with
```
accelerate launch --multi_gpu train_sVLM.py \
  --data "data/train.jsonl" --output ./checkpoints/svlm_distill
```
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torchvision import transforms as T

from accelerate import Accelerator, DistributedType, PartialState, infer_auto_device_map
from accelerate.utils import set_seed
from transformers import (AutoConfig, AutoTokenizer, get_scheduler,
                          PreTrainedTokenizer, logging as hf_logging)
from transformers import AutoModelForVision2Seq
from transformers import AutoProcessor
from datasets import load_dataset

from dataset import *

# # —————————————————— smolVLM specific imports ————————————————————
# from smolvlm.modeling_smolvlm import SmolVLMForConditionalGeneration  # vision‑text model
# from mamba2.wrapper import Smollm_custom, PhiMambaConfig  # provided by user env
from mamba2.hybrid_mamba_config import PhiMambaConfig
from mamba2.hybrid_model import  PhiMambaDecoderLayer  # type: ignore
from model.model_wrapper import *
from typing import Optional

logger = logging.getLogger(__name__)

# ----------------------------------
# 1. Configuration
# ----------------------------------
@dataclass
class TrainConfig:
    # I/O                                  # 仅 JSONL 数据时需要；Cauldron 可忽略
    output: str = "/home/jinkaiyan/MaTVLM/smolVLM/outputs"
    logging_dir: Optional[str] = "/home/jinkaiyan/MaTVLM/smolVLM/outputs/logs"

    # Dataset meta
    dataset: str = "cauldron"     # ["cauldron", "jsonl"]
    subset:  str = "vqav2"
    split:   str = "train"        # HuggingFace Datasets split argument
    streaming: bool = False       # True → iterable dataset
    num_workers: int = 8

    # Optimisation
    num_epochs: int = 3
    lr: float = 5e-5
    weight_decay: float = 0.05
    batch_size: int = 4
    grad_accum: int = 4
    warmup_steps: int = 1000
    max_steps: int = -1  # –1 => derived from epochs
    seed: int = 42

    # Distillation
    temperature: float = 1.0
    ce_weight: float = 1.0
    kl_weight: float = 1.0
    itc_weight: float = 0.0  # image‑text contrastive
    mse_weight: float = 0.0  # hidden‑state alignment

    # Model specifics
    teacher_name: str = "/home/jinkaiyan/MaTVLM/smolVLM/SmolVLM-Intruct"
    ssm_layers: List[int] = field(default_factory=lambda: [0, 4, 8, 12, 16, 20])
    bf16: bool = True
    grad_checkpoint: bool = True

    # Evaluation / logging
    eval_steps: int = 250
    save_steps: int = 2500
    log_dir: str = "runs"

    def to_cmd(self) -> List[str]:
        return [f"--{k} {v}" for k, v in asdict(self).items() if v is not None]




# ----------------------------------
# 3. Build models & losses
# ----------------------------------

def build_models(cfg: TrainConfig):
    dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    logger.info("Loading teacher %s", cfg.teacher_name)
    teacher = AutoModelForVision2Seq.from_pretrained(cfg.teacher_name,local_files_only=True, torch_dtype=dtype)
    teacher.eval().requires_grad_(False)

    logger.info("Building student via Smollm_custom wrapper")
    # base_cfg = AutoConfig.from_pretrained(cfg.teacher_name)
    # d_state = base_cfg.hidden_size // base_cfg.num_attention_heads
    phi_cfg = PhiMambaConfig(
        2048,
        {"expand": 1, "ngroups":32, "d_state": 64,"d_conv": 1},
        1e-05,
        d_inner=2048,
        d_xb=2048,
        intermediate_size=8192,
        hidden_act="silu",
        n_layer=24,
        attn_layers=[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,21,22,23],
        resid_pdrop=0.1,
        bidirectional=False,
        is_bias=False
    )
    base_model = AutoModelForVision2Seq.from_pretrained(cfg.teacher_name,local_files_only=True, torch_dtype=dtype)
    if cfg.grad_checkpoint:
        base_model.gradient_checkpointing_enable()
    ssm_layers = [0, 4, 8, 12, 16, 20]
    student = Smollm_custom(
        transformer_model=base_model,
        target_id=ssm_layers,
        use_bias=False,
        mamba_config=phi_cfg,
        checkpoint_path="",
        dtype=dtype,
    )
    for name, param in student.named_parameters():
        if f"mamba" not in name:
            param.requires_grad = False
    for name, param in student.named_parameters():
        if param.requires_grad:
            print(f"Parameter: {name}, requires_grad: {param.requires_grad}")
    student.save_config(cfg.output)
    logger.info("Building tok")

    tok = AutoTokenizer.from_pretrained(cfg.teacher_name,local_files_only=True, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    logger.info("Processor")
    processor = AutoProcessor.from_pretrained(cfg.teacher_name,local_files=True)

    return teacher, student, tok, processor


# ----------------------------------
# 4. Distillation utilities
# ----------------------------------

def distil_loss(cfg: TrainConfig, teacher_logits, student_logits, student_ce, hidden_t=None, hidden_s=None):
    loss = cfg.ce_weight * student_ce
    if cfg.kl_weight:
        T = cfg.temperature
        kl = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T ** 2)
        loss += cfg.kl_weight * kl
    if cfg.mse_weight and hidden_t is not None:
        l2 = F.mse_loss(hidden_s[-1], hidden_t[-1])
        loss += cfg.mse_weight * l2
    return loss


# ----------------------------------
# 5. Evaluation stub
# ----------------------------------

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


# ----------------------------------
# 6. Main
# ----------------------------------

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

    accelerator = Accelerator(mixed_precision="bf16" if cfg.bf16 else "fp16", log_with="tensorboard",project_dir= cfg.logging_dir)
    # if accelerator.is_main_process:
    #     (Path(cfg.output) / "cfg.json").parent.mkdir(parents=True, exist_ok=True)
    #     json.dump(asdict(cfg), open(Path(cfg.output) / "cfg.json", "w"), indent=2)
    set_seed(cfg.seed)

    teacher, student, tok,processor = build_models(cfg)
    logger.info("Loading data")

    train_loader= get_dataloader(cfg,processor)
    # total steps
    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum)
    max_steps = cfg.max_steps if cfg.max_steps > 0 else steps_per_epoch * cfg.num_epochs

    # Optimiser & scheduler
    logger.info("Optimizer & Scheduler Setup")
    no_decay = {"bias", "LayerNorm.weight"}
    optim_groups = [
        {"params": [p for n, p in student.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
         "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in student.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=cfg.lr)
    scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=max_steps)

    # prepare
    logger.info("Prepare")
    teacher, student, optimizer, train_loader, scheduler = accelerator.prepare(
        teacher, student, optimizer, train_loader, scheduler)

    tb = accelerator.get_tracker("tensorboard")
    global_step = 0
    teacher.eval()
    logger.info("Train")
    from contextlib import nullcontext
    scaler_ctx = accelerator.autocast() if cfg.bf16 or accelerator.mixed_precision == "fp16" else nullcontext()
    for epoch in range(cfg.num_epochs):
        for step, batch in enumerate(train_loader):
            student.train()
            ids      = batch["input_ids"].to(accelerator.device, non_blocking=True)
            attn     = batch["attention_mask"].to(accelerator.device, non_blocking=True)
            labels   = batch["labels"].to(accelerator.device, non_blocking=True)
            images   = batch["pixel_values"].to(
                            accelerator.device,
                            dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
                            non_blocking=True,
                        )

            batch = {
                "input_ids": ids,            # still int64
                "attention_mask": attn,      # int64
                "labels": labels,            # int64
                "pixel_values": images,      # bf16 / fp16
            }
            if step == 0 and accelerator.is_main_process:
                print("dtypes:", batch["input_ids"].dtype, batch["pixel_values"].dtype)
                print(tok.batch_decode(batch["input_ids"][0][:20]))
                assert batch["input_ids"].dtype in (torch.int64, torch.int32)
            with scaler_ctx:
                with torch.no_grad():
                    t_out = teacher(**batch, output_hidden_states=True)
                s_out = student(**batch, output_hidden_states=True)
                loss = distil_loss(cfg, t_out.logits, s_out.logits, s_out.loss,
                                   hidden_t=t_out.hidden_states, hidden_s=s_out.hidden_states)
                loss = loss / cfg.grad_accum
            accelerator.backward(loss)

            if (step + 1) % cfg.grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if accelerator.is_main_process and global_step % 10 == 0:
                    tb.writer.add_scalar("train/loss", loss.item() * cfg.grad_accum, global_step)
                    tb.writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)

                if cfg.save_steps and global_step % cfg.save_steps == 0:
                    path = Path(cfg.output) / f"step_{global_step}"
                    accelerator.save_state(path)

                if cfg.eval_steps and global_step % cfg.eval_steps == 0:
                    eval_loss = evaluate(student, None, accelerator)  # no val loader stub
                    if accelerator.is_main_process:
                        tb.add_scalar("eval/loss", eval_loss, global_step)

            if global_step >= max_steps:
                break
        if global_step >= max_steps:
            break

    # final save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        student.unwrap().save_pretrained(Path(cfg.output) / "final")
        tok.save_pretrained(Path(cfg.output) / "final")
    accelerator.end_training()


if __name__ == "__main__":
    main()
