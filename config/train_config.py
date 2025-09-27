from dataclasses import asdict, dataclass, field
from typing import Optional, List
import torch


# ----------------------------------
# 1. Configuration
# ----------------------------------
@dataclass
class TrainConfig:
    # I/O                                  # 仅 JSONL 数据时需要；Cauldron 可忽略
    output: str = "/home/jinkaiyan/outputs/0926"
    logging_dir: Optional[str] = "/home/jinkaiyan/outputs/0926/logs"

    # Dataset meta
    dataset: str = "HuggingFaceM4/the_cauldron"     # ["cauldron", "jsonl"]
    subset:  str = "vqav2"
    split:   str = "train"        # HuggingFace Datasets split argument
    streaming: bool = False       # True → iterable dataset
    num_workers: int = 16

    # Optimisation
    num_epochs: int = 1
    lr_scheduler_type: str = "linear"  # ["linear", "cosine", "cosine_w_restarts", "polynomial", "constant", "constant_with_warmup", "warmup_stable_decay"]
    stable_steps: int = 10000
    lr: float = 2e-4
    weight_decay: float = 0.001
    batch_size: int = 1
    grad_accum: int = 8
    warmup_steps: int = 783
    decay_steps: int = 10000  # if >0, overrides num_epochs
    max_steps: int = 20783
    seed: int = 141  # 42
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999

    # Distillation
    temperature: float = 1.0
    ce_weight: float = 1.0
    kl_weight: float = 1.0
    itc_weight: float = 0.0  # image‑text contrastive
    mse_weight: float = 0.0  # hidden‑state alignment

    # Model specifics
    teacher_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct"  # Teacher model name
    check_point_path: str = "/home/jinkaiyan/outputs/0913/final_checkpoint"
    prev_model_path : str = "/home/jinkaiyan/outputs/0926/final_model"
    resume_model:bool = False
    resume_from_checkpoint : bool = False
    attn_layers: List[int] = field(default_factory=lambda: [0,4,8,12,16])
    dtype = torch.bfloat16
    bf16: bool = True
    grad_checkpoint: bool = True

    # Evaluation / logging
    eval_steps: int = 250
    save_steps: int = 8000
    log_dir: str = "runs"

    def to_cmd(self) -> List[str]:
        return [f"--{k} {v}" for k, v in asdict(self).items() if v is not None]


