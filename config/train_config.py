from dataclasses import asdict, dataclass, field
from typing import Optional, List
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


