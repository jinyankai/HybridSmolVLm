from __future__ import annotations

"""Cauldron → SmolVLM dataset (**single-image version, variable-length text**)
robust to any extra dims returned by the processor.

Output guarantees
-----------------
* **pixel_values** → `(3, H, W)` – we recursively slice the very first
  element along leading dimensions until only 3 dims remain.
* **input_ids / attention_mask** → padded in `collate_fn`.
* **pixel_values** is unsqueezed to `(B, 1, 3, H, W)` inside
  `collate_fn` to match SmolVLM’s expected shape.
"""

from typing import Any, Dict, List

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import PreTrainedTokenizer, ProcessorMixin
import os
os.environ.pop("HF_DATASETS_OFFLINE", None)
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ['HF_ENDPOINT']="https://hf-mirror.com"

# ────────────────────────────────────────────────────────────────────────────────
# helper
# ────────────────────────────────────────────────────────────────────────────────

def _build_prompt(texts: List[str | Dict[str, str]], image_tok: str = "<image>") -> str:
    if not texts:
        return image_tok
    if isinstance(texts[0], str):
        user = "\n".join(texts)
    else:
        user = "\n".join(d.get("text", "") for d in texts)
    return f"{image_tok}\n{user}"




# ────────────────────────────────────────────────────────────────────────────────
# dataset
# ────────────────────────────────────────────────────────────────────────────────
class CauldronDataset(Dataset):


    def __init__(
            self,
            processor: ProcessorMixin,
            subset: str = "vqav2",
            split: str = "train",
            streaming: bool = False,
    ) -> None:
        super().__init__()
        self.ds = load_dataset(
            "HuggingFaceM4/the_cauldron",
            subset,
            split=split,
            streaming=streaming,
            num_proc=None if streaming else 8,
            cache_dir="/home/jinkaiyan/.cache/huggingface/datasets/"
        )
        if not hasattr(processor, "tokenizer"):
            raise ValueError("`processor` must expose .tokenizer")
        self.processor: ProcessorMixin = processor
        self.tok: PreTrainedTokenizer = processor.tokenizer
        self.streaming = streaming
        if not streaming:
            self._len = len(self.ds)

    # ---------------------------------------------------------------------
    def __len__(self):
        if self.streaming:
            raise TypeError("Streaming dataset has no static length.")
        return self._len

    # ---------------------------------------------------------------------
    def _process(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        img = sample["images"]  # first image only
        prompt = _build_prompt(sample["texts"])

        enc = self.processor(
            images=img,
            text=prompt,
            return_tensors="pt",
        )
        # pix = enc["pixel_value"]
        # remove first d

        out: Dict[str, torch.Tensor] = {}
        for k, v in enc.items():
            out[k] = v.squeeze(0) if v.ndim > 1 and v.size(0) == 1 else v
        return out

    # ---------------- map / iterable wrappers ----------------
    def __getitem__(self, idx: int):
        return self._process(self.ds[idx])

    def __iter__(self):
        for s in self.ds:
            yield self._process(s)


# ────────────────────────────────────────────────────────────────────────────────
# collate_fn (text padding only)
# ────────────────────────────────────────────────────────────────────────────────

def collate_fn(batch: List[Dict[str, torch.Tensor]], tok: PreTrainedTokenizer):
    ids = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=tok.pad_token_id
    ).long()
    am = torch.nn.utils.rnn.pad_sequence(
        [b["attention_mask"] for b in batch], batch_first=True, padding_value=0
    ).long()

    imgs = pad_pixel_values(batch)
    # print("img shape",imgs.shape) # (B,3,H,W)
    # imgs = imgs.unsqueeze(1)                                # (B,1,3,H,W)

    return {
        "input_ids": ids,
        "attention_mask": am,
        "pixel_values": imgs,
        "labels": ids.clone(),
    }


# ────────────────────────────────────────────────────────────────────────────────
# DataLoader helper
# ────────────────────────────────────────────────────────────────────────────────
class TrainConfig:
    dataset: str = "cauldron"
    subset: str = "vqav2"
    split: str = "train"
    streaming: bool = False
    batch_size: int = 32
    num_workers: int = 4


def get_dataloader(cfg: TrainConfig, processor: ProcessorMixin) -> DataLoader:
    if "cauldron" not in cfg.dataset.lower():
        raise ValueError("Only 'cauldron' handled here.")

    ds = CauldronDataset(
        processor=processor,
        subset=cfg.subset,
        split=cfg.split,
        streaming=cfg.streaming,
    )
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=not isinstance(ds, IterableDataset),
        collate_fn=lambda b: collate_fn(b, ds.tok),
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )


def pad_pixel_values(batch_pixel_values):
    # 找到batch中最大的nums_img
    max_nums_img = max([pv["pixel_values"].shape[0] for pv in batch_pixel_values])

    padded_batch = []
    for pixel_values in batch_pixel_values:

        # 当前样本的nums_img
        current_nums_img = pixel_values["pixel_values"].shape[0]
        # 计算需要填充的数量
        padding_needed = max_nums_img - current_nums_img
        new_ = pixel_values["pixel_values"].unsqueeze(0)
        # print("pix shape",new_.shape)

        if padding_needed > 0:
            # 创建填充张量，形状为[1, padding_needed, C, H, W]，值全为0
            padding = torch.zeros(
                (1, padding_needed) + new_.shape[2:],
                dtype=pixel_values["pixel_values"].dtype,
                device=pixel_values["pixel_values"].device
            )
            # print("pad",padding.shape)
            # 将原始像素值和填充部分连接起来
            padded_pv = torch.cat([new_, padding], dim=1)
            # print("padded", padded_pv.shape)
            padded_batch.append(padded_pv)
        else:
            # 如果不需要填充，直接使用原始像素值
            new_ = pixel_values["pixel_values"].unsqueeze(0)
            padded_batch.append(new_)

    # 将填充后的样本堆叠成一个张量
    return torch.cat(padded_batch, dim=0)


def download_cauldron_dataset():
    """Download the Cauldron dataset to the local cache."""
    load_dataset("HuggingFaceM4/the_cauldron", "vqav2", split="train", streaming=False, num_proc=8)
    print("Cauldron dataset downloaded and cached.")