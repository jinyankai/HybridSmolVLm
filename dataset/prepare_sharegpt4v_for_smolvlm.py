#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ShareGPT4V → SmolVLM 训练用数据处理脚本（模板）
------------------------------------------------
适配数据集：Lin-Chen/ShareGPT4V-ShareGPT4V-PT
目标模型：HuggingFaceTB/SmolVLM-Instruct（或兼容的 SmolVLM）

功能：
1) 从 HF 加载数据集，抽取 (human → gpt) 配对，多轮对话拆成多条样本
2) 统一模板：<|system|>/<|user|>/<|assistant|> + <image> 占位符
3) 将 image 列转为 datasets.Image()，返回 PIL.Image 对象，避免相对路径问题
4) 生成可直接喂给 processor 的字段：{"text": "...", "images": PIL.Image.Image}
5) 提供 DataCollator 可与 AutoProcessor 配合：processor(images=..., text=..., return_tensors="pt")
6) 可保存处理后的数据集到磁盘；也可快速做一个 batch 的干跑（dry-run）检查

使用：
python prepare_sharegpt4v_for_smolvlm.py \
  --dataset "Lin-Chen/ShareGPT4V-ShareGPT4V-PT" \
  --split "train" \
  --out_dir "./proc_sharegpt4v_smolvlm" \
  --max_samples -1 \
  --seed 42 \
  --do_save \
  --dry_run
"""
from __future__ import annotations

import argparse
from typing import List, Dict, Any, Tuple
import datasets
from datasets import load_dataset
from PIL import Image

# 可根据需要替换
SYSTEM_PROMPT = "You are a helpful, detail‑oriented vision-language assistant."
USER_TAG = "<|user|>"
ASSISTANT_TAG = "<|assistant|>"
SYSTEM_TAG = "<|system|>"
END_TAG = "<|end|>"
IMAGE_PLACEHOLDER = "<image>"  # SmolVLM 的 processor 会识别并展开为 image_seq_len 个 token

TEMPLATE = (
    f"{SYSTEM_TAG} {{system}}\n{END_TAG}\n"
    f"{USER_TAG}\n{{user}}\n{END_TAG}\n"
    f"{ASSISTANT_TAG}\n{{assistant}}\n{END_TAG}"
)

def build_samples_from_conversations(example: Dict[str, Any]) -> Dict[str, List[Any]]:
    """
    输入一条原始样本：
      {
        'id': 'sa_243527',
        'image': <PIL.Image 或 路径>,
        'conversations': [{'from':'human','value':'<image>...'}, {'from':'gpt','value':'...'}, ...]
      }
    输出多条样本（与每个 human->gpt 配对对应）：
      {
        'id': [...],
        'text': [...],   # 已套用模板
        'images': [...]  # PIL.Image
      }
    """
    cid = example.get("id", "")
    image = example["image"]  # 已在主流程中 cast 为 datasets.Image()，此处为字典或 PIL.Image
    # datasets.Image() 返回一个 dict: {"path", "bytes", "array" ...}；.convert("RGB") 需要 PIL.Image 对象
    if isinstance(image, dict) and "bytes" in image:
        image = Image.open(datasets.filesystems.tempfiles.BytesIO(image["bytes"])).convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        # 兜底：可能是字符串路径
        image = Image.open(image).convert("RGB")

    convs: List[Dict[str, str]] = example.get("conversations", [])
    # 将多轮对话拆成多条 (human → gpt) 样本
    out_ids, out_texts, out_images = [], [], []
    system = SYSTEM_PROMPT

    i = 0
    while i < len(convs) - 1:
        human = convs[i]
        bot = convs[i + 1]
        if human.get("from") == "human" and bot.get("from") in ("gpt", "assistant"):
            user_text = human.get("value", "").strip()
            # 规范化：确保至少一个 <image> 占位符；若发现多个 <image>，保留第一个，其余删掉，避免与 processor 期望不一致
            # （SmolVLM 一般是一条样本对应一张图；若你要多图，请改为将 images 列改成 List[PIL] 并在 collator 里传入嵌套结构）
            num_img_tokens = user_text.count(IMAGE_PLACEHOLDER)
            if num_img_tokens == 0:
                user_text = IMAGE_PLACEHOLDER + "\n" + user_text
            elif num_img_tokens > 1:
                # 仅保留第一个，其他去掉，避免 image_token 数不匹配导致 "not divisible by patch_size" 异常
                parts = user_text.split(IMAGE_PLACEHOLDER)
                user_text = IMAGE_PLACEHOLDER + "".join(parts[1:])

            assistant_text = bot.get("value", "").strip()

            text = TEMPLATE.format(system=system, user=user_text, assistant=assistant_text)
            out_ids.append(f"{cid}__turn{i//2}")
            out_texts.append(text)
            out_images.append(image)
            i += 2
        else:
            i += 1

    if not out_texts:
        # 没有配对成功就跳过
        return {"id": [], "text": [], "images": []}
    return {"id": out_ids, "text": out_texts, "images": out_images}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Lin-Chen/ShareGPT4V-ShareGPT4V-PT")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--out_dir", type=str, default="./proc_sharegpt4v_smolvlm")
    parser.add_argument("--max_samples", type=int, default=-1, help="-1 表示使用全部数据")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--do_save", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="做一个 processor 干跑检查（需要 transformers 已安装）")
    args = parser.parse_args()

    datasets.logging.set_verbosity_info()
    ds = load_dataset(args.dataset, split=args.split)

    # 将 'image' 列强制转换为 datasets.Image()，避免路径解析麻烦
    if "image" in ds.column_names:
        ds = ds.cast_column("image", datasets.Image())

    # map 成训练样本结构
    proc = ds.map(
        build_samples_from_conversations,
        remove_columns=ds.column_names,
        batched=False,
        desc="Converting ShareGPT4V conversations → SmolVLM samples",
    )

    # 过滤掉空样本
    proc = proc.filter(lambda x: len(x["text"]) > 0)

    # 随机打乱 & 采样
    if args.max_samples and args.max_samples > 0:
        proc = proc.shuffle(seed=args.seed).select(range(min(args.max_samples, len(proc))))

    print(proc)


    if args.do_save:
        proc.save_to_disk(args.out_dir)
        print(f"✅ Saved processed dataset to: {args.out_dir}")

    if args.dry_run:
        try:
            from transformers import AutoProcessor
            import torch

            processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")

            def simple_collate(features: List[Dict[str, Any]]) -> Dict[str, Any]:
                texts = [f["text"] for f in features]
                imgs = [f["images"] for f in features]  # 单图场景
                enc = processor(images=imgs, text=texts, return_tensors="pt", padding=True)
                # 语言模型训练通常需要 labels，这里简单做：labels = input_ids（如果需要仅学习 assistant，可自行基于 tag 做 mask）
                enc["labels"] = enc["input_ids"].clone()
                return enc

            batch = simple_collate([proc[i] for i in range(min(2, len(proc)))])
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(k, v.shape)
                else:
                    print(k, type(v))
            print("✅ Dry-run passed: processor 能正常对齐 <image> 与像素张量。")
        except Exception as e:
            print("⚠️ Dry-run failed:", e)


if __name__ == "__main__":
    main()
