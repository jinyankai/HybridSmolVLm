
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
load_from_disk 数据集体检脚本（带丰富调试输出）

功能：
1) 直接使用 `datasets.load_from_disk(<data_path>)` 读取你的真实数据集目录。
2) 详细打印：对象类型、可用 split、features、样本数量、字段分布、对话结构、图片路径解析与存在性检查、异常示例等。
3) （可选）与你改过的 `sharegpt4v.py` 做集成验证：
   - 通过 `--sharegpt4v_file` 和 `--params_file` 提供源码文件路径，脚本动态 import：
       * 从 `sharegpt4v.py` 中获取 `LazySupervisedDataset`
       * 从 `params.py` 中获取 `DataArguments`
   - 只实例化不 __getitem__（除非你显式指定 --run_getitem），用于验证“磁盘目录路径 + load_from_disk”打通。

使用示例：
  基本体检（强烈推荐先跑一次）：
    python test_sharegpt4v_load_from_disk.py --data_path /path/to/ds --image_folder /path/to/images --max_show 3 --scan_limit 500

  叠加 sharegpt4v 集成测试（仅验证初始化）：
    python test_sharegpt4v_load_from_disk.py --data_path /path/to/ds \
        --image_folder /path/to/images \
        --sharegpt4v_file /your/repo/sharegpt4v.py \
        --params_file /your/repo/params.py

  如需进一步触发 __getitem__（需要可用的 processor；若没有可用，使用脚本内 DummyProcessor 仅做通路检测）：
    python test_sharegpt4v_load_from_disk.py ... --run_getitem

注意：
- 本脚本不会修改数据，仅做读取和统计。
- 若你的数据集比较大，可通过 --scan_limit 限制扫描条数，加快存在性检查。
"""

import argparse
import os
import sys
import json
import time
import types
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_from_disk, Dataset, DatasetDict, Features, Value, Sequence

# ========== 实用函数 ==========

def human_int(n: int) -> str:
    s = f"{n:,}"
    return s

def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def safe_head(x, k=3):
    return x[:k] if isinstance(x, list) else x

def print_kv(title: str, kv: Dict[str, Any], prefix="  "):
    print(f"\n[{ts()}] {title}")
    for k, v in kv.items():
        print(f"{prefix}{k}: {v}")

def dynamic_import_from_file(file_path: str, obj_name: Optional[str] = None):
    """
    动态 import 一个 .py 文件；若 obj_name 给定，则返回该对象；否则返回模块。
    """
    p = Path(file_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    mod_name = p.stem + "_dynimp"
    spec = importlib.util.spec_from_file_location(mod_name, p.as_posix())
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create spec for {p}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)  # type: ignore
    if obj_name:
        if not hasattr(module, obj_name):
            raise AttributeError(f"{p.name} does not contain object: {obj_name}")
        return getattr(module, obj_name)
    return module

# ========== Dummy Processor（仅用于通路测试） ==========

class DummyProcessor:
    """
    极简占位，避免 __init__ 报错。__getitem__ 时若被调用，会返回最小结构。
    你的真实 processor 到手后，可通过 --no_dummy_processor 禁用，并传入真实 processor 的导入方法进行替换。
    """
    def __call__(self, images=None, text=None, return_tensors="pt"):
        import torch
        # 仅返回最小必要字段，占位测试
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        pixel_values = torch.zeros((1, 1, 3, 32, 32), dtype=torch.float32)
        pixel_attention_mask = torch.ones((1, 1, 32, 32), dtype=torch.bool)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "pixel_attention_mask": pixel_attention_mask,
        }

# ========== 主体逻辑 ==========

def inspect_dataset(ds_obj, max_show: int = 3):
    """
    打印 Dataset / DatasetDict 的基本结构信息，并返回选定的 Dataset 对象
    """
    print_kv("对象类型与概览", {
        "type": type(ds_obj).__name__,
        "repr": repr(ds_obj)[:200] + ("..." if len(repr(ds_obj)) > 200 else "")
    })

    if isinstance(ds_obj, DatasetDict):
        keys = list(ds_obj.keys())
        print_kv("DatasetDict splits", {"splits": keys})
        choose = "train" if "train" in ds_obj else keys[0]
        print(f"[{ts()}] 选用 split: {choose}")
        ds = ds_obj[choose]
    else:
        ds = ds_obj

    # Features
    feats: Features = ds.features
    print_kv("Features", {name: str(dtype) for name, dtype in feats.items()})

    # 规模
    n = len(ds)
    print_kv("规模", {"num_rows": human_int(n)})

    # 展示前若干条原始字典
    for i in range(min(max_show, n)):
        row = ds[i]
        # 避免打印过长
        sample = {
            "id": row.get("id", None),
            "image": row.get("image", None),
            "conversations_head": safe_head(row.get("conversations", []), 2),
        }
        print(f"\n[{ts()}] 样本[{i}] 预览：")
        print(json.dumps(sample, ensure_ascii=False, indent=2))
    return ds

def check_schema_and_stats(ds: Dataset, image_folder: Optional[Path], scan_limit: int = 500):
    """
    检查关键字段、对话结构、图片路径存在性，并统计概况
    """
    n = len(ds)
    limit = min(scan_limit, n)

    num_image_str = num_image_list = 0
    num_abs = num_rel = 0
    num_missing = 0
    missing_examples: List[Tuple[int, Any]] = []
    conv_ok = conv_bad = 0

    for i in range(limit):
        row = ds[i]
        img = row.get("image", None)
        # image 字段类型
        if isinstance(img, str):
            num_image_str += 1
            p = Path(img)
            is_abs = p.is_absolute()
            if is_abs:
                num_abs += 1
                resolved = p
            else:
                num_rel += 1
                resolved = (image_folder / p) if image_folder else p
            if not resolved.exists():
                num_missing += 1
                if len(missing_examples) < 10:
                    missing_examples.append((i, str(resolved)))
        elif isinstance(img, list):
            num_image_list += 1
            # 仅检查第一张
            if img:
                p0 = Path(img[0])
                is_abs = p0.is_absolute()
                if is_abs:
                    num_abs += 1
                    resolved = p0
                else:
                    num_rel += 1
                    resolved = (image_folder / p0) if image_folder else p0
                if not resolved.exists():
                    num_missing += 1
                    if len(missing_examples) < 10:
                        missing_examples.append((i, str(resolved)))
        else:
            # 未知结构
            pass

        # conversations 结构检查
        conv = row.get("conversations", None)
        ok = isinstance(conv, list) and all(isinstance(x, dict) and "from" in x and "value" in x for x in (conv or []))
        if ok:
            conv_ok += 1
        else:
            conv_bad += 1

    print_kv("字段统计（基于扫描子集）", {
        "scanned_rows": limit,
        "image as str": num_image_str,
        "image as list[str]": num_image_list,
        "absolute paths": num_abs,
        "relative paths": num_rel,
        "missing image files": num_missing,
        "conversations OK": conv_ok,
        "conversations BAD": conv_bad,
    })

    if num_missing > 0:
        print(f"\n[{ts()}] 缺失图片样本示例（最多展示 10 条）：")
        for idx, miss_path in missing_examples:
            print(f"  - row {idx}: {miss_path}")

def maybe_test_sharegpt4v_integration(ds_dir: Path,
                                      image_folder: Optional[Path],
                                      sharegpt4v_file: Optional[Path],
                                      params_file: Optional[Path],
                                      run_getitem: bool,
                                      index: int):
    """
    动态加载 sharegpt4v.LazySupervisedDataset 与 params.DataArguments，验证初始化与（可选）getitem
    """
    if not sharegpt4v_file or not params_file:
        print(f"\n[{ts()}] 未提供 --sharegpt4v_file / --params_file，跳过 sharegpt4v 集成测试。")
        return

    print(f"\n[{ts()}] 尝试动态导入：{sharegpt4v_file}")
    LazySupervisedDataset = dynamic_import_from_file(str(sharegpt4v_file), "LazySupervisedDataset")
    print(f"[{ts()}] 尝试动态导入：{params_file}")
    DataArguments = dynamic_import_from_file(str(params_file), "DataArguments")

    # 准备 DataArguments
    da = DataArguments(
        data_path=str(ds_dir),
        image_folder=str(image_folder) if image_folder else None,
    )

    # 准备一个 DummyProcessor（仅打通流程）
    proc = DummyProcessor()

    # 实例化数据集
    print(f"[{ts()}] 实例化 LazySupervisedDataset（仅初始化，不触发 __getitem__）")
    ds = LazySupervisedDataset(
        data_path=da.data_path,
        processor=proc,
        data_args=da,
        padding=True,
    )
    print_kv("LazySupervisedDataset 初始化成功", {
        "len": len(ds),
        "type": type(ds).__name__,
        "has_list_data_dict": hasattr(ds, "list_data_dict"),
    })

    # 可选：触发 __getitem__
    if run_getitem and len(ds) > 0:
        print(f"\n[{ts()}] 触发 __getitem__({index}) 以做最小通路测试（DummyProcessor 占位）")
        try:
            item = ds[index]
            if isinstance(item, dict):
                # 打印关键 keys 与张量形状
                shapes = {}
                for k, v in item.items():
                    try:
                        import torch
                        if isinstance(v, torch.Tensor):
                            shapes[k] = tuple(v.shape)
                        else:
                            shapes[k] = type(v).__name__
                    except Exception:
                        shapes[k] = type(v).__name__
                print_kv("__getitem__ 返回结构", shapes)
            else:
                print_kv("__getitem__ 返回非 dict", {"type": type(item).__name__})
        except Exception as e:
            print(f"[{ts()}] __getitem__ 失败（通常是因为真实 processor 逻辑与占位不匹配）：{repr(e)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True, help="datasets.save_to_disk 生成的数据集目录")
    ap.add_argument("--image_folder", type=str, default=None, help="若 image 字段是相对路径，这里给出根目录")
    ap.add_argument("--max_show", type=int, default=3, help="打印多少条样本预览")
    ap.add_argument("--scan_limit", type=int, default=500, help="扫描多少条用于做统计与存在性检查")
    ap.add_argument("--sharegpt4v_file", type=str, default=None, help="你的 sharegpt4v.py 源码路径（用于动态导入 LazySupervisedDataset）")
    ap.add_argument("--params_file", type=str, default=None, help="你的 params.py 源码路径（用于动态导入 DataArguments）")
    ap.add_argument("--run_getitem", action="store_true", help="在集成测试中尝试触发 __getitem__")
    ap.add_argument("--index", type=int, default=0, help="__getitem__ 的索引位置")
    args = ap.parse_args()

    ds_dir = Path(args.data_path).resolve()
    if not ds_dir.is_dir():
        raise SystemExit(f"--data_path 必须是目录：{ds_dir}")

    image_folder = Path(args.image_folder).resolve() if args.image_folder else None
    if image_folder:
        print_kv("图像根目录", {"image_folder": str(image_folder)})

    print_kv("开始读取", {"data_path": str(ds_dir)})
    ds_obj = load_from_disk(str(ds_dir))
    print(f"[{ts()}] load_from_disk 成功。")

    # 基本结构与若干样本预览
    ds = inspect_dataset(ds_obj, max_show=args.max_show)

    # 进一步 schema 与存在性检查
    check_schema_and_stats(ds, image_folder=image_folder, scan_limit=args.scan_limit)

    # 可选：与 sharegpt4v 集成测试
    maybe_test_sharegpt4v_integration(
        ds_dir=ds_dir,
        image_folder=image_folder,
        sharegpt4v_file=Path(args.sharegpt4v_file) if args.sharegpt4v_file else None,
        params_file=Path(args.params_file) if args.params_file else None,
        run_getitem=args.run_getitem,
        index=args.index,
    )

if __name__ == "__main__":
    main()
