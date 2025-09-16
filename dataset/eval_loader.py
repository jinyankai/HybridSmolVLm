import copy
import os
from dataclasses import dataclass, field
from typing import Dict, List
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset
from PIL import Image

# 从源文件中复用大部分辅助函数和常量
from params import DataArguments
from constants import *


# 假设这些常量定义在 constants.py 中
# LLAVA_IMAGE_TOKEN = "<image>"
# LLAVA_VIDEO_TOKEN = "<video>"
# EOS_TOKEN = "<end_of_utterance>"
# IGNORE_INDEX = -100

# 从源文件中复用这些工具函数
def video_to_image_tokens(input_string, num_frames):
    frame_tokens = "\n".join([LLAVA_IMAGE_TOKEN] * num_frames)
    input_string = input_string.replace(LLAVA_VIDEO_TOKEN, frame_tokens)
    return input_string


def replace_image_tokens(input_string, start_count=1):
    count = start_count
    if LLAVA_IMAGE_TOKEN not in input_string:
        return input_string, count
    while LLAVA_IMAGE_TOKEN + '\n' in input_string:
        input_string = input_string.replace(LLAVA_IMAGE_TOKEN + '\n', "<image>", 1)
        count += 1
    return input_string, count


def llava_to_openai(conversations, is_video=False, num_frames=None):
    role_mapping = {"human": "user", "gpt": "assistant"}
    transformed_data = []
    image_count = 1
    for conversation in conversations:
        if is_video:
            conversation['value'] = video_to_image_tokens(conversation["value"], num_frames)

        transformed_content, image_count = replace_image_tokens(conversation["value"], image_count)
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content
        }
        transformed_data.append(transformed_entry)
    return transformed_data


def pad_pixel_values(pixel_values_list, pad_value=0.0):
    if not any(pv is not None for pv in pixel_values_list):
        return None

    # 过滤掉 None 值，以防某些样本没有图像
    valid_pvs = [pv for pv in pixel_values_list if pv is not None]
    if not valid_pvs:
        return None

    batch_size = len(pixel_values_list)
    frame_lengths = [pv.shape[1] for pv in valid_pvs]
    T_max = max(frame_lengths)
    _, _, C, H, W = valid_pvs[0].shape
    dtype = valid_pvs[0].dtype
    device = valid_pvs[0].device

    output = torch.full((batch_size, T_max, C, H, W),
                        fill_value=pad_value,
                        dtype=dtype,
                        device=device)

    # 拷贝有效值
    valid_idx = 0
    for i, pv in enumerate(pixel_values_list):
        if pv is not None:
            t_i = pv.shape[1]
            output[i, :t_i] = pv[0]
            valid_idx += 1
    return output


class LazyEvalDataset(Dataset):
    """
    用于验证的数据集。
    它会处理好多轮对话历史，并只返回生成任务所需的 prompt 部分。
    """

    def __init__(
            self,
            data_path: str | list,
            processor: transformers.ProcessorMixin,
            data_args: DataArguments,
    ):
        super(LazyEvalDataset, self).__init__()
        # 数据加载逻辑与 LazySupervisedDataset 保持一致
        list_data_dict = None
        if isinstance(data_path, list):
            list_data_dict = data_path
        elif isinstance(data_path, str):
            if os.path.isfile(data_path) and data_path.lower().endswith(".json"):
                with open(data_path, "r") as f:
                    list_data_dict = json.load(f)
            else:
                try:
                    with open(data_path, "r") as f:
                        list_data_dict = json.load(f)
                except Exception as e:
                    raise FileNotFoundError(f"无法解析 data_path='{data_path}'. 期望是 JSON 文件路径。")
        else:
            raise TypeError("data_path 必须是 list 或 str (JSON 路径)。")

        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor | str]:
        sources = self.list_data_dict[i]

        # 图像处理逻辑与训练脚本类似
        pixel_values = None
        if "image" in sources:
            image_files = sources["image"]
            image_folder = self.data_args.image_folder
            if isinstance(image_files, str):
                image_files = [image_files]

            images = []
            for image_file in image_files:
                if not os.path.exists(image_file):
                    image_file = os.path.join(image_folder, image_file)
                images.append(Image.open(image_file).convert("RGB"))
        else:
            images = None

        # 将对话格式转换为 openai 格式
        conversations = llava_to_openai(sources['conversations'])

        # --- 验证逻辑的核心区别 ---
        # 1. 构建包含所有对话历史的 prompt
        # 2. 将最后一个 assistant 的回复作为 ground_truth
        prompt_parts = []

        # 遍历到倒数第二条消息（即最后一个用户问题）
        for turn in conversations[:-1]:
            role = turn['role']
            content = turn['content']
            if role == 'user':
                # 空格很重要
                if content.startswith(LLAVA_IMAGE_TOKEN):
                    prompt_parts.append(f"User:{content}{EOS_TOKEN}")
                else:
                    prompt_parts.append(f"User: {content}{EOS_TOKEN}")
            else:  # assistant
                prompt_parts.append(f"Assistant: {content}{EOS_TOKEN}")

        # 添加最后的 "Assistant: "，提示模型开始生成
        prompt_parts.append("Assistant: ")

        final_prompt = "\n".join(prompt_parts)

        # 提取用于评估的真实答案
        ground_truth = conversations[-1]['content']

        # 使用 processor 处理文本和图像
        if images:
            inputs = self.processor(text=final_prompt, images=images, return_tensors='pt')
            input_ids = inputs['input_ids'].squeeze(0)
            pixel_values = inputs.get('pixel_values')
        else:
            inputs = self.processor.tokenizer(final_prompt, return_tensors='pt')
            input_ids = inputs['input_ids'].squeeze(0)

        return dict(
            input_ids=input_ids,
            pixel_values=pixel_values,
            ground_truth=ground_truth  # 保存真实答案文本
        )


class DataCollatorForEvalDataset(object):
    """
    为验证任务收集和填充数据。
    - 对 input_ids 进行左填充（Left Padding）。
    - 收集 ground_truth 文本。
    """

    def __init__(self, processor: transformers.ProcessorMixin):
        self.processor = processor

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor | List[str]]:
        batch_input_ids = [example["input_ids"] for example in examples]
        batch_pixel_values = [example.get("pixel_values") for example in examples]
        ground_truths = [example["ground_truth"] for example in examples]

        # 使用 processor.tokenizer.pad 进行填充，更健壮
        # 对于生成任务，应使用 left padding
        padded_inputs = self.processor.tokenizer.pad(
            {"input_ids": batch_input_ids},
            padding=True,
            return_tensors="pt",
            padding_side="left"
        )

        # 填充 pixel_values（如果存在）
        pixel_values = pad_pixel_values(batch_pixel_values, pad_value=0.0)

        batch_dict = dict(
            input_ids=padded_inputs.input_ids,
            attention_mask=padded_inputs.attention_mask,
            ground_truths=ground_truths,
        )

        if pixel_values is not None:
            batch_dict['pixel_values'] = pixel_values

        return batch_dict


def make_eval_data_module(processor, data_args):
    """为验证构建 dataset 和 data_collator。"""
    eval_dataset = LazyEvalDataset(
        data_path=data_args.data_path, processor=processor, data_args=data_args
    )
    data_collator = DataCollatorForEvalDataset(processor=processor)

    return dict(eval_dataset=eval_dataset,
                data_collator=data_collator)