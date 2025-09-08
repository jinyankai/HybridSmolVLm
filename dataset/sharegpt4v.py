import copy
import os
from dataclasses import dataclass, field
from typing import Dict
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset
from PIL import Image

try:
    from datasets import load_from_disk, DatasetDict  # 可选依赖
except Exception:
    load_from_disk = None
    DatasetDict = None


from .params import DataArguments
from .constants import *

EOS_TOKEN = "<end_of_utterance>"


def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output



def pad_pixel_values(pixel_values_list, pad_value=0.0):
    batch_size = len(pixel_values_list)
    frame_lengths = [pv.shape[1] for pv in pixel_values_list]
    T_max = max(frame_lengths)
    _, _, C, H, W = pixel_values_list[0].shape
    dtype = pixel_values_list[0].dtype
    device = pixel_values_list[0].device

    output = torch.full((batch_size, T_max, C, H, W),
                        fill_value=pad_value,
                        dtype=dtype,
                        device=device)

    # Copy actual value
    for i, pv in enumerate(pixel_values_list):
        t_i = pv.shape[1]
        output[i, :t_i] = pv[0]
    return output


def pad_pixel_attention_masks(mask_list, pad_value=0):
    batch_size = len(mask_list)
    frame_lengths = [mask.shape[1] for mask in mask_list]
    T_max = max(frame_lengths)

    _, _, H, W = mask_list[0].shape
    dtype = mask_list[0].dtype
    device = mask_list[0].device

    output = torch.full(
        (batch_size, T_max, H, W),
        fill_value=pad_value,
        dtype=dtype,
        device=device
    )

    for i, m in enumerate(mask_list):
        t_i = m.shape[1]
        output[i, :t_i] = m[0]

    return output


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
            self,
            data_path: str | list,
            processor: transformers.ProcessorMixin,
            data_args: DataArguments,
            padding=True,
    ):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = None

        if isinstance(data_path, list):
            # 直接传入了样本列表
            list_data_dict = data_path

        elif isinstance(data_path, str):
            # 1) 若是文件且后缀为 .json => 按 JSON 读
            if os.path.isfile(data_path) and data_path.lower().endswith(".json"):
                with open(data_path, "r") as f:
                    list_data_dict = json.load(f)

            # 2) 若是目录 => 尝试 datasets.load_from_disk
            elif os.path.isdir(data_path) and load_from_disk is not None:
                try:
                    ds_obj = load_from_disk(data_path)
                    # DatasetDict: 取 train split 优先，否则取第一个
                    if DatasetDict is not None and isinstance(ds_obj, DatasetDict):
                        if "train" in ds_obj:
                            ds = ds_obj["train"]
                        else:
                            first_key = next(iter(ds_obj.keys()))
                            ds = ds_obj[first_key]
                    else:
                        ds = ds_obj  # 已经是 Dataset

                    # 转为 Python 列表（每个元素是一个 dict）
                    list_data_dict = ds.to_list()

                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load dataset from disk path '{data_path}' via datasets.load_from_disk: {e}"
                    )
            else:
                # 3) 兜底：尝试按 JSON 读取（允许用户传入无 .json 后缀的 JSON 文件名）
                try:
                    with open(data_path, "r") as f:
                        list_data_dict = json.load(f)
                except Exception as e:
                    raise FileNotFoundError(
                        f"Unable to interpret data_path='{data_path}'. "
                        f"Expected one of: a JSON file path, a directory created by datasets.save_to_disk, or a Python list. "
                        f"Original error: {e}"
                    )
        else:
            raise TypeError(
                "data_path must be a list (in-memory samples) or a str (JSON path or datasets 'save_to_disk' directory)."
            )

        self.processor = processor
        self.list_data_dict = list_data_dict

        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.max_num_frames = data_args.max_num_frames

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        is_video = False
        num_frames = None

        processor = self.processor
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
            # Making dummy images for the case where there are no images
            # This is neccessary for using deepspeed zeor3.
            # All the inputs in the same batch should have the same activation flow.
            pixel_values = torch.zeros(1, 13, 3, 384, 384)
            pixel_attention_mask = torch.zeros(1, 13, 384, 384)

        sources = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=is_video, num_frames=num_frames))

        all_input_ids = [torch.tensor([1])]  # bos token id
        all_labels = [torch.tensor([-100])]  # ignore bos token

        for idx, j in enumerate(range(0, len(sources), 2)):
            user_input = sources[j]
            gpt_response = sources[j + 1]

            is_last_turn = (idx == (len(sources) // 2 - 1))

            # The white space is important here.
            # If the user prompt starts with a image token then it won't have a white space.
            if user_input['content'].startswith(LLAVA_IMAGE_TOKEN):
                user_prompt = f"User:{user_input['content']}{EOS_TOKEN}\nAssistant: "
            else:
                user_prompt = f"User: {user_input['content']}{EOS_TOKEN}\nAssistant: "

            if is_last_turn:
                gpt_prompt = f"{gpt_response['content']}{EOS_TOKEN}"
            else:
                gpt_prompt = f"{gpt_response['content']}{EOS_TOKEN}\n"

            if LLAVA_IMAGE_TOKEN in user_prompt:
                inputs = processor(text=user_prompt, images=images, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                pixel_values = inputs.get('pixel_values', None)
                pixel_attention_mask = inputs.get('pixel_attention_mask', None)

            else:
                prompt_input_ids = processor.tokenizer(user_prompt, add_special_tokens=False, return_tensors='pt')[
                    'input_ids']

            response_input_ids = processor.tokenizer(gpt_prompt, add_special_tokens=False, return_tensors='pt')[
                'input_ids']

            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)

        attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            attention_mask=attention_mask,
            labels=labels,
        )

        return data_dict


class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_attention_mask = []

        for example in examples:
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])
            batch_pixel_values.append(example.get("pixel_values"))
            batch_pixel_attention_mask.append(example.get("pixel_attention_mask"))

        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )

        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)
        pixel_values = pad_pixel_values(batch_pixel_values, pad_value=0.0)
        pixel_attention_mask = pad_pixel_attention_masks(batch_pixel_attention_mask, pad_value=0)

        batch_dict = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        if pixel_values is not None:
            batch_dict.update(pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask)

        return batch_dict


def replace_image_tokens(input_string, start_count=1):
    count = start_count

    if LLAVA_IMAGE_TOKEN not in input_string:
        return input_string, count

    while LLAVA_IMAGE_TOKEN + '\n' in input_string:
        input_string = input_string.replace(LLAVA_IMAGE_TOKEN + '\n', "<image>", 1)
        count += 1

    return input_string, count


def video_to_image_tokens(input_string, num_frames):
    frame_tokens = "\n".join([LLAVA_IMAGE_TOKEN] * num_frames)
    input_string = input_string.replace(LLAVA_VIDEO_TOKEN, frame_tokens)

    return input_string


def llava_to_openai(conversations, is_video=False, num_frames=None):
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    image_count = 1  # Initialize image count here
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


def make_supervised_data_module(processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = LazySupervisedDataset(
        data_path=data_args.data_path, processor=processor, data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id)

    return dict(train_dataset=sft_dataset,
                eval_dataset=None,
                data_collator=data_collator)