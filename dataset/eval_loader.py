import os
import ujson as json
from typing import Dict, List, Union

import torch
from torch.utils.data import Dataset
from PIL import Image
import transformers

# 假设这些常量在你的项目其他地方有定义，与训练时保持一致。
# 如果没有，你可以取消下面的注释并设置正确的值。
# from .constants import EOS_TOKEN
IGNORE_INDEX = -100

LLAVA_IMAGE_TOKEN = "<image>"
LLAVA_VIDEO_TOKEN = "<video>"

EOS_TOKEN = "<end_of_utterance>" # 这是一个常见的 EOS token，你可能需要根据你的模型进行调整

# --- 辅助函数 (从 sharegpt4v.py 复制而来，以保证文件独立性) ---

def pad_sequence(sequences: List[torch.Tensor], padding_side: str = 'right', padding_value: int = 0) -> torch.Tensor:
    """
    将一个序列列表填充到相同的长度。
    sequences: 形状为 [seq_len, *] 的张量列表
    """
    assert padding_side in ['right', 'left']
    if not sequences:
        return torch.empty(0)
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


# --- 评估数据集 ---

class EvalDataset(Dataset):
    """
    用于评估的数据集，专门处理以下特定JSON格式：
    {"question_id": "...", "image": "...", "text": "..."}
    """

    def __init__(
        self,
        data_path: Union[str, List[Dict]],
        processor: transformers.ProcessorMixin,
        data_args,  # 应该包含 image_folder 路径
    ):
        """
        Args:
            data_path (Union[str, List[Dict]]): 数据文件路径 (JSON/JSONL) 或内存中的数据列表。
            processor (transformers.ProcessorMixin): 用于处理文本和图像的处理器。
            data_args: 包含额外参数的对象，如 data_args.image_folder。
        """
        super().__init__()
        # --- MODIFIED: 处理文件路径 (str) 和内存中的列表 (list) ---
        if isinstance(data_path, str):
            try:
                with open(data_path, "r", encoding='utf-8') as f:
                    # 尝试将其作为单个JSON对象（可能是列表）加载
                    try:
                        self.list_data_dict = json.load(f)
                        if not isinstance(self.list_data_dict, list):
                             raise ValueError("JSON文件内容必须是一个列表。")
                    except json.JSONDecodeError:
                        # 如果作为单个JSON对象失败，回退到按行读取JSONL
                        f.seek(0)
                        self.list_data_dict = [json.loads(line) for line in f if line.strip()]
            except Exception as e:
                raise IOError(f"读取或解析文件失败 {data_path}: {e}")
        elif isinstance(data_path, list):
            # 如果 data_path 本身就是一个列表，直接使用它
            self.list_data_dict = data_path
        else:
            raise TypeError(
                f"data_path 必须是 str (文件路径) 或 list (内存中的数据)，但收到了 {type(data_path)}"
            )

        self.processor = processor
        self.data_args = data_args
        self.image_token = "<image>"

    def __len__(self) -> int:
        return len(self.list_data_dict)

    def __getitem__(self, i: int) -> Dict[str, Union[torch.Tensor, str]]:
        """获取并处理单个数据样本。"""
        item = self.list_data_dict[i]
        question_id = item.get("question_id")
        image_file = item.get("image")
        text = item.get("text")

        if not all([question_id, image_file, text]):
             raise ValueError(f"数据索引 {i} 缺少必要的键 ('question_id', 'image', 'text')")

        # 1. 构建 prompt
        # 这个格式至关重要，应与模型微调时使用的格式匹配。
        # 格式: "User: <image>\n{question}\nAssistant:"
        prompt = f"User: {self.image_token}\n{text.strip()}\n{EOS_TOKEN}\nAssistant: "

        # 2. 加载和处理图像
        image_folder = self.data_args.image_folder
        if image_folder and not os.path.isabs(image_file):
            image_path = os.path.join(image_folder, image_file)
        else:
            image_path = image_file

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件未找到: {image_path}")
        image = Image.open(image_path).convert("RGB")

        # 3. 使用 processor 准备模型输入
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors='pt',
        )

        input_ids = inputs['input_ids'].squeeze(0)
        pixel_values = inputs.get('pixel_values')
        if pixel_values is not None:
            pixel_values = pixel_values.squeeze(0)

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "question_id": question_id,
        }


# --- 评估数据整理器 ---

class EvalDataCollator:
    """
    为评估整理样本。
    填充输入并为模型生成准备批次。
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples: List[Dict]) -> Dict[str, Union[torch.Tensor, List[str]]]:
        batch_input_ids = [ex["input_ids"] for ex in examples]
        batch_pixel_values = [ex["pixel_values"] for ex in examples if ex.get("pixel_values") is not None]
        question_ids = [ex["question_id"] for ex in examples]

        # 填充 input_ids。对于生成任务，左填充是标准做法。
        input_ids = pad_sequence(
            batch_input_ids, padding_side='left', padding_value=self.pad_token_id
        )

        # 从填充后的 input_ids 创建 attention_mask
        attention_mask = input_ids.ne(self.pad_token_id)

        # 如果存在像素值，则堆叠它们
        if batch_pixel_values:
            pixel_values = torch.stack(batch_pixel_values, dim=0)
        else:
            pixel_values = None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "question_ids": question_ids,
        }

