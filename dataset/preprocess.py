import argparse
import os
from functools import partial
import datasets
from datasets import load_dataset
from transformers import AutoProcessor
from tqdm import tqdm

# 从您的项目中导入必要的函数和类
from sharegpt4v import llava_to_openai, IGNORE_INDEX, LLAVA_IMAGE_TOKEN


def preprocess_function(example, processor):
    """
    对单个样本进行完整的预处理，将其转换为模型可以直接使用的张量格式。
    """
    # 1. 加载图片
    try:
        # datasets.Image() 会返回一个PIL.Image对象
        image = example["image"].convert("RGB")
    except Exception as e:
        print(f"Skipping sample with ID {example.get('id', 'N/A')} due to image loading error: {e}")
        return None  # 返回None，以便后续过滤掉这个坏样本

    # 2. 处理对话
    conversations = example.get('conversations', [])
    if not conversations:
        return None

    sources = llava_to_openai(conversations)

    all_input_ids = [torch.tensor([processor.tokenizer.bos_token_id])]
    all_labels = [torch.tensor([IGNORE_INDEX])]

    # 3. 模板化和分词
    for idx, j in enumerate(range(0, len(sources), 2)):
        user_input = sources[j]
        gpt_response = sources[j + 1]

        user_prompt_parts = []
        # 确保图片占位符在文本的开头
        if LLAVA_IMAGE_TOKEN not in user_input['content']:
            user_input['content'] = LLAVA_IMAGE_TOKEN + '\n' + user_input['content']

        # 构建对话模板
        user_prompt = f"User: {user_input['content']}\nAssistant: "
        gpt_prompt = f"{gpt_response['content']}{processor.tokenizer.eos_token}"
        if not (idx == (len(sources) // 2 - 1)):
            gpt_prompt += "\n"  # 如果不是最后一轮对话，添加换行符

        # 分词
        prompt_input_ids = processor.tokenizer(user_prompt, add_special_tokens=False)['input_ids']
        response_input_ids = processor.tokenizer(gpt_prompt, add_special_tokens=False)['input_ids']

        input_ids = prompt_input_ids + response_input_ids
        labels = [IGNORE_INDEX] * len(prompt_input_ids) + response_input_ids

        all_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
        all_labels.append(torch.tensor(labels, dtype=torch.long))

    # 4. 图片处理
    try:
        image_inputs = processor(images=image, return_tensors='pt')
        pixel_values = image_inputs.pixel_values.squeeze(0)
        pixel_attention_mask = image_inputs.pixel_attention_mask.squeeze(0)
    except Exception as e:
        print(f"Skipping sample with ID {example.get('id', 'N/A')} due to image processing error: {e}")
        return None

    # 5. 拼接并返回最终结果
    final_input_ids = torch.cat(all_input_ids)
    final_labels = torch.cat(all_labels)

    return {
        "input_ids": final_input_ids,
        "labels": final_labels,
        "pixel_values": pixel_values,
        "pixel_attention_mask": pixel_attention_mask,
    }


def main(args):
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, local_files_only=True)

    print(f"Loading raw dataset from: {args.data_path}")
    # 注意：确保image_folder的路径是正确的，以便datasets库能找到图片
    # 这里我们假设原始数据中的图片路径是相对于image_folder的
    # datasets库在加载时会自动处理
    raw_dataset = load_dataset("json", data_files=args.data_path, split="train")

    # 将'image'列转换为datasets.Image()类型，方便处理
    # 注意：这里假设您的原始JSON文件中，'image'字段是图片的相对路径
    def resolve_image_path(example):
        example['image'] = os.path.join(args.image_folder, example['image'])
        return example

    raw_dataset = raw_dataset.map(resolve_image_path, num_proc=args.num_proc)
    raw_dataset = raw_dataset.cast_column("image", datasets.Image())

    print("Starting preprocessing...")

    # 使用functools.partial来传递固定的processor参数
    bound_preprocess_function = partial(preprocess_function, processor=processor)

    processed_dataset = raw_dataset.map(
        bound_preprocess_function,
        batched=False,  # 一次处理一个样本
        num_proc=args.num_proc,
        remove_columns=raw_dataset.column_names
    )

    # 过滤掉处理失败的样本
    processed_dataset = processed_dataset.filter(lambda x: x is not None)

    print(f"Preprocessing finished. Number of samples: {len(processed_dataset)}")
    print("Example of a processed sample:")
    print(processed_dataset[0])

    print(f"Saving processed dataset to: {args.output_dir}")
    processed_dataset.save_to_disk(args.output_dir)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the source JSON file.")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the directory containing images.")
    parser.add_argument("--model_name_or_path", type=str, default="HuggingFaceTB/SmolVLM-256M-Instruct",
                        help="Model name to load processor.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the processed dataset.")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of processes for mapping.")
    args = parser.parse_args()

    # 为了能正确导入项目中的模块，需要将项目根目录添加到Python路径中
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import torch

    main(args)


# ** 第2步：运行预处理脚本 **
#
# 在您的终端中，运行这个新脚本。请确保路径正确：
# ```bash
#  python dataset/preprocess.py --data_path /data/jyk_data/data/sharegpt4v/cleaned_sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json  --image_folder /data/jyk_data/data/ --output_dir /data/jyk_data/processed_data/ --num_proc 16
