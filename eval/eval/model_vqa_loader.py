import argparse

import math
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import shortuuid


import ujson as json

from model.load_model import *

# ------------------ 关键改动：从您的 eval_loader.py 中导入数据处理模块 ------------------
from dataset.eval_loader import EvalDataset, EvalDataCollator

from dataset.params import DataArguments  # 假设 DataArguments 在 params.py 中
from transformers import AutoProcessor, AutoModelForVision2Seq
import os
os.environ['HF_ENDPOINT']="https://hf-mirror.com"


# ------------------------------------------------------------------------------------

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]




def eval_model(args):
    # 1. 模型加载部分 (与 model_vqa_loader.py 保持一致)
    disable_torch_init()
    model_name = args.model_base
    model_path = os.path.expanduser(args.model_path)
    # 注意：我们现在期望加载一个集成的 processor，而不是独立的 tokenizer 和 image_processor
    # load_pretrained_model 可能需要调整，或我们在这里使用 AutoProcessor
    try:
        processor = AutoProcessor.from_pretrained(model_name,local_files_only=True)
        # model = AutoModelForVision2Seq.from_pretrained(model_name,local_files_only=True)
    except Exception:
        raise NotImplementedError("请确保 transformers 版本支持 AutoProcessor 或手动加载 processor。")

    model, _,image_processor, _ = load_pretrained_model(model_name,pretrain_path=model_path)#TODO
    tokenizer = processor.tokenizer  # 确保 tokenizer 一致
    model.to(device='cuda')

    # 2. 数据加载部分 (与 model_vqa_loader.py 保持一致)
    questions = [json.loads(q) for q in open(args.question_file, "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = args.answers_file
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # ------------------ 关键改动：使用 eval_loader 的数据处理模块 ------------------
    # 创建一个 DataArguments 实例来配置数据加载器
    data_args = DataArguments(
        data_path=args.question_file,  # 直接传入问题列表
        image_folder=args.image_folder
    )

    # 使用 make_eval_data_module 来获取数据集和数据整理器
    # data_module = make_eval_data_module(processor=processor, data_args=data_args)
    eval_dataset = EvalDataset(args.question_file,processor=processor,data_args=data_args)
    data_collator = EvalDataCollator(pad_token_id=processor.tokenizer.pad_token_id)

    # 创建 DataLoader
    # batch_size 可以大于 1，因为 DataCollatorForEvalDataset 支持批处理
    data_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=1,  # 改为可配置的 batch_size
        num_workers=16,
        shuffle=False,
        collate_fn=data_collator
    )
    # ------------------------------------------------------------------------------------

    # 3. 推理循环部分 (修改以适应新的数据加载器)
    # 我们需要同时迭代 data_loader 和原始的 questions 列表来获取 question_id 等元数据
    # 为了处理批处理，我们需要一个索引

    for batch,line in tqdm(zip(data_loader,questions), total=len(questions)):

        # 将批次数据移动到GPU
        input_ids = batch['input_ids'].to(device='cuda', non_blocking=True)
        attention_mask = batch['attention_mask'].to(device='cuda', non_blocking=True)

        # pixel_values 是新的键，替换了 image_tensor
        pixel_values = batch.get('pixel_values')
        if pixel_values is not None:
            pixel_values = pixel_values.to(dtype=torch.float16, device='cuda', non_blocking=True)

        # 注意：eval_loader 不提供 image_sizes，如果模型严格需要，需要额外处理
        # 对于大多数模型，attention_mask 已经足够
        # 如果确实需要，可以修改 LazyEvalDataset 来返回 image.size

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                # attention_mask=attention_mask,
                pixel_values=pixel_values,  # 使用 pixel_values
                # temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id
            )

        # 解码时要考虑批处理
        # output_ids 包含输入的 prompt 部分，需要移除
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

        outputs_ids_only = output_ids[:, input_token_len:]
        outputs = tokenizer.batch_decode(outputs_ids_only, skip_special_tokens=True)[0].strip()
        print("Printing outputs")
        print(outputs)
        time.sleep(5)
        # 写入结果，需要为批次中的每个样本写入

        idx = line["question_id"]
        cur_prompt = line["text"]


        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": args.model_base,
                                   "metadata": {}}) + "\n")


    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 保留 model_vqa_loader.py 的所有参数
    parser.add_argument("--model-base", type=str, default="HuggingFaceTB/SmolVLM-256M-Instruct")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    # conv-mode 不再需要，因为 prompt 逻辑在 eval_loader 中
    parser.add_argument("--conv-mode", type=str, default="llama")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    # 新增 batch_size 参数
    parser.add_argument("--batch_size", type=int, default=1)  # 允许大于1的批处理
    args = parser.parse_args()

    eval_model(args)