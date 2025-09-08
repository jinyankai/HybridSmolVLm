import os
os.environ.pop("HF_DATASETS_OFFLINE", None)
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ['HF_ENDPOINT']="https://hf-mirror.com"
from datasets import load_dataset,load_from_disk

# 第一次：从 Hugging Face Hub 下载
ds = load_from_disk("/home/jinkaiyan/data/ShareGPT4V-PT_dataset")

# 保存到本地目录
print(ds[0]['image'])