import math
import os
os.environ.pop("HF_DATASETS_OFFLINE", None)
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ['HF_ENDPOINT']="https://hf-mirror.com"
from dataset.cauldron import CauldronDataset,get_dataloader
from config.train_config import TrainConfig
from config.test_config import TestConfig
from model.decoder_layer import HybridDecoderLayers
from model.mamba2.hybrid_mamba_config import PhiMambaConfig
from model.model_wrapper import HybridSmolVLMForConditionalGeneration,HybridSmolVLMWrapper
from transformers import AutoProcessor, AutoTokenizer
import logging
logger = logging.getLogger(__name__)
import torch
import argparse
from transformers.image_utils  import load_image


def build_model(cfg:TestConfig):
    # Prepare Teacher Model
    teacher_model = HybridSmolVLMForConditionalGeneration.from_pretrained(cfg.teacher_name,torch_dtype=cfg.dtype)
    for param in teacher_model.parameters():
        param.requires_grad = False
    logger.log(logging.INFO, f"Teacher model {cfg.teacher_name} loaded successfully.")
    mamba_cfg = PhiMambaConfig(
        d_model=2048,
        ssm_cfg={"expand": 1, "ngroups":32, "d_state": 64,"d_conv": 4},
        rms_norm_eps=1e-05,
        d_inner=2048,
        d_xb=2048,
        intermediate_size=8192,
        hidden_act="silu",
        n_layer=24,
        attn_layers=[0,1,2,3,5,6,7,9,10,11,12,13,14,15,17,18,19,21,22,23],
        resid_pdrop=0.1,
        bidirectional=False,
        is_bias=False
        )
    student_wrapper = HybridSmolVLMWrapper.init_distillation(checkpoint_path=None,
                                                             tranformer_name=cfg.teacher_name,
                                                             mamba_config=mamba_cfg,
                                                             attn_layers=mamba_cfg.attn_layers,
                                                             dtype=cfg.dtype)
    student_model = student_wrapper.model
    student_wrapper.from_pretrain(cfg.check_point_path)
    logger.info("Successfully load Student, Building tok")

    tok = AutoTokenizer.from_pretrained(cfg.teacher_name, local_files_only=True, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    logger.info("Processor")
    processor = AutoProcessor.from_pretrained(cfg.teacher_name, local_files_only=True)

    return teacher_model,student_wrapper,student_model,tok, processor


def parse_args() -> TestConfig:
    p = argparse.ArgumentParser()
    for f in TestConfig.__dataclass_fields__.values():
        name = f"--{f.name}"
        p.add_argument(name, type=type(f.default) if f.default is not None else str, default=f.default, nargs="?",
                       help=f.metadata.get("help", ""))
    args = p.parse_args()
    return TestConfig(**vars(args))

def main():
    cfg = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    teacher_model , student_wrapper, student_model,tok,processor = build_model(cfg)
    student_model.to("cuda")
    logger.info("Successfully build model")
    image = load_image("https://huggingface.co/spaces/merve/chameleon-7b/resolve/main/bee.jpg")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Can you describe this images?"}
            ]
        },
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to("cuda")


    with torch.no_grad():
        gen_ids = student_model.generate(**inputs)
    decoded_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
    print(f"  - Generated text: \"{decoded_text.strip()}\"")

    print("  - Generation successful.")

    print("\n--- All tests passed successfully! ---")




if __name__ == "__main__":
    # accelerate
    # launch - -num_processes
    # 4 - -mixed_precision
    # bf16 - -deepspeed_config_file
    # deepspeed_config.json
    # train.py
    main()
