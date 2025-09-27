import torch
from collections import OrderedDict
from transformers import AutoProcessor, BitsAndBytesConfig
from model.mamba2.hybrid_mamba_config import PhiMambaConfig
from model.inference.eval_hybridwrapper import EvalMamba2TransformerHybridModelWrapper
import os
os.environ['HF_ENDPOINT']="https://hf-mirror.com"




def load_base_ckp_for_lora(ckp_path):
    ckp = torch.load(ckp_path, map_location=torch.device('cpu'))
    new_ckp = OrderedDict()
    for k, v in ckp.items():
        new_k = k.replace('.base_layer', '')
        new_ckp[new_k] = v
    return new_ckp


def load_base_ckp_for_pretrain(ckp_path):
    ckp = torch.load(ckp_path, map_location=torch.device('cpu'))
    # new_ckp = OrderedDict()
    # for k, v in ckp.items():
    #     new_k = k.replace('.base_layer', '')
    #     new_ckp[new_k] = v
    return ckp


def load_pretrained_model(model_name_or_path,pretrain_path, load_type='hf', load_8bit=False, load_4bit=False, device_map="auto",
                          device="cuda", **kwargs):
    kwargs = {"device_map": device_map, **kwargs}
    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16
    if model_name_or_path is not None and 'lora' not in model_name_or_path:
        mamba_cfg = PhiMambaConfig(
            d_model=576,
            ssm_cfg={"expand": 1, "ngroups": 9, "d_state": 64, "d_conv": 4},
            rms_norm_eps=1e-05,
            d_inner=576,
            d_xb=192,
            intermediate_size=1536,
            hidden_act="silu",
            n_layer=30,
            attn_layers=[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                         29],
            resid_pdrop=0.1,
            bidirectional=False,
            is_bias=False
        )
        # base_model = HybridSmolVLMForConditionalGeneration.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct",torch_dtype=torch.bfloat16,)
        wrapper = EvalMamba2TransformerHybridModelWrapper.from_pretrained(pretrain_path,torch_dtype=torch.float16)
        model = wrapper
        processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct",local_files_only=True)
        image_processor = processor.image_processor
        tokenizer = processor.tokenizer
    else:
        raise Exception("Please ensure the model is compatible with the loading function or manually load the processor.")

    context_len = getattr(model.config, 'max_sequence_length', 2048)
    # tokenizer = AutoTokenizer.from_pretrained(model.config.llm_model_name_or_path, use_fast=False, padding_side="right")
    # tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, image_processor, context_len
