from hybrid_model import HybridSmolVLMForConditionalGeneration
import torch
import torch.nn as nn
from decoder_layer import HybridDecoderLayers
import json
from mamba2.hybrid_mamba_config import PhiMambaConfig
from mamba_ssm.utils.hf import  load_state_dict_hf
from load_sftensor import load_safetensors_to_dict
import os
from typing import List

class HybridSmolVLMWrapper(nn.Module):
    def __init__(self, base_model : HybridSmolVLMForConditionalGeneration,
                 mamba_config: PhiMambaConfig,
                 attn_layers: List[int],
                 init_with_qkv: bool,
                 dtype,
                 load_from_hub:bool= False,
                 checkpoint_path = None,
                 **kwargs
                 ):
        super().__init__(HybridSmolVLMWrapper,self)
        self.mamba_config = mamba_config
        self.attn_layers = attn_layers
        self.dtype = dtype
        self.model = base_model
        self.config = self.model.config
        self.layers = self.model.text_model.layers

        for layer_idx in  range(self.layers):
            if layer_idx not in self.attn_layers:
                mamba_decoder = HybridDecoderLayers(self.mamba_config,layer_idx)

                if init_with_qkv:
                    # ---------- 获取原始 Llama 层 ----------
                    llama_layer = self.layers[layer_idx]

                    # ---------- 1) MLP & RMSNorm ----------
                    llama_mlp = llama_layer.mlp  # LlamaMLP
                    mamba_mlp = mamba_decoder.mlp  # 你自己的两层 MLP

                    # fc1 ← gate_proj
                    mamba_mlp.fc1.weight.data.copy_(llama_mlp.gate_proj.weight.data)
                    if mamba_mlp.fc1.bias is not None and llama_mlp.gate_proj.bias is not None:
                        mamba_mlp.fc1.bias.data.copy_(llama_mlp.gate_proj.bias.data)

                    # fc2 ← down_proj
                    mamba_mlp.fc2.weight.data.copy_(llama_mlp.down_proj.weight.data)
                    if mamba_mlp.fc2.bias is not None and llama_mlp.down_proj.bias is not None:
                        mamba_mlp.fc2.bias.data.copy_(llama_mlp.down_proj.bias.data)

                    # ---------- 复制 LayerNorm / RMSNorm 权重 ----------
                    ln_state = llama_layer.input_layernorm.state_dict()  # 只有 weight
                    mamba_decoder.input_layernorm.load_state_dict(ln_state, strict=False)
                    # bias 没有对应源值，保留初始化（默认为 0）

                    # ---------- 2) 输出投影 (o_proj → out_proj) ----------
                    mamba_decoder.mamba.out_proj.weight.data.copy_(
                        llama_layer.self_attn.o_proj.weight.data
                    )

                    # ---------- 3) 输入投影 (q,k,v → in_proj) ----------
                    ip_w = mamba_decoder.mamba.in_proj.weight.data
                    d_in, d_xb = mamba_config.d_inner, mamba_config.d_xb

                    # 先 v，再 k，再 q（保持你原来的 slice 顺序）
                    ip_w[d_in: d_in + d_xb] = llama_layer.self_attn.v_proj.weight.data
                    ip_w[d_in + d_xb: d_in + 2 * d_xb] = llama_layer.self_attn.k_proj.weight.data
                    ip_w[d_in + 2 * d_xb: 2 * d_in + 2 * d_xb] = llama_layer.self_attn.q_proj.weight.data

                    # ---------- 4) bias（如果有的话） ----------
                    if mamba_config.is_bias:
                        mamba_decoder.mamba.out_proj.bias.data.copy_(llama_layer.self_attn.o_proj.bias.data)

                        ip_b = mamba_decoder.mamba.in_proj.bias.data
                        ip_b[d_in: d_in + d_xb] = llama_layer.self_attn.v_proj.bias.data
                        ip_b[d_in + d_xb: d_in + 2 * d_xb] = llama_layer.self_attn.k_proj.bias.data
                        ip_b[d_in + 2 * d_xb: 2 * d_in + 2 * d_xb] = llama_layer.self_attn.q_proj.bias.data

                    # ---------- 5) 确保 dtype ----------
                    mamba_decoder.mlp = mamba_decoder.mlp.to(dtype)
                    mamba_decoder.input_layernorm = mamba_decoder.input_layernorm.to(dtype)

                    # keep dtype to be the same
                    mamba_decoder.mlp = mamba_decoder.mlp.to(dtype)
                    mamba_decoder.input_layernorm = mamba_decoder.input_layernorm.to(dtype)

                if checkpoint_path is not None:
                    if load_from_hub:
                        # load from a huggingface hub
                        self.model.load_state_dict(
                            load_state_dict_hf(checkpoint_path, device=torch.device("cpu"), dtype=dtype))
                    else:
                        # load from a local directory
                        if os.path.exists(f"{checkpoint_path}/pytorch_model.bin"):
                            # support save from bin file
                            self.model.load_state_dict(
                                torch.load(f"{checkpoint_path}/pytorch_model.bin", map_location=torch.device("cpu")))
                        else:
                            # support save from safetensors
                            self.model.load_state_dict(load_safetensors_to_dict(checkpoint_path))


    def allocate_mamba_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.model.text_model.layers)
            if isinstance(layer, HybridDecoderLayers)

        }

    def forward(
        self,
        input_ids,
        **kwargs,
    ):
        return self.model(input_ids, **kwargs)

    def generate(
            self,
            input_ids,
            **kwargs,
    ):
        output = self.model.generate(
            input_ids,
            use_cache=False,
            **kwargs,
        )
        return output

    @staticmethod
    def init_distillation(
            checkpoint_path,
            tranformer_name,
            mamba_config,
            attn_layers,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            init_with_kqvo=True,
            **kwargs,
    ):
        transformer_model = HybridSmolVLMForConditionalGeneration.from_pretrained(tranformer_name, torch_dtype=dtype,
                                                                                    attn_implementation=attn_implementation)
        return HybridSmolVLMWrapper(base_model=transformer_model,
                                    mamba_config=mamba_config,
                                    attn_layers=attn_layers,
                                    dtype=dtype,
                                    checkpoint_path=checkpoint_path,
                                    init_with_qkv=init_with_kqvo)

    def save_config(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, 'mamba_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.mamba_config.__dict__, f, indent=4)


