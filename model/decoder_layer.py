import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union

from torch.nn import factory_kwargs
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
    LlamaAttention,
    LlamaMLP,
    LlamaRMSNorm,
    ACT2FN
)
from transformers import PreTrainedModel, PretrainedConfig

from .mamba2.hybrid_mamba_layer import Mamba2
from .mamba2.hybrid_mamba_config import PhiMambaConfig

class HybridLlamaMLP(nn.Module):
    def __init__(self, config:PhiMambaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.d_model
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.is_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.is_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.is_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class HybridDecoderLayers(nn.Module):
    def __init__(self, mamba_config: PhiMambaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = mamba_config.d_model
        self.layer_idx = layer_idx

        # --- Mamba2 block ---
        self.self_attn_mamba = Mamba2(
            d_model=mamba_config.d_model,
            d_xb=mamba_config.d_xb,
            d_inner=mamba_config.d_inner,
            layer_idx=layer_idx,
            bias=mamba_config.is_bias,
            **mamba_config.ssm_cfg,
        )
        # 统一标识名，供上层判断
        self.is_mamba = True

        # --- 与 Llama 对齐的 LN + MLP ---
        self.mlp = HybridLlamaMLP(mamba_config)
        self.input_layernorm = LlamaRMSNorm(mamba_config.d_model, eps=mamba_config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(mamba_config.d_model, eps=mamba_config.rms_norm_eps)

    # 如果需要，改成正确的成员
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        if hasattr(self.self_attn_mamba, "allocate_inference_cache"):
            return self.self_attn_mamba.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
        return None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,  # 传进来也“忽略”
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # 会被忽略
        **kwargs,
    ):
        # 前置 LN
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # --- Mamba2 前向 ---
        # 如果你的 Mamba 实现支持状态（state）/缓存，就用它自己的；否则直接调用
        out = self.self_attn_mamba(hidden_states)
        if isinstance(out, tuple):
            mamba_out, mamba_state = out[0], out[1]
        else:
            mamba_out, mamba_state = out, None

        hidden_states = residual + mamba_out

        # MLP 残差
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # --- 返回格式与 HF 对齐 ---
        # 对于使用 DynamicCache 的上层，Llama 层会在内部 update 全局 cache；
        # Mamba 层不参与 KV，返回 present_key_value=None, attentions=None 即可。
        present_key_value = None
        attn_weights = None

        return (hidden_states, present_key_value, attn_weights)

# mamba_config = PhiMambaConfig(
#         d_model=2048,
#         ssm_cfg={"expand": 1, "ngroups":32, "d_state": 64,"d_conv": 4},
#         rms_norm_eps=1e-05,
#         d_inner=2048,
#         d_xb=2048,
#         intermediate_size=8192,
#         hidden_act="silu",
#         n_layer=24,
#         attn_layers=[0,1,2,3,5,6,7,9,10,11,13,14,15,16,17,18,19,21,22,23],
#         resid_pdrop=0.1,
#         bidirectional=False,
#         is_bias=False
# )
# test_class = HybridDecoderLayers(mamba_config=mamba_config,layer_idx=1)
# for name,param in test_class.named_parameters():
#     if param.requires_grad:
#         print(name, param.data.shape, param.dtype)


