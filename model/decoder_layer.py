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

from mamba2.hybrid_mamba_layer import Mamba2
from mamba2.hybrid_mamba_config import PhiMambaConfig

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
    def __init__(self,
                 mamba_config: PhiMambaConfig,
                 layer_idx: int):
        """
        Args:

        """
        super().__init__()
        self.hidden_size = mamba_config.d_model
        self.layer_idx = layer_idx


        self.self_attn_mamba = Mamba2(d_model=mamba_config.d_model, d_xb=mamba_config.d_xb, d_inner=mamba_config.d_inner,
                                      layer_idx=layer_idx, bias=mamba_config.is_bias, **mamba_config.ssm_cfg)
        self.is_mamba_layer = True
        # print(f"INFO: Decoder layer {layer_idx} is configured to use Mamba2.")

        # MLP 和 LayerNorm 模块与原始 LlamaDecoderLayer 保持完全一致
        # 这确保了无论注意力机制如何，这些部分的权重都可以从 smolvlm-instruct 加载
        self.mlp = HybridLlamaMLP(mamba_config)
        self.input_layernorm = LlamaRMSNorm(mamba_config.d_model, eps=mamba_config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(mamba_config.d_model, eps=mamba_config.rms_norm_eps)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mamba.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            **kwargs
    ) :

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Mamba2 层的前向传播
        hidden_states, present_key_value = self.self_attn_mamba(
            hidden_states=hidden_states
        )

        hidden_states = residual + hidden_states
        # 全连接层 (MLP)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)

        if kwargs is None:
            return hidden_states, None, None
        else:
            past_key_value = kwargs.get("past_key_value", None)
            if past_key_value is not None:
                dummy_keys = torch.ones(
                    1, 1, hidden_states.size(1), 1, device=hidden_states.device, dtype=hidden_states.dtype
                )
                dummy_values = torch.ones(
                    1, 1, hidden_states.size(1), 1, device=hidden_states.device, dtype=hidden_states.dtype
                )
                # Update kv cache with dummy values
                past_key_value.update(dummy_keys, dummy_values, self.layer_idx)
            return hidden_states, None , past_key_value


