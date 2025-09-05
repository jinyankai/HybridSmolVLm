import torch
import torch.nn as nn
from typing import Tuple, Optional, Sequence

def _standardize(h: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = h.mean(dim=-1, keepdim=True)
    var  = h.var(dim=-1, keepdim=True, unbiased=False)
    return (h - mean) / (var + eps).sqrt()

class VLMDitillationLoss(nn.Module):
    """
    KL(teacher||student) + 同尺度 L2(选定层) + CE
    - 支持 attention_mask：KL/L2 仅在有效 token 上平均
    - 支持 Top-k KD（可开关）
    - 内部用 fp32 计算，再 cast 回输入 dtype
    - teacher hidden 在内部 detach
    """

    def __init__(self, config,
                 use_topk: bool = False,
                 k_top: int = 50,
                 includes_embedding_slot: bool = True,
                 z_loss_alpha: float = 0.0):
        super().__init__()
        self.config = config
        self.use_topk = use_topk
        self.k_top = k_top
        self.includes_embedding_slot = includes_embedding_slot
        self.z_loss_alpha = z_loss_alpha
        self.mse = nn.MSELoss(reduction="none")  # 我们自己做按mask平均

    # ---------- 公共入口 ----------
    def forward(
        self,
        student_logits: torch.Tensor,           # [B,T,V] 或 [B,V]
        teacher_logits: torch.Tensor,           # [B,T,V] 或 [B,V]
        student_ce_loss: torch.Tensor,          # 标准 CE 已按 ignore_index 算好
        student_hidden_states: Tuple[torch.Tensor, ...],  # 每层 [B,T,D]
        teacher_hidden_states: Tuple[torch.Tensor, ...],
        attention_mask: Optional[torch.Tensor] = None,    # [B,T] 0/1
    ):
        dtype_out = student_logits.dtype

        # ---- KL (soft-CE) with temperature & token mask ----
        T = float(self.config.temperature)
        kd_loss = self._kd_loss_masked(student_logits, teacher_logits, attention_mask, T)

        # ---- L2 on selected layers (standardized + mask) ----
        l2_loss = self._multi_layer_l2_masked(
            student_hidden_states, teacher_hidden_states, attention_mask,
            layers=self.config.l2_loss_layers or [-1],
            includes_embedding_slot=self.includes_embedding_slot
        )

        # ---- z-loss（可选）----
        z_loss = self._z_loss(student_logits, attention_mask, alpha=self.z_loss_alpha)

        # ---- 组合 ----
        total = (
            self.config.kl_weight * kd_loss +
            self.config.l2_weight * l2_loss +
            self.config.ce_weight * student_ce_loss +
            z_loss
        )

        comps = {
            "kl_loss": kd_loss.detach().to(dtype_out),
            "l2_loss": l2_loss.detach().to(dtype_out),
            "ce_loss": student_ce_loss.detach().to(dtype_out),
            "z_loss": z_loss.detach().to(dtype_out),
            "total_loss": total.detach().to(dtype_out),
        }
        return total.to(dtype_out), comps

    # ---------- 细节实现 ----------
    def _kd_loss_masked(self, logits_s, logits_t, attn_mask, T: float):
        # 统一到 [B,T,V]
        if logits_s.dim() == 2:
            logits_s = logits_s.unsqueeze(1)
            logits_t = logits_t.unsqueeze(1)
            if attn_mask is None:
                attn_mask = torch.ones(logits_s.size(0), 1, device=logits_s.device, dtype=torch.long)

        # fp32 计算更稳
        ls = logits_s.float() / T
        lt = logits_t.float() / T

        if self.use_topk:
            # Top-k KD
            val_t, idx_t = torch.topk(lt, k=min(self.k_top, lt.size(-1)), dim=-1)
            with torch.no_grad():
                q_top = torch.softmax(val_t, dim=-1)              # teacher 概率
            sel = torch.gather(ls, -1, idx_t)
            log_p_top = torch.log_softmax(sel, dim=-1)            # student log prob
            per_tok = -(q_top * log_p_top).sum(dim=-1)            # [B,T]
        else:
            with torch.no_grad():
                q = torch.softmax(lt, dim=-1)                     # teacher 概率
            log_p = torch.log_softmax(ls, dim=-1)                 # student log prob
            per_tok = -(q * log_p).sum(dim=-1)                    # [B,T]

        if attn_mask is None:
            return per_tok.mean() * (T * T)

        valid = attn_mask.float()
        denom = valid.sum().clamp_min(1.0)
        return (per_tok * valid).sum() / denom * (T * T)

    def _multi_layer_l2_masked(
        self,
        hs_s: Tuple[torch.Tensor, ...],
        hs_t: Tuple[torch.Tensor, ...],
        attn_mask: Optional[torch.Tensor],
        layers: Sequence[int],
        includes_embedding_slot: bool = True,
        eps: float = 1e-6,
    ):
        # 统一：层索引 0-based；如包含 embedding 槽位，则实际访问 idx+1
        L = len(hs_s) - (1 if includes_embedding_slot else 0)
        # 处理负索引
        abs_layers = []
        for idx in layers:
            if idx < 0:
                idx = L + idx
            if not (0 <= idx < L):
                raise IndexError(f"L2 layer index {idx} out of range [0,{L-1}]")
            abs_layers.append(idx)

        off = 1 if includes_embedding_slot else 0
        total, count = 0.0, 0

        for idx in abs_layers:
            s = hs_s[idx + off].float()
            t = hs_t[idx + off].float().detach()

            # [B,T,D] 约定；若 [B,D]，升成 [B,1,D]
            if s.dim() == 2:
                s = s.unsqueeze(1); t = t.unsqueeze(1)
                if attn_mask is None:
                    attn_mask = torch.ones(s.size(0), 1, device=s.device, dtype=torch.long)

            s = _standardize(s, eps)
            t = _standardize(t, eps)

            # 先对 D 求均值 → [B,T]
            mse_tok = self.mse(s, t).mean(dim=-1)

            if attn_mask is not None:
                valid = attn_mask.float()
                denom = valid.sum().clamp_min(1.0)
                loss_i = (mse_tok * valid).sum() / denom
            else:
                loss_i = mse_tok.mean()

            total += loss_i
            count += 1

        return total / max(count, 1)

    def _z_loss(self, logits_s, attn_mask, alpha: float = 0.0):
        if alpha <= 0:
            return logits_s.new_tensor(0.0)
        if logits_s.dim() == 2:
            logits_s = logits_s.unsqueeze(1)
        z = torch.logsumexp(logits_s.float(), dim=-1)     # [B,T]
        if attn_mask is None:
            return (z.pow(2).mean()) * alpha
        valid = attn_mask.float()
        denom = valid.sum().clamp_min(1.0)
        return ((z.pow(2) * valid).sum() / denom) * alpha

from dataclasses import dataclass, field
from typing import List
@dataclass
class DistillationLossConfig:
    """
    Configuration class for VLM Distillation Loss.

    Attributes:
        kl_weight (float): The weight for the KL divergence loss between teacher and student logits.
        l2_weight (float): The weight for the L2 distance (MSE) loss between teacher and student hidden states.
        ce_weight (float): The weight for the cross-entropy loss between student predictions and ground truth.
        temperature (float): The temperature scaling factor for KL divergence. Must be greater than 0.
        l2_loss_layers (List[int]): A list of layer indices to use for calculating the L2 hidden state loss.
                                    If None or empty, it defaults to using only the last layer's hidden state.
                                    e.g., [0, 6, 11] for a 12-layer model.
    """
    kl_weight: float = 0.5
    l2_weight: float = 0.1
    ce_weight: float = 0.4
    temperature: float = 2.0
    l2_loss_layers: List[int] = field(default_factory=lambda: [-1])

    def __post_init__(self):
        """Validate that the parameters are sensible."""
        if self.temperature <= 0:
            raise ValueError("Temperature must be positive.")