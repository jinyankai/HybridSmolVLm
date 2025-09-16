import torch
import torch.nn as nn
from typing import Tuple, Optional, Sequence,Dict



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
        self.warm_up_step = 83133

    # ---------- 公共入口 ----------
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_ce_loss: Optional[torch.Tensor],
        distillation_alignment_outputs: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
        attention_mask: Optional[torch.Tensor] = None,
        step : int =None
    ):
        dtype_out = student_logits.dtype

        new_kl_weight = self.kl_weight_scheduler(step) if step is not None else self.config.kl_weight
        # ---- 计算各个损失分量 ----
        # KL Loss
        kl_loss = self._kd_loss_masked(student_logits, teacher_logits, float(self.config.temperature))

        # L2 Loss
        l2_loss = torch.tensor(0.0, device=student_logits.device)
        if distillation_alignment_outputs:
            l2_loss = self._l2_loss_aligned(distillation_alignment_outputs)

        # Z-Loss
        z_loss = self._z_loss(student_logits, attention_mask, alpha=self.z_loss_alpha)

        # ---- 使用配置中的权重组合总损失 ----
        # 确保 student_ce_loss 不为 None
        if student_ce_loss is None:
            # 如果不提供CE loss，可以认为其权重为0或抛出错误
            ce_loss_component = torch.tensor(0.0, device=student_logits.device)
        else:
            ce_loss_component = student_ce_loss

        # 这是用于反向传播的总损失
        total_loss = (new_kl_weight * kl_loss +
                      self.config.l2_weight * l2_loss +
                      self.config.ce_weight * ce_loss_component +
                      z_loss) # z_loss 自带了 alpha 权重

        # ---- 准备用于日志记录的字典 ----
        comps = {
            "kl_loss": kl_loss.detach().to(dtype_out),
            "l2_loss": l2_loss.detach().to(dtype_out),
            "ce_loss": ce_loss_component.detach().to(dtype_out),
            "z_loss": z_loss.detach().to(dtype_out),
            "total_loss": total_loss.detach().to(dtype_out),
            "kl_weight": new_kl_weight
        }
        return total_loss.to(dtype_out), comps

    # ---------- 细节实现 ----------
    def _kd_loss_masked(self, logits_s, logits_t, T: float,attn_mask=None):
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

    def _l2_loss_aligned(
            self,
            align_outputs: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
            kd_mask: Optional[torch.Tensor] = None,
            eps: float = 1e-6,
    ):
        if not align_outputs:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        total_mse = 0.0
        for layer_idx, (s_out, t_out) in align_outputs.items():
            s = s_out.float()
            t = t_out.detach().float()

            # 标准化 (与你之前的逻辑保持一致)


            # Token 级 MSE -> [B, T]
            mse_tok = (s - t).pow(2).mean(dim=-1)

            # 掩码聚合
            if kd_mask is not None:
                valid = kd_mask.to(mse_tok.dtype)
                # 保证 mask 和 mse_tok 长度一致
                T_min = min(valid.size(1), mse_tok.size(1))
                valid_trunc = valid[:, :T_min]
                mse_tok_trunc = mse_tok[:, :T_min]

                denom = valid_trunc.sum().clamp_min(1.0)
                loss_i = (mse_tok_trunc * valid_trunc).sum() / denom
            else:
                loss_i = mse_tok.mean()

            total_mse += loss_i

        return total_mse

    def _multi_layer_l2_masked(
            self,
            hs_s: Tuple[torch.Tensor, ...],
            hs_t: Tuple[torch.Tensor, ...],
            kd_mask: Optional[torch.Tensor],  # 建议用 (labels != -100).to(dtype)
            layers: Sequence[int],
            includes_embedding_slot: bool = True,
            eps: float = 1e-6,
    ):
        """
        对多层 hidden state 做 token-wise 标准化后 L2 (MSE) 蒸馏。
        - hs_*: 每个元素形状约定为 [B, T, D]，若为 [B, D] 会在内部升维到 [B,1,D]
        - kd_mask: [B, T]，1 表示参与蒸馏、0 表示跳过（请勿使用 raw attention_mask）
        - layers: 使用 0-based 的 block 索引；若 includes_embedding_slot=True，则会在访问时 +1
        """
        # 计算可用 block 层数（不含 embedding 槽位）
        L = len(hs_s) - (1 if includes_embedding_slot else 0)

        # 处理负索引并边界检查
        abs_layers = []
        for idx in layers:
            if idx < 0:
                idx = L + idx
            if not (0 <= idx < L):
                raise IndexError(f"L2 layer index {idx} out of range [0,{L - 1}]")
            abs_layers.append(idx)

        off = 1 if includes_embedding_slot else 0
        total, count = 0.0, 0

        for idx in abs_layers:
            s = hs_s[idx + off]
            t = hs_t[idx + off]

            # 统一到 [B,T,D]
            if s.dim() == 2:
                s = s.unsqueeze(1)  # [B,1,D]
            if t.dim() == 2:
                t = t.unsqueeze(1)

            # 显式断言形状一致（至少 B,T 一致）
            assert s.shape[:2] == t.shape[:2], f"Hidden state shape mismatch at layer {idx}: {s.shape} vs {t.shape}"

            # 本层使用局部 mask，避免修改外部 kd_mask
            if kd_mask is not None:
                local_mask = kd_mask
                if local_mask.dim() == 1:
                    local_mask = local_mask.unsqueeze(1)  # [B] -> [B,1]
                # 若当前层的 T 与 mask 不一致，尝试截断到最小长度（更稳妥是保证预处理阶段严格对齐）
                if local_mask.size(1) != s.size(1):
                    T_min = min(local_mask.size(1), s.size(1))
                    s = s[:, :T_min]
                    t = t[:, :T_min]
                    local_mask = local_mask[:, :T_min]
            else:
                local_mask = None

            # 数值稳定与尺度对齐（最后一维 D 上标准化）
            s = s.float()
            t = t.float()

            s_mean = s.mean(dim=-1, keepdim=True)
            t_mean = t.mean(dim=-1, keepdim=True)
            s_std = s.std(dim=-1, keepdim=True).clamp_min(eps)
            t_std = t.std(dim=-1, keepdim=True).clamp_min(eps)
            s = (s - s_mean) / s_std
            t = (t - t_mean) / t_std

            # token 级 MSE -> [B,T]
            mse_tok = (s - t).pow(2).mean(dim=-1)

            # 掩码聚合
            if local_mask is not None:
                valid = local_mask.to(mse_tok.dtype)
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

    def kl_weight_scheduler(self,step : int,
                            ):
        return self.config.kl_weight * (1 + 1 * step / self.warm_up_step) if self.warm_up_step > 0 else self.config.kl_weight

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