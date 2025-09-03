import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import  List, Tuple


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


class VLMDitillationLoss(nn.Module):
    """
    Implements a composite loss for teacher-student knowledge distillation in VLMs.
    This version returns a loss tensor with the same dtype as the model's output (e.g., bfloat16),
    trusting the Accelerator to handle the backward pass correctly.
    """

    def __init__(self, config: DistillationLossConfig):
        """
        Initializes the loss function with the given configuration.

        Args:
            config (DistillationLossConfig): An object containing the weights, temperature, and layer indices.
        """
        super().__init__()
        self.config = config

        self.kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.l2_loss_fn = nn.MSELoss()

    def forward(
            self,
            student_logits: torch.Tensor,
            teacher_logits: torch.Tensor,
            student_ce_loss: torch.Tensor,
            student_hidden_states: Tuple[torch.Tensor, ...],
            teacher_hidden_states: Tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Calculates the total distillation loss.
        """
        # 1. KL Divergence Loss with Temperature Scaling
        soft_student_logits = F.log_softmax(student_logits / self.config.temperature, dim=-1)
        soft_teacher_logits = F.softmax(teacher_logits / self.config.temperature, dim=-1)
        kl_loss = self.kl_loss_fn(soft_student_logits, soft_teacher_logits) * (self.config.temperature ** 2)

        # 2. L2 Distance Loss (MSE) on specified Hidden States
        l2_loss = self._calculate_multi_layer_l2_loss(student_hidden_states, teacher_hidden_states)

        # 3. Cross-Entropy Loss is passed in directly
        ce_loss = student_ce_loss

        # 4. Combine losses with weights.
        #    NOTE: We are NOT casting to float32 here. We let the loss tensor remain in the
        #    mixed-precision format (e.g., bfloat16) to allow `accelerate` to manage it.
        total_loss = (
                self.config.kl_weight * kl_loss +
                self.config.l2_weight * l2_loss +
                self.config.ce_weight * ce_loss
        )

        loss_components = {
            'kl_loss': kl_loss.detach(),
            'l2_loss': l2_loss.detach(),
            'ce_loss': ce_loss.detach(),
            'total_loss': total_loss.detach()  # This will now be bfloat16
        }

        return total_loss, loss_components

    def _calculate_multi_layer_l2_loss(
            self,
            student_hidden_states: Tuple[torch.Tensor, ...],
            teacher_hidden_states: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """Helper function to compute the L2 loss across specified layers."""
        if not isinstance(student_hidden_states, (list, tuple)) or not isinstance(teacher_hidden_states, (list, tuple)):
            raise TypeError("Hidden states must be provided as a list or tuple of tensors.")

        layer_indices = self.config.l2_loss_layers if self.config.l2_loss_layers else [-1]

        num_layers_for_loss = len(layer_indices)
        if num_layers_for_loss == 0:
            return torch.tensor(0.0, device=student_hidden_states[0].device)

        total_l2_loss = torch.tensor(0.0, device=student_hidden_states[0].device, dtype=student_hidden_states[0].dtype)
        for i in range(1, len(teacher_hidden_states)):
            if i - 1 in layer_indices:
                s_hidden = student_hidden_states[i]
                t_hidden = teacher_hidden_states[i]

                if s_hidden.shape != t_hidden.shape:
                    raise ValueError(
                        f"Student and Teacher hidden states for layer {i} must have the same shape. "
                        f"Got student: {s_hidden.shape}, teacher: {t_hidden.shape}"
                    )
                total_l2_loss += self.l2_loss_fn(s_hidden, t_hidden)

        return total_l2_loss / num_layers_for_loss


# ================== Example Usage ==================
if __name__ == '__main__':
    # This example requires a CUDA device to test bfloat16
    if torch.cuda.is_available():
        # Configuration
        batch_size = 8
        num_classes = 100
        hidden_dim = 768
        num_hidden_layers = 12

        loss_config = DistillationLossConfig(
            kl_weight=0.6,
            l2_weight=0.1,
            ce_weight=0.3,
            temperature=2.5,
            l2_loss_layers=[0, 6, -1]
        )
        distil_loss = VLMDitillationLoss(config=loss_config)

        # Simulate model outputs in bfloat16 by explicitly setting the dtype of dummy tensors.
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                t_out_logits = torch.randn(batch_size, num_classes, device='cuda', dtype=torch.bfloat16)
                t_out_hidden_states = tuple(
                    torch.randn(batch_size, hidden_dim, device='cuda', dtype=torch.bfloat16) for _ in
                    range(num_hidden_layers))

            s_out_logits = torch.randn(batch_size, num_classes, device='cuda', dtype=torch.bfloat16, requires_grad=True)
            s_out_hidden_states = tuple(
                torch.randn(batch_size, hidden_dim, device='cuda', dtype=torch.bfloat16, requires_grad=True) for _ in
                range(num_hidden_layers))
            s_out_loss = torch.tensor(1.2345, device='cuda', dtype=torch.bfloat16)  # Dummy scalar loss

            print(f"Input dtypes: s_out_logits={s_out_logits.dtype}, s_out_loss={s_out_loss.dtype}")

            total_loss, loss_components = distil_loss(
                student_logits=s_out_logits,
                teacher_logits=t_out_logits,
                student_ce_loss=s_out_loss,
                student_hidden_states=s_out_hidden_states,
                teacher_hidden_states=t_out_hidden_states
            )

        print(f"\nOutput total_loss dtype: {total_loss.dtype}")  # Should be torch.bfloat16
        assert total_loss.dtype == torch.bfloat16

        print("\nCalculated Losses:")
        for name, value in loss_components.items():
            print(f"  - {name:<12}: {value.item():.4f} (dtype: {value.dtype})")

        print("\nBackward pass should now succeed in a real training script.")
        print("Test case passed!")
    else:
        print("CUDA not available, skipping bfloat16 test case.")
