"""Loss helpers for soft-label knowledge distillation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    alpha: float,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute hard-label CE plus soft-label KL distillation loss.

    Args:
        student_logits: Student model logits with shape ``(batch_size, num_labels)``.
        teacher_logits: Teacher model logits with shape ``(batch_size, num_labels)``.
        labels: Hard original labels with shape ``(batch_size,)``.
        temperature: Distillation temperature.
        alpha: Weight assigned to soft distillation loss.
        class_weights: Optional class weights for hard cross-entropy.

    Returns:
        ``(total_loss, hard_ce_loss, soft_kl_loss)``.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")

    hard_ce_loss = F.cross_entropy(
        student_logits,
        labels,
        weight=class_weights,
    )
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    soft_kl_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    )
    total_loss = (
        (1 - alpha) * hard_ce_loss
        + alpha * (temperature ** 2) * soft_kl_loss
    )
    return total_loss, hard_ce_loss, soft_kl_loss
