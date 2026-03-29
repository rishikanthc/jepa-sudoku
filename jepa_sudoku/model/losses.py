from __future__ import annotations

import torch
from einops import einsum, reduce
from jaxtyping import Float
from torch import Tensor


def cosine_loss(
    a: Float[Tensor, "b s d"],
    b: Float[Tensor, "b s d"],
    eps: float = 1e-8,
) -> Float[Tensor, ""]:
    a_norm = torch.linalg.norm(a, dim=-1, keepdim=True).clamp_min(eps)
    b_norm = torch.linalg.norm(b, dim=-1, keepdim=True).clamp_min(eps)

    a = a / a_norm
    b = b / b_norm

    cos_sim = einsum(a, b, "b s d, b s d -> b s")
    losses = 1.0 - cos_sim

    loss = reduce(losses, "b s ->", "mean")
    return loss


def cosine_contrastive_loss(
    pred_vectors: Float[Tensor, "b s d"],
    target_vectors: Float[Tensor, "b s d"],
    target_digits: Tensor,
    digit_prototypes: Float[Tensor, "v d"],
    *,
    non_target_weight: float = 1.0,
    margin: float = 0.2,
    eps: float = 1e-8,
) -> Float[Tensor, ""]:
    if non_target_weight < 0.0:
        raise ValueError("non_target_weight must be non-negative.")
    if margin < 0.0:
        raise ValueError("margin must be non-negative.")

    attraction = cosine_loss(pred_vectors, target_vectors, eps=eps)

    if pred_vectors.numel() == 0 or non_target_weight == 0.0:
        return attraction

    pred_norm = pred_vectors / torch.linalg.norm(
        pred_vectors, dim=-1, keepdim=True
    ).clamp_min(eps)
    prototypes_norm = digit_prototypes / torch.linalg.norm(
        digit_prototypes, dim=-1, keepdim=True
    ).clamp_min(eps)
    similarities = torch.einsum("bsd,vd->bsv", pred_norm, prototypes_norm)

    target_indices = target_digits.to(dtype=torch.long) - 1
    if target_indices.numel() > 0:
        min_class = int(target_indices.min().item())
        max_class = int(target_indices.max().item())
        if min_class < 0 or max_class >= digit_prototypes.shape[0]:
            raise ValueError(
                "target_digits must index valid digit prototypes."
            )

    wrong_mask = torch.ones_like(similarities, dtype=torch.bool)
    wrong_mask.scatter_(-1, target_indices.unsqueeze(-1), False)
    wrong_similarities = similarities.masked_select(wrong_mask).reshape(
        *similarities.shape[:2], similarities.shape[-1] - 1
    )
    repulsion = torch.relu(wrong_similarities - margin).mean()
    return attraction + (non_target_weight * repulsion)
