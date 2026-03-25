import torch
from einops import einsum, reduce
from jaxtyping import Float
from torch import Tensor
import torch.nn.functional as F


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


def prototype_classification_loss(
    logits: Float[Tensor, "b t 9"],
    target_digits: Float[Tensor, "b t"] | Tensor,
) -> Float[Tensor, ""]:
    if logits.ndim != 3 or logits.shape[-1] != 9:
        raise ValueError(f"Expected logits shape (B, T, 9), got {tuple(logits.shape)}")

    if logits.shape[:2] != target_digits.shape[:2]:
        raise ValueError(
            f"Logits batch/sequence dims {tuple(logits.shape[:2])} "
            f"must match target dims {tuple(target_digits.shape[:2])}."
        )

    if target_digits.numel() == 0:
        return logits.sum() * 0.0

    target_classes = target_digits.to(dtype=torch.long) - 1
    min_class = int(target_classes.min().item())
    max_class = int(target_classes.max().item())
    if min_class < 0 or max_class > 8:
        raise ValueError(
            f"Target digits must lie in [1, 9], got class index range [{min_class}, {max_class}]."
        )

    return F.cross_entropy(logits.reshape(-1, 9), target_classes.reshape(-1))
