import torch
from einops import einsum, reduce
from torch import Tensor
from jaxtyping import Float


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
