from __future__ import annotations

import torch
from torch.nn.functional import cosine_similarity

from jepa_sudoku.model.losses import cosine_contrastive_loss, cosine_loss


def test_cosine_loss_matches_expected_values() -> None:
    eps = 1e-8

    # Perfect match -> zero loss
    a = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
    b = torch.tensor([[[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]])
    loss = cosine_loss(a, b, eps=eps)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    # Opposite vectors -> loss 2 per element => mean 2
    a = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    b = torch.tensor([[[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]])
    loss = cosine_loss(a, b, eps=eps)
    assert torch.isclose(loss, torch.tensor(2.0), atol=1e-6)

    # Orthogonal vectors -> loss 1 per element => mean 1
    a = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    b = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]])
    loss = cosine_loss(a, b, eps=eps)
    assert torch.isclose(loss, torch.tensor(1.0), atol=1e-6)


def test_cosine_loss_is_differentiable_and_shape_scalar() -> None:
    a = torch.randn(2, 4, 6, requires_grad=True)
    b = torch.randn(2, 4, 6, requires_grad=True)

    loss = cosine_loss(a, b)
    assert loss.dim() == 0  # scalar

    loss.backward()
    assert a.grad is not None and b.grad is not None


def test_cosine_loss_matches_torch_formula() -> None:
    a = torch.randn(3, 5, 8)
    b = torch.randn(3, 5, 8)
    eps = 1e-8

    expected = 1.0 - cosine_similarity(a, b, dim=-1, eps=eps)
    expected_loss = expected.mean()

    actual = cosine_loss(a, b, eps=eps)
    assert torch.allclose(actual, expected_loss, atol=1e-6)


def test_cosine_contrastive_loss_matches_cosine_loss_when_wrong_digits_are_far() -> None:
    pred = torch.tensor([[[1.0, 0.0, 0.0]]])
    target = torch.tensor([[[1.0, 0.0, 0.0]]])
    target_digits = torch.tensor([[1.0]])
    prototypes = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    loss = cosine_contrastive_loss(
        pred,
        target,
        target_digits,
        prototypes,
        non_target_weight=1.0,
        margin=0.2,
    )
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_cosine_contrastive_loss_penalizes_similarity_to_non_targets() -> None:
    pred = torch.tensor([[[1.0, 1.0, 0.0]]])
    target = torch.tensor([[[1.0, 0.0, 0.0]]])
    target_digits = torch.tensor([[1.0]])
    prototypes = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    plain = cosine_loss(pred, target)
    contrastive = cosine_contrastive_loss(
        pred,
        target,
        target_digits,
        prototypes,
        non_target_weight=1.0,
        margin=0.2,
    )
    assert contrastive > plain
