from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import Encoder, Predictor, TransformerConfig
from ssp import ThreeAxisSSP, ThreeAxisSSPConfig


def _model_config() -> tuple[TransformerConfig, ThreeAxisSSPConfig]:
    transformer_config = TransformerConfig(
        context_size=81,
        n_heads=4,
        head_dim=16,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
    )
    embedding_config = ThreeAxisSSPConfig(
        dim=transformer_config.d_model,
        seed=123,
    )
    return transformer_config, embedding_config


def test_encoder_outputs_expected_shape_and_backprop() -> None:
    b = 4
    transformer_config, embedding_config = _model_config()
    embedding = ThreeAxisSSP(embedding_config).to("cpu")
    model = Encoder(transformer_config, embedding=embedding).to("cpu")
    x = torch.randint(low=0, high=10, size=(b, 81, 3), dtype=torch.float32)

    out = model(x)
    assert out.shape == (b, 81, transformer_config.d_model)

    loss = out.pow(2).mean()
    loss.backward()

    assert any(
        param.grad is not None for param in model.parameters() if param.requires_grad
    )


def test_predictor_outputs_expected_shape_and_backprop() -> None:
    b = 3
    t = 12
    transformer_config, embedding_config = _model_config()
    embedding = ThreeAxisSSP(embedding_config).to("cpu")
    predictor = Predictor(transformer_config, embedding=embedding).to("cpu")

    query = torch.randint(low=0, high=10, size=(b, t, 3), dtype=torch.float32)
    encoder_out = torch.randn(
        b, 81, transformer_config.d_model, requires_grad=True, dtype=torch.float32
    )

    out = predictor(query, encoder_out)
    assert out.shape == (b, t, transformer_config.d_model)

    loss = out.abs().mean()
    loss.backward()

    assert any(
        param.grad is not None for param in predictor.parameters() if param.requires_grad
    )


def test_encoder_and_predictor_share_embedding_instance() -> None:
    transformer_config, embedding_config = _model_config()
    embedding = ThreeAxisSSP(embedding_config).to("cpu")

    encoder = Encoder(transformer_config, embedding=embedding).to("cpu")
    predictor = Predictor(transformer_config, embedding=embedding).to("cpu")

    assert encoder.embedding is predictor.embedding
