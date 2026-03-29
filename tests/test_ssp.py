from __future__ import annotations

import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jepa_sudoku.model.ssp import ThreeAxisSSP, ThreeAxisSSPConfig


def test_encode_decode_roundtrip_3d():
    rng = torch.Generator().manual_seed(123)
    coords = torch.randint(low=1, high=10, size=(5, 3), generator=rng).float()

    ssp = ThreeAxisSSP(ThreeAxisSSPConfig(dim=256, seed=42)).to("cpu")
    encoded = ssp.encode(coords)
    decoded, _ = ssp.decode(encoded)

    assert torch.equal(decoded, coords)


def test_encode_decode_roundtrip_with_one_zero_axis():
    coord_rng = torch.Generator().manual_seed(456)

    coords = torch.randint(low=1, high=10, size=(5, 3), generator=coord_rng).float()
    coords[:, 2] = 0.0

    ssp = ThreeAxisSSP(ThreeAxisSSPConfig(dim=256, seed=42)).to("cpu")
    encoded = ssp.encode(coords)
    decoded, _ = ssp.decode(encoded)

    assert torch.equal(decoded, coords)


def test_encoder_output_is_detached():
    xy = torch.randint(low=1, high=10, size=(2, 81, 2), dtype=torch.float32)
    z = torch.randint(low=0, high=10, size=(2, 81, 1), dtype=torch.float32)
    coords = torch.cat([xy, z], dim=-1).requires_grad_(True)
    ssp = ThreeAxisSSP(ThreeAxisSSPConfig(dim=256, seed=42)).to("cpu")

    encoded = ssp(coords)
    assert encoded.requires_grad is False

    # decode path is inference-only as well
    decoded, _ = ssp.decode(encoded)
    assert decoded.shape == coords.shape
    assert not decoded.requires_grad

    xy = coords[0, :, :2]
    z_hat, z_sim = ssp.decode_z_given_xy(encoded[0], xy)
    assert z_hat.requires_grad is False
    assert z_sim.requires_grad is False

    sims = ssp.similarity_to_codebook(encoded[0:2])
    assert sims.requires_grad is False
