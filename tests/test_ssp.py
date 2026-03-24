from __future__ import annotations

import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ssp import ThreeAxisSSP, ThreeAxisSSPConfig


def test_encode_decode_roundtrip_3d():
    rng = torch.Generator().manual_seed(123)
    coords = torch.randint(low=1, high=10, size=(5, 3), generator=rng).float()

    ssp = ThreeAxisSSP(ThreeAxisSSPConfig(dim=256, seed=42)).to("cpu")
    encoded = ssp.encode(coords)
    decoded, _ = ssp.decode(encoded)

    assert torch.equal(decoded, coords)


def test_encode_decode_roundtrip_with_one_zero_axis():
    coord_rng = torch.Generator().manual_seed(456)
    axis_rng = torch.Generator().manual_seed(789)

    coords = torch.randint(low=1, high=10, size=(5, 3), generator=coord_rng).float()
    zeros = torch.randint(low=0, high=3, size=(5,), generator=axis_rng)

    for i in range(coords.shape[0]):
        coords[i, zeros[i]] = 0.0

    ssp = ThreeAxisSSP(ThreeAxisSSPConfig(dim=256, seed=42)).to("cpu")
    encoded = ssp.encode(coords)
    decoded, _ = ssp.decode(encoded)

    assert torch.equal(decoded, coords)
