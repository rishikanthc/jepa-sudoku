from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ssp import SSPHypervectorStore, ThreeAxisSSP, ThreeAxisSSPConfig


def _coords() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [4.0, 5.0, 0.0],
            [9.0, 9.0, 9.0],
        ],
        dtype=torch.float32,
    )


def test_save_and_reload_hypervector_store(tmp_path: Path) -> None:
    config = ThreeAxisSSPConfig(dim=256, seed=42)
    ssp = ThreeAxisSSP(config).to("cpu")

    sample = _coords()
    encoded = ssp.encode(sample)

    out = tmp_path / "three_axis_ssp_bundle.pt"
    ssp.save_hypervectors(out)

    restored = ThreeAxisSSP.from_hypervector_store(out, device="cpu")
    restored_encoded = restored.encode(sample)

    assert torch.equal(ssp.hypervector_store().config, restored.hypervector_store().config)
    assert torch.allclose(encoded, restored_encoded, atol=0.0, rtol=0.0)
    assert torch.equal(restored.decode(encoded)[0], sample)


def test_roundtrip_through_loaded_store_object(tmp_path: Path) -> None:
    config = ThreeAxisSSPConfig(dim=128, seed=11)
    ssp = ThreeAxisSSP(config).to("cpu")
    store = ssp.hypervector_store()

    bundle_file = tmp_path / "bundle.pt"
    store.save(bundle_file)
    loaded = SSPHypervectorStore.load(bundle_file, device="cpu")

    rebuilt = ThreeAxisSSP.from_hypervector_store(loaded, device="cpu")
    sample = _coords()
    assert torch.equal(rebuilt.decode(ssp.encode(sample))[0], sample)
