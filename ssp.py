from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


@dataclass
class ThreeAxisSSPConfig:
    """
    Configuration for a bounded 3-axis SSP.

    Coordinate ranges:
        x in [0, 9] (0 skips x)
        y in [0, 9] (0 skips y)
        z in [0, 9]

    dim:
        Final SSP dimensionality.
        Must be even because we use [cos, sin] pairs.

    seed:
        Random seed for reproducible wavevector generation.

    freq_scale_min / freq_scale_max:
        Range of magnitudes used for wavevectors.
        A mixture of low and high frequencies helps balance
        local resolution and global uniqueness.
    """

    dim: int = 256
    seed: int = 42
    freq_scale_min: float = 0.6
    freq_scale_max: float = 3.0


@dataclass
class SSPHypervectorStore:
    """
    Serialized bundle of the SSP base hypervectors.
    """

    config: ThreeAxisSSPConfig
    K: torch.Tensor
    coords: torch.Tensor
    codebook: torch.Tensor

    def to(self, device: str | torch.device) -> "SSPHypervectorStore":
        device = torch.device(device)
        return SSPHypervectorStore(
            config=self.config,
            K=self.K.to(device),
            coords=self.coords.to(device),
            codebook=self.codebook.to(device),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": 1,
                "config": asdict(self.config),
                "K": self.K.cpu(),
                "coords": self.coords.cpu(),
                "codebook": self.codebook.cpu(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "SSPHypervectorStore":
        loaded = torch.load(Path(path), map_location=device)
        return cls(
            config=ThreeAxisSSPConfig(**loaded["config"]),
            K=loaded["K"].to(device),
            coords=loaded["coords"].to(device),
            codebook=loaded["codebook"].to(device),
        )


class ThreeAxisSSP(nn.Module):
    """
    Semantic Spatial Pointer encoder/decoder for a bounded 3-axis discrete space.

    Representation:
        For m = dim // 2 frequency channels and coordinate c = [x, y, z],

            phase_j = <k_j, c>

        SSP(c) = [cos(phase_1), ..., cos(phase_m),
                  sin(phase_1), ..., sin(phase_m)]

    Decoding:
        Since the domain is small (9 * 9 * 10 = 810 states), we precompute
        a codebook for every valid coordinate and decode by max cosine similarity.

    Notes:
        - x, y are in [0, 9], where 0 skips that axis
        - z in [0, 9]
        - Supports batch encoding and decoding
    """

    def __init__(
        self,
        config: ThreeAxisSSPConfig,
        *,
        store: SSPHypervectorStore | None = None,
    ):
        super().__init__()
        self.config = config
        assert config.dim % 2 == 0, "dim must be even"

        self.m = config.dim // 2  # number of frequency channels

        if store is None:
            # Build wavevector matrix K of shape (m, 3)
            K = self._make_wavevectors(
                m=self.m,
                seed=config.seed,
                scale_min=config.freq_scale_min,
                scale_max=config.freq_scale_max,
            )  # (m, 3)

            # Build the bounded coordinate codebook
            coords = self._make_all_valid_coords()  # (810, 3)
            codebook = self._encode_with_k(coords=coords, K=K, dim=config.dim)  # (810, dim)
        else:
            if store.config != config:
                raise ValueError(
                    "Store configuration does not match provided config. "
                    "Pass the store's config or use from_hypervector_store."
                )
            self._validate_store(store)
            K = store.K
            coords = store.coords
            codebook = store.codebook

        self.register_buffer("K", K)
        self.register_buffer("coords", coords)
        self.register_buffer("codebook", codebook)

    @property
    def device(self) -> torch.device:
        return self.K.device

    @torch.no_grad()
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Forward alias for encoding.

        Accepts:
            coords: (B, 81, 3) or any shape (..., 3) coordinates.
        Returns:
            (B, 81, dim) or (..., dim) tensor.
        """
        return self.encode(coords)

    def hypervector_store(self) -> SSPHypervectorStore:
        """
        Return the exact current base hypervectors used by this instance.
        """
        return SSPHypervectorStore(
            config=self.config,
            K=self.K.detach().cpu(),
            coords=self.coords.detach().cpu(),
            codebook=self.codebook.detach().cpu(),
        )

    def save_hypervectors(self, path: str | Path) -> None:
        """
        Persist base hypervectors so encoding/decoding can be resumed later
        with the same exact basis.
        """
        self.hypervector_store().save(path)

    @classmethod
    def from_hypervector_store(
        cls,
        path_or_store: str | Path | SSPHypervectorStore,
        *,
        device: str | torch.device = "cpu",
    ) -> "ThreeAxisSSP":
        """
        Restore an SSP instance from a serialized or in-memory bundle.
        """
        if isinstance(path_or_store, (str, Path)):
            store = SSPHypervectorStore.load(path_or_store, device=device)
        else:
            store = path_or_store.to(device)
        return cls(store.config, store=store)

    @staticmethod
    def _validate_store(store: SSPHypervectorStore) -> None:
        if store.config.dim % 2 != 0:
            raise ValueError("Invalid store: embedded config has odd dim")
        if store.config.dim // 2 != store.K.shape[0]:
            raise ValueError(
                "Invalid store: K first dim must match dim/2 from stored config"
            )
        if store.codebook.shape[1] != store.config.dim:
            raise ValueError("Invalid store: codebook dim must match stored config dim")
        if store.K.ndim != 2 or store.K.shape[1] != 3:
            raise ValueError("Invalid store: K must have shape (m, 3)")
        if store.coords.ndim != 2 or store.coords.shape[1] != 3:
            raise ValueError("Invalid store: coords must have shape (N, 3)")
        if store.codebook.ndim != 2:
            raise ValueError("Invalid store: codebook must be 2D")

    @staticmethod
    def _make_wavevectors(
        m: int,
        seed: int,
        scale_min: float,
        scale_max: float,
    ) -> torch.Tensor:
        """
        Create diverse 3D wavevectors K in R^{m x 3}.

        - Sample random directions on the unit sphere
        - Sample magnitudes across a range
        - Multiply direction * magnitude

        This gives a mixture of spatial frequencies in multiple directions.
        """
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)

        # Random directions: normalize Gaussian samples
        dirs = torch.randn(m, 3, generator=g)
        dirs = F.normalize(dirs, dim=-1)

        # Smooth spread of magnitudes
        # Use linearly spaced magnitudes, then shuffle a bit for diversity
        mags = torch.linspace(scale_min, scale_max, steps=m)
        perm = torch.randperm(m, generator=g)
        mags = mags[perm]

        K = dirs * mags[:, None]  # (m, 3)
        return K

    @staticmethod
    def _make_all_valid_coords() -> torch.Tensor:
        """
        Enumerate all valid coordinates.

        Returns:
            coords: (810, 3) tensor where each row is [x, y, z]
        """
        xs = torch.arange(1, 10)
        ys = torch.arange(1, 10)
        zs = torch.arange(0, 10)

        grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs, indexing="ij")
        coords = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (9, 9, 10, 3)
        coords = rearrange(coords, "x y z c -> (x y z) c").float()
        return coords

    def _validate_coords(self, coords: torch.Tensor) -> None:
        """
        Validate bounded coordinates.

        coords: (..., 3)
        """
        if coords.shape[-1] != 3:
            raise ValueError(
                f"Expected coords shape (..., 3), got {tuple(coords.shape)}"
            )

        x = coords[..., 0]
        y = coords[..., 1]
        z = coords[..., 2]

        valid = (
            (x >= 0)
            & (x <= 9)
            & (y >= 0)
            & (y <= 9)
            & (z >= 0)
            & (z <= 9)
        )

        if not torch.all(valid).item():
            bad = coords[~valid]
            raise ValueError(
                "Found out-of-range coordinates. "
                "Expected x,y in [0,9] where 0 means skip, and z in [0,9]. "
                f"Examples of invalid rows: {bad[:5]}"
            )

    @torch.no_grad()
    def encode(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Encode coordinates into SSP vectors.

        Args:
            coords: (..., 3) tensor of [x, y, z]

        Returns:
            ssp: (..., dim) normalized SSP vectors
        """
        coords = coords.to(dtype=self.K.dtype, device=self.device)
        squeeze_batch = coords.ndim == 1
        if squeeze_batch:
            coords = coords[None, :]
        self._validate_coords(coords)
        encoded = self._encode_with_k(
            coords=coords,
            K=self.K,
            dim=self.config.dim,
        )
        if squeeze_batch:
            return encoded[0]
        return encoded

    def _encode_with_k(
        self,
        coords: torch.Tensor,
        K: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        # Flatten leading dimensions and compute all phases in one matmul.
        flat_coords = rearrange(coords, "... c -> (...) c")
        phase = torch.einsum("...c,mc->...m", flat_coords, K)

        cos_part = torch.cos(phase)
        sin_part = torch.sin(phase)

        ssp = torch.cat([cos_part, sin_part], dim=-1)
        ssp = F.normalize(ssp, dim=-1)

        out_shape = coords.shape[:-1] + (dim,)
        return ssp.reshape(out_shape)

    @torch.no_grad()
    def similarity_to_codebook(self, ssp: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between input SSPs and all valid codebook entries.

        Args:
            ssp: (..., dim)

        Returns:
            sims: (..., 810)
        """
        ssp = ssp.to(dtype=self.K.dtype, device=self.device)
        batch_shape = ssp.shape[:-1]

        flat_ssp = rearrange(ssp, "... d -> (...) d")
        flat_ssp = F.normalize(flat_ssp, dim=-1)

        # codebook is already normalized
        sims = flat_ssp @ self.codebook.T  # (B, 810)
        sims = sims.reshape(batch_shape + (self.codebook.shape[0],))
        return sims

    @torch.no_grad()
    def decode(self, ssp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full decode: recover [x, y, z] by nearest codebook match.

        Args:
            ssp: (..., dim)

        Returns:
            decoded_coords: (..., 3)
            best_similarity: (...)
        """
        sims = self.similarity_to_codebook(ssp)  # (..., 810)

        flat_sims = sims.reshape(-1, sims.shape[-1])
        best_idx = flat_sims.argmax(dim=-1)  # (B,)
        best_sim = flat_sims.gather(1, best_idx[:, None]).squeeze(1)

        decoded = self.coords[best_idx]  # (B, 3)

        decoded = decoded.reshape(sims.shape[:-1] + (3,))
        best_sim = best_sim.reshape(sims.shape[:-1])
        return decoded, best_sim

    @torch.no_grad()
    def decode_z_given_xy(
        self,
        ssp: torch.Tensor,
        xy: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode only z, assuming x and y are known.

        Args:
            ssp: (..., dim)
            xy:  (..., 2), with x,y in [1,9]

        Returns:
            z_hat: (...,) tensor of decoded z values in [0,9]
            best_similarity: (...,)
        """
        ssp = ssp.to(dtype=self.K.dtype, device=self.device)
        xy = xy.to(dtype=self.K.dtype, device=self.device)

        if xy.shape[-1] != 2:
            raise ValueError(f"Expected xy shape (..., 2), got {tuple(xy.shape)}")

        x = xy[..., 0]
        y = xy[..., 1]
        valid = (x >= 1) & (x <= 9) & (y >= 1) & (y <= 9)
        if not torch.all(valid).item():
            bad = xy[~valid]
            raise ValueError(
                f"Found out-of-range x/y. Expected x,y in [1,9]. Examples: {bad[:5]}"
            )

        batch_shape = xy.shape[:-1]
        flat_ssp = rearrange(ssp, "... d -> (...) d")
        flat_ssp = F.normalize(flat_ssp, dim=-1)
        flat_xy = rearrange(xy, "... c -> (...) c")

        if flat_ssp.shape[0] != flat_xy.shape[0]:
            raise ValueError(
                "decode_z_given_xy received mismatched batch sizes. "
                f"Got ssp batch {tuple(flat_ssp.shape[:-1])}, xy batch {tuple(flat_xy.shape[:-1])} "
                "after flattening to 2D."
            )

        B = flat_xy.shape[0]
        z_candidates = torch.arange(0, 10, device=self.device).float()  # (10,)

        # Build candidate coordinates:
        # For each batch item, pair known (x,y) with z in [0..9]
        # Result shape: (B, 10, 3)
        xy_expanded = repeat(flat_xy, "b c -> b z c", z=10)  # (B, 10, 2)
        z_expanded = repeat(z_candidates, "z -> b z 1", b=B)  # (B, 10, 1)
        candidate_coords = torch.cat([xy_expanded, z_expanded], dim=-1)  # (B, 10, 3)

        candidate_ssp = self.encode(candidate_coords)  # (B, 10, dim)

        # Similarity per candidate z:
        # (B, 1, dim) * (B, 10, dim) -> (B, 10)
        sims = torch.einsum("bd,bzd->bz", flat_ssp, candidate_ssp)

        best_idx = sims.argmax(dim=-1)  # (B,)
        best_sim = sims.gather(1, best_idx[:, None]).squeeze(1)
        z_hat = z_candidates[best_idx].long()

        z_hat = z_hat.reshape(batch_shape)
        best_sim = best_sim.reshape(batch_shape)
        return z_hat, best_sim

    @torch.no_grad()
    def make_noisy_copy(
        self, ssp: torch.Tensor, noise_std: float = 0.05
    ) -> torch.Tensor:
        """
        Utility for testing decoder robustness.
        """
        noisy = ssp + noise_std * torch.randn_like(ssp)
        return F.normalize(noisy, dim=-1)


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Example usage
    # ------------------------------------------------------------
    torch.set_printoptions(precision=4, sci_mode=False)

    config = ThreeAxisSSPConfig(
        dim=256,
        seed=42,
        freq_scale_min=0.6,
        freq_scale_max=3.0,
    )
    ssp = ThreeAxisSSP(config).to("cpu")

    # ----------------------------
    # 1) Encode a single coordinate
    # ----------------------------
    coord = torch.tensor([3, 7, 5]).float()  # [x, y, z]
    vec = ssp.encode(coord)  # (256,)
    print("Encoded vector shape:", vec.shape)

    # ----------------------------
    # 2) Full decode
    # ----------------------------
    decoded_coord, sim = ssp.decode(vec)
    print("Original coord:", coord.long())
    print("Decoded coord :", decoded_coord.long())
    print("Similarity    :", sim.item())

    # ----------------------------
    # 3) Decode z only, given x and y
    # ----------------------------
    xy = torch.tensor([3, 7]).float()
    z_hat, z_sim = ssp.decode_z_given_xy(vec, xy)
    print("Decoded z given x,y:", z_hat.item(), "| similarity:", z_sim.item())

    # ----------------------------
    # 4) Batched encoding / decoding
    # ----------------------------
    batch_coords = torch.tensor(
        [
            [1, 1, 1],
            [9, 9, 9],
            [4, 2, 6],
            [1, 7, 3],
        ]
    ).float()

    batch_vecs = ssp.encode(batch_coords)  # (B, 256)
    batch_decoded, batch_sims = ssp.decode(batch_vecs)  # (B, 3), (B,)

    print("\nBatch original:")
    print(batch_coords.long())
    print("Batch decoded:")
    print(batch_decoded.long())
    print("Batch similarities:")
    print(batch_sims)

    # ----------------------------
    # 5) Noisy decode demo
    # ----------------------------
    noisy_vec = ssp.make_noisy_copy(vec, noise_std=0.08)
    noisy_decoded, noisy_sim = ssp.decode(noisy_vec)
    print("\nNoisy decode:")
    print("Decoded coord :", noisy_decoded.long())
    print("Similarity    :", noisy_sim.item())
