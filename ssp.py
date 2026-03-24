from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


@dataclass
class ThreeAxisSSPConfig:
    """
    Configuration for a bounded 3-axis SSP.

    Coordinate ranges:
        x in [0, 9]
        y in [0, 9]
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


class ThreeAxisSSP:
    """
    Semantic Spatial Pointer encoder/decoder for a bounded 3-axis discrete space.

    Representation:
        For m = dim // 2 frequency channels and coordinate c = [x, y, z],

            phase_j = <k_j, c>

        SSP(c) = [cos(phase_1), ..., cos(phase_m),
                  sin(phase_1), ..., sin(phase_m)]

    Decoding:
        Since the domain is small (10 * 10 * 10 = 1000 states), we precompute
        a codebook for every valid coordinate and decode by max cosine similarity.

    Notes:
        - x, y, z are in [0, 9]
        - Supports batch encoding and decoding
    """

    def __init__(self, config: ThreeAxisSSPConfig):
        self.config = config
        assert config.dim % 2 == 0, "dim must be even"
        self.device = torch.device("cpu")

        self.m = config.dim // 2  # number of frequency channels

        # Build wavevector matrix K of shape (m, 3)
        self.K = self._make_wavevectors(
            m=self.m,
            seed=config.seed,
            scale_min=config.freq_scale_min,
            scale_max=config.freq_scale_max,
        )  # (m, 3)

        # Build the bounded coordinate codebook
        self.coords = self._make_all_valid_coords()  # (1000, 3)
        self.codebook = self.encode(self.coords)  # (1000, dim)

    def to(self, device: torch.device | str) -> "ThreeAxisSSP":
        """
        Move model tensors to device.
        """
        self.device = torch.device(device)
        self.K = self.K.to(self.device)
        self.coords = self.coords.to(self.device)
        self.codebook = self.codebook.to(self.device)
        return self

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
            coords: (1000, 3) tensor where each row is [x, y, z]
        """
        xs = torch.arange(0, 10)
        ys = torch.arange(0, 10)
        zs = torch.arange(0, 10)

        grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs, indexing="ij")
        coords = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (10, 10, 10, 3)
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

        valid = (x >= 0) & (x <= 9) & (y >= 0) & (y <= 9) & (z >= 0) & (z <= 9)

        if not torch.all(valid):
            bad = coords[~valid]
            raise ValueError(
                "Found out-of-range coordinates. "
                "Expected x,y in [0,9] and z in [0,9]. "
                f"Examples of invalid rows: {bad[:5]}"
            )

    def encode(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Encode coordinates into SSP vectors.

        Args:
            coords: (..., 3) tensor of [x, y, z]

        Returns:
            ssp: (..., dim) normalized SSP vectors
        """
        coords = coords.to(self.K.dtype).to(self.device)
        self._validate_coords(coords)

        # Flatten batch for easier math
        flat_coords = coords.reshape(-1, coords.shape[-1])  # (B, 3)

        # phase = coords @ K^T
        # (B, 3) @ (3, m) -> (B, m)
        phase = flat_coords @ self.K.T

        cos_part = torch.cos(phase)
        sin_part = torch.sin(phase)

        ssp = torch.cat([cos_part, sin_part], dim=-1)  # (B, dim)
        ssp = F.normalize(ssp, dim=-1)

        # Restore original batch shape
        # `einops` versions differ on parenthesized-ellipsis handling,
        # so restore shape with a direct reshape.
        out_shape = coords.shape[:-1] + (self.config.dim,)
        ssp = ssp.reshape(out_shape)
        return ssp

    def similarity_to_codebook(self, ssp: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between input SSPs and all valid codebook entries.

        Args:
            ssp: (..., dim)

        Returns:
            sims: (..., 1000)
        """
        ssp = ssp.to(self.device)
        batch_shape = ssp.shape[:-1]

        flat_ssp = ssp.reshape(-1, ssp.shape[-1])
        flat_ssp = F.normalize(flat_ssp, dim=-1)

        # codebook is already normalized
        sims = flat_ssp @ self.codebook.T  # (B, 1000)
        sims = sims.reshape(batch_shape + (self.codebook.shape[0],))
        return sims

    def decode(self, ssp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full decode: recover [x, y, z] by nearest codebook match.

        Args:
            ssp: (..., dim)

        Returns:
            decoded_coords: (..., 3)
            best_similarity: (...)
        """
        sims = self.similarity_to_codebook(ssp)  # (..., 1000)

        flat_sims = sims.reshape(-1, sims.shape[-1])
        best_idx = flat_sims.argmax(dim=-1)  # (B,)
        best_sim = flat_sims.gather(1, best_idx[:, None]).squeeze(1)

        decoded = self.coords[best_idx]  # (B, 3)

        decoded = decoded.reshape(sims.shape[:-1] + (3,))
        best_sim = best_sim.reshape(sims.shape[:-1])
        return decoded, best_sim

    def decode_z_given_xy(
        self,
        ssp: torch.Tensor,
        xy: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode only z, assuming x and y are known.

        Args:
            ssp: (..., dim)
            xy:  (..., 2), with x,y in [0,9]

        Returns:
            z_hat: (...,) tensor of decoded z values in [0,9]
            best_similarity: (...,)
        """
        ssp = ssp.to(self.device)
        xy = xy.to(self.device).float()

        if xy.shape[-1] != 2:
            raise ValueError(f"Expected xy shape (..., 2), got {tuple(xy.shape)}")

        x = xy[..., 0]
        y = xy[..., 1]
        valid = (x >= 0) & (x <= 9) & (y >= 0) & (y <= 9)
        if not torch.all(valid):
            bad = xy[~valid]
            raise ValueError(
                f"Found out-of-range x/y. Expected x,y in [0,9]. Examples: {bad[:5]}"
            )

        batch_shape = xy.shape[:-1]
        flat_ssp = ssp.reshape(-1, ssp.shape[-1])
        flat_xy = xy.reshape(-1, xy.shape[-1])

        B = flat_xy.shape[0]
        z_candidates = torch.arange(0, 10, device=self.device).float()  # (10,)

        # Build candidate coordinates:
        # For each batch item, pair known (x,y) with z in [0..9]
        # Result shape: (B, 10, 3)
        xy_expanded = repeat(flat_xy, "b c -> b z c", z=10)  # (B, 10, 2)
        z_expanded = repeat(z_candidates, "z -> b z 1", b=B)  # (B, 10, 1)
        candidate_coords = torch.cat([xy_expanded, z_expanded], dim=-1)  # (B, 10, 3)

        candidate_ssp = self.encode(candidate_coords)  # (B, 10, dim)
        flat_ssp = F.normalize(flat_ssp, dim=-1)

        # Similarity per candidate z:
        # (B, 1, dim) * (B, 10, dim) -> (B, 10)
        sims = (flat_ssp[:, None, :] * candidate_ssp).sum(dim=-1)

        best_idx = sims.argmax(dim=-1)  # (B,)
        best_sim = sims.gather(1, best_idx[:, None]).squeeze(1)
        z_hat = z_candidates[best_idx].long()

        z_hat = z_hat.reshape(batch_shape)
        best_sim = best_sim.reshape(batch_shape)
        return z_hat, best_sim

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
