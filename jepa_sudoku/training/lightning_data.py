from __future__ import annotations

import lightning.pytorch as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from jepa_sudoku.data.datamodule import (
    SudokuDataConfig,
    SudokuPuzzleDataset,
    load_precomputed_solutions,
)


class LightningSudokuDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        train_config: SudokuDataConfig,
        adaptive_curriculum_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.train_config = train_config
        self.adaptive_curriculum_enabled = adaptive_curriculum_enabled
        self.train_dataset: SudokuPuzzleDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is None:
            self.train_dataset = SudokuPuzzleDataset(
                num_samples=self.train_config.num_samples,
                num_cells_to_mask=self.train_config.num_cells_to_mask,
                solution_boards=(
                    load_precomputed_solutions(self.train_config.dataset_path)
                    if self.train_config.dataset_path is not None
                    else None
                ),
                seed=self.train_config.seed,
                unique_solution=self.train_config.unique_solution,
                randomize_mask_per_access=self.train_config.randomize_mask_per_access,
                mask_cells_curriculum=self.train_config.mask_cells_curriculum,
            )

        if self.adaptive_curriculum_enabled:
            self.set_num_cells_to_mask(self.train_config.num_cells_to_mask)

    def train_dataloader(self) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before requesting the training dataloader.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_config.batch_size,
            shuffle=self.train_config.shuffle,
            num_workers=self.train_config.num_workers,
            pin_memory=self.train_config.pin_memory,
            drop_last=self.train_config.drop_last,
            persistent_workers=self.train_config.num_workers > 0,
            generator=torch.Generator().manual_seed(self.train_config.seed),
        )

    def set_epoch(self, epoch: int) -> None:
        if self.train_dataset is not None:
            self.train_dataset.set_epoch(epoch)

    def set_num_cells_to_mask(self, num_cells_to_mask: int) -> None:
        if self.train_dataset is not None:
            self.train_dataset.set_num_cells_to_mask(num_cells_to_mask)

    @property
    def current_num_cells_to_mask(self) -> int:
        if self.train_dataset is None:
            return self.train_config.num_cells_to_mask
        return self.train_dataset._num_cells_for_epoch()
