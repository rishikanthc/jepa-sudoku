from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from sudoku_generator import SudokuBoardGenerator


@dataclass(frozen=True)
class SudokuDataConfig:
    num_samples: int
    num_cells_to_mask: int
    seed: int = 0
    unique_solution: bool = False
    batch_size: int = 32
    num_workers: int = 0
    shuffle: bool = True
    pin_memory: bool = False
    drop_last: bool = False


def _board_to_value_tensor(board: list[list[int]]) -> torch.Tensor:
    values = [cell for row in board for cell in row]
    return torch.tensor(values, dtype=torch.float32)


class SudokuPuzzleDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """
    Dataset yielding:
      - puzzle:  (81, 3) coordinates with zeros for empty cells in z
      - solution: (81, 3) full solved coordinates
      - mask:     (81,) boolean mask, True where puzzle has a clue
    """

    def __init__(
        self,
        num_samples: int,
        num_cells_to_mask: int,
        seed: int = 0,
        unique_solution: bool = False,
    ) -> None:
        if num_samples < 0:
            raise ValueError("num_samples must be >= 0")
        if not 0 <= num_cells_to_mask <= 80:
            raise ValueError("num_cells_to_mask must be between 0 and 80 inclusive")

        self.num_samples = num_samples
        self.num_cells_to_mask = num_cells_to_mask
        self.seed = int(seed)
        self.unique_solution = unique_solution

        # Base x,y coordinates are fixed, 1..9 in row-major order.
        xs = torch.arange(1, 10)
        ys = torch.arange(1, 10)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
        self._xy = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1).float()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        if not 0 <= index < self.num_samples:
            raise IndexError(f"Index {index} out of range for dataset of size {self.num_samples}")

        sample_seed = self.seed + index
        generator = SudokuBoardGenerator(seed=sample_seed)
        puzzle_data = generator.generate_puzzle(
            removed_cells=self.num_cells_to_mask,
            unique_solution=self.unique_solution,
        )

        solution_values = _board_to_value_tensor(puzzle_data.solution)
        puzzle_values = _board_to_value_tensor(puzzle_data.puzzle)

        solution = torch.empty((81, 3), dtype=torch.float32)
        solution[:, :2] = self._xy
        solution[:, 2] = solution_values

        puzzle = torch.empty((81, 3), dtype=torch.float32)
        puzzle[:, :2] = self._xy
        puzzle[:, 2] = puzzle_values

        mask = puzzle_values.ne(0.0)
        return puzzle, solution, mask


class SudokuDataModule:
    """
    Lightweight datamodule wrapper with a torch DataLoader-compatible Dataset.
    """

    def __init__(self, config: SudokuDataConfig) -> None:
        self.config = config
        self.dataset = SudokuPuzzleDataset(
            num_samples=config.num_samples,
            num_cells_to_mask=config.num_cells_to_mask,
            seed=config.seed,
            unique_solution=config.unique_solution,
        )
        self._dataloader_generator = torch.Generator().manual_seed(config.seed)

    def train_dataloader(self) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last,
            generator=self._dataloader_generator,
        )

    def get_single(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.dataset[idx]

    def __len__(self) -> int:
        return len(self.dataset)
