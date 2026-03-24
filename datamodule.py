from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Callable

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from sudoku_generator import SudokuBoardGenerator


MaskSchedule = Callable[[int], int]


@dataclass(frozen=True)
class LinearMaskCurriculum:
    """
    Linear schedule for masking difficulty.

    start: number of cells to mask at epoch 0.
    max_mask: maximum number of masked cells after `num_epochs`.
    num_epochs: total epochs over which to linearly increase from start to max.
    """

    start: int
    max_mask: int
    num_epochs: int = 1

    def __post_init__(self) -> None:
        if self.num_epochs < 0:
            raise ValueError("num_epochs must be >= 0")
        if not 0 <= self.start <= 64:
            raise ValueError("start must be between 0 and 64")
        if not 0 <= self.max_mask <= 64:
            raise ValueError("max_mask must be between 0 and 64")
        if self.max_mask < self.start:
            raise ValueError("max_mask must be >= start")

    def __call__(self, epoch: int) -> int:
        if self.num_epochs == 0:
            return self.max_mask

        clamped_epoch = max(0, epoch)
        if clamped_epoch >= self.num_epochs:
            return self.max_mask

        ratio = clamped_epoch / self.num_epochs
        return int(round(self.start + (self.max_mask - self.start) * ratio))


@dataclass(frozen=True)
class SudokuDataConfig:
    num_samples: int
    num_cells_to_mask: int
    seed: int = 0
    unique_solution: bool = False
    mask_cells_curriculum: MaskSchedule | None = None
    batch_size: int = 32
    num_workers: int = 0
    shuffle: bool = True
    pin_memory: bool = False
    drop_last: bool = False


def _board_to_value_tensor(board: list[list[int]]) -> torch.Tensor:
    values = [cell for row in board for cell in row]
    return torch.tensor(values, dtype=torch.float32)


class SudokuPuzzleDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    """
    Dataset yielding:
      - puzzle:  (81, 3) coordinates with zeros for empty cells in z
      - solution: (N, 3) solved coordinates at empty cell positions
      - query:    (N, 3) query coordinates for each empty cell (z=0)
      - mask:     (81,) boolean mask, True where puzzle has a clue
    """

    def __init__(
        self,
        num_samples: int,
        num_cells_to_mask: int,
        seed: int = 0,
        unique_solution: bool = False,
        mask_cells_curriculum: MaskSchedule | None = None,
    ) -> None:
        if num_samples < 0:
            raise ValueError("num_samples must be >= 0")
        if not 0 <= num_cells_to_mask <= 64:
            raise ValueError("num_cells_to_mask must be between 0 and 64 inclusive")

        self.num_samples = num_samples
        self.num_cells_to_mask = num_cells_to_mask
        self.seed = int(seed)
        self.unique_solution = unique_solution
        self.mask_cells_curriculum = mask_cells_curriculum
        self.current_epoch = 0

        # Optional cache keyed by index for efficient curriculum masking.
        # Stores full solution + a deterministic removal order for each sample.
        self._solution_cache: dict[int, torch.Tensor] = {}
        self._removal_cache: dict[int, torch.Tensor] = {}

        # Base x,y coordinates are fixed, 1..9 in row-major order.
        xs = torch.arange(1, 10)
        ys = torch.arange(1, 10)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
        self._xy = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1).float()

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def _num_cells_for_epoch(self) -> int:
        if self.mask_cells_curriculum is None:
            num_cells = self.num_cells_to_mask
        else:
            num_cells = int(self.mask_cells_curriculum(self.current_epoch))

        return max(0, min(64, num_cells))

    def _get_or_build_template(self, index: int) -> tuple[Tensor, Tensor]:
        if index in self._solution_cache and index in self._removal_cache:
            return self._solution_cache[index], self._removal_cache[index]

        sample_seed = self.seed + index
        generator = SudokuBoardGenerator(seed=sample_seed)
        solution_board = generator.generate_full_board()
        solution_values = _board_to_value_tensor(solution_board)

        rng = Random(sample_seed)
        removal_order = list(range(81))
        rng.shuffle(removal_order)
        removal_order_tensor = torch.tensor(removal_order, dtype=torch.long)

        self._solution_cache[index] = solution_values
        self._removal_cache[index] = removal_order_tensor
        return solution_values, removal_order_tensor

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if not 0 <= index < self.num_samples:
            raise IndexError(
                f"Index {index} out of range for dataset of size {self.num_samples}"
            )

        num_cells_to_mask = self._num_cells_for_epoch()

        if self.unique_solution:
            if self.mask_cells_curriculum is not None:
                sample_seed = self.seed + index
                puzzle_data = SudokuBoardGenerator(seed=sample_seed).generate_puzzle(
                    removed_cells=num_cells_to_mask,
                    unique_solution=True,
                )
                solution_values = _board_to_value_tensor(puzzle_data.solution)
                puzzle_values = _board_to_value_tensor(puzzle_data.puzzle)
            else:
                sample_seed = self.seed + index
                puzzle_data = SudokuBoardGenerator(seed=sample_seed).generate_puzzle(
                    removed_cells=self.num_cells_to_mask,
                    unique_solution=True,
                )
                solution_values = _board_to_value_tensor(puzzle_data.solution)
                puzzle_values = _board_to_value_tensor(puzzle_data.puzzle)
            if int((puzzle_values == 0).sum()) != num_cells_to_mask:
                solution_values, removal_order = self._get_or_build_template(index)
                puzzle_values = solution_values.clone()
                n = min(num_cells_to_mask, 64)
                if n > 0:
                    puzzle_values[removal_order[:n]] = 0.0
        else:
            solution_values, removal_order = self._get_or_build_template(index)
            puzzle_values = solution_values.clone()
            n = min(num_cells_to_mask, 64)
            if n > 0:
                puzzle_values[removal_order[:n]] = 0.0

        solution = torch.empty((81, 3), dtype=torch.float32)
        solution[:, :2] = self._xy
        solution[:, 2] = solution_values

        puzzle = torch.empty((81, 3), dtype=torch.float32)
        puzzle[:, :2] = self._xy
        puzzle[:, 2] = puzzle_values

        mask = puzzle_values.ne(0.0)
        empty_indices = torch.nonzero(~mask, as_tuple=False).reshape(-1)
        solution_cells = solution[empty_indices]
        query = torch.empty((empty_indices.numel(), 3), dtype=torch.float32)
        query[:, :2] = self._xy[empty_indices]
        query[:, 2] = 0.0
        return puzzle, solution_cells, query, mask


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
            mask_cells_curriculum=config.mask_cells_curriculum,
        )
        self._dataloader_generator = torch.Generator().manual_seed(config.seed)

    def set_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(epoch)

    @property
    def current_num_cells_to_mask(self) -> int:
        return self.dataset._num_cells_for_epoch()

    def train_dataloader(self) -> DataLoader[tuple[Tensor, Tensor, Tensor, Tensor]]:
        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last,
            generator=self._dataloader_generator,
        )

    def get_single(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.dataset[idx]

    def __len__(self) -> int:
        return len(self.dataset)
