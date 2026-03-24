from __future__ import annotations

import torch

from datamodule import SudokuDataConfig, SudokuDataModule, SudokuPuzzleDataset


def test_dataset_shapes_and_mask() -> None:
    dataset = SudokuPuzzleDataset(num_samples=5, num_cells_to_mask=24, seed=123)
    puzzle, solution, mask = dataset[0]

    assert puzzle.shape == (81, 3)
    assert solution.shape == (81, 3)
    assert mask.shape == (81,)

    # x and y are 1..9, z is 0..9
    assert torch.all((puzzle[:, 0] >= 1) & (puzzle[:, 0] <= 9))
    assert torch.all((puzzle[:, 1] >= 1) & (puzzle[:, 1] <= 9))
    assert torch.all((puzzle[:, 2] >= 0) & (puzzle[:, 2] <= 9))
    assert torch.all((solution[:, 2] >= 1) & (solution[:, 2] <= 9))

    # mask tracks puzzle clues
    assert mask.dtype == torch.bool
    assert torch.equal(mask, puzzle[:, 2] != 0)


def test_dataloader_shapes_match_batching() -> None:
    config = SudokuDataConfig(
        num_samples=16,
        num_cells_to_mask=40,
        seed=7,
        batch_size=4,
        num_workers=0,
        shuffle=False,
    )
    module = SudokuDataModule(config)
    loader = module.train_dataloader()

    batch_puzzle, batch_solution, batch_mask = next(iter(loader))
    assert batch_puzzle.shape == (config.batch_size, 81, 3)
    assert batch_solution.shape == (config.batch_size, 81, 3)
    assert batch_mask.shape == (config.batch_size, 81)
    assert batch_mask.dtype == torch.bool


def test_reproducibility_across_dataset_instances() -> None:
    dataset_a = SudokuPuzzleDataset(num_samples=3, num_cells_to_mask=20, seed=555)
    dataset_b = SudokuPuzzleDataset(num_samples=3, num_cells_to_mask=20, seed=555)

    assert torch.allclose(dataset_a[2][0], dataset_b[2][0])
    assert torch.equal(dataset_a[2][2], dataset_b[2][2])
