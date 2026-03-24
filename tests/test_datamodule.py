from __future__ import annotations

import torch

from datamodule import (
    LinearMaskCurriculum,
    SudokuDataConfig,
    SudokuDataModule,
    SudokuPuzzleDataset,
)


def test_dataset_shapes_and_mask() -> None:
    dataset = SudokuPuzzleDataset(num_samples=5, num_cells_to_mask=24, seed=123)
    puzzle, solution, query, mask = dataset[0]
    empty_count = int((puzzle[:, 2] == 0).sum().item())

    assert puzzle.shape == (81, 3)
    assert solution.shape == (empty_count, 3)
    assert mask.shape == (81,)
    assert query.shape == (empty_count, 3)

    # x and y are 1..9, z is 0..9
    assert torch.all((puzzle[:, 0] >= 1) & (puzzle[:, 0] <= 9))
    assert torch.all((puzzle[:, 1] >= 1) & (puzzle[:, 1] <= 9))
    assert torch.all((puzzle[:, 2] >= 0) & (puzzle[:, 2] <= 9))
    assert torch.all((solution[:, 2] >= 1) & (solution[:, 2] <= 9))
    assert torch.all(query[:, 2] == 0)

    # mask tracks puzzle clues
    assert mask.dtype == torch.bool
    assert torch.equal(mask, puzzle[:, 2] != 0)

    # query should contain exactly the empty cell coordinates
    empty_coords = puzzle[~mask][:, :2]
    assert torch.equal(query[:, :2], empty_coords)


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

    batch_puzzle, batch_solution, batch_query, batch_mask = next(iter(loader))
    n = module.current_num_cells_to_mask

    assert batch_puzzle.shape == (config.batch_size, 81, 3)
    assert batch_solution.shape == (config.batch_size, n, 3)
    assert batch_query.shape == (config.batch_size, n, 3)
    assert batch_mask.shape == (config.batch_size, 81)
    assert batch_mask.dtype == torch.bool


def test_reproducibility_across_dataset_instances() -> None:
    dataset_a = SudokuPuzzleDataset(num_samples=3, num_cells_to_mask=20, seed=555)
    dataset_b = SudokuPuzzleDataset(num_samples=3, num_cells_to_mask=20, seed=555)

    assert torch.allclose(dataset_a[2][0], dataset_b[2][0])
    assert torch.equal(dataset_a[2][2], dataset_b[2][2])


def test_curriculum_masking_is_epoch_driven_and_increasing() -> None:
    curriculum = LinearMaskCurriculum(start=1, max_mask=5, num_epochs=4)
    config = SudokuDataConfig(
        num_samples=2,
        num_cells_to_mask=5,
        seed=99,
        mask_cells_curriculum=curriculum,
        batch_size=1,
        shuffle=False,
    )
    module = SudokuDataModule(config)

    module.set_epoch(0)
    puzzle_easy, _, query_easy, mask_easy = module.get_single(0)
    module.set_epoch(4)
    puzzle_hard, _, query_hard, mask_hard = module.get_single(0)

    assert (mask_easy.sum() == 80)  # one cell masked
    assert (mask_hard.sum() == 76)  # five cells masked
    assert not torch.equal(puzzle_easy, puzzle_hard)
    assert query_easy.shape[0] == 1
    assert query_hard.shape[0] == 5


def test_curriculum_reproducibility_across_instances() -> None:
    curriculum = LinearMaskCurriculum(start=2, max_mask=6, num_epochs=3)
    config = SudokuDataConfig(
        num_samples=3,
        num_cells_to_mask=6,
        seed=17,
        mask_cells_curriculum=curriculum,
        batch_size=1,
        shuffle=False,
    )

    module_a = SudokuDataModule(config)
    module_b = SudokuDataModule(config)
    module_a.set_epoch(2)
    module_b.set_epoch(2)

    puzzle_a, solution_a, query_a, mask_a = module_a.get_single(1)
    puzzle_b, solution_b, query_b, mask_b = module_b.get_single(1)

    assert torch.equal(puzzle_a, puzzle_b)
    assert torch.equal(solution_a, solution_b)
    assert torch.equal(query_a, query_b)
    assert torch.equal(mask_a, mask_b)


def test_non_unique_masking_position_changes_with_epoch() -> None:
    config = SudokuDataConfig(
        num_samples=4,
        num_cells_to_mask=18,
        seed=123,
        batch_size=1,
        shuffle=False,
    )
    module = SudokuDataModule(config)

    module.set_epoch(0)
    puzzle_epoch0, _, query_epoch0, _ = module.get_single(0)
    module.set_epoch(3)
    puzzle_epoch3, _, query_epoch3, _ = module.get_single(0)

    assert puzzle_epoch0.shape == (81, 3)
    assert query_epoch0.shape[0] == 18
    assert query_epoch3.shape[0] == 18
    assert not torch.equal(puzzle_epoch0, puzzle_epoch3)


def test_randomized_masking_changes_per_access_for_same_sample() -> None:
    dataset = SudokuPuzzleDataset(num_samples=3, num_cells_to_mask=10, seed=7)

    first, _, first_query, _ = dataset[0]
    second, _, second_query, _ = dataset[0]

    assert first_query.shape[0] == 10
    assert second_query.shape[0] == 10
    assert not torch.equal(first, second)
