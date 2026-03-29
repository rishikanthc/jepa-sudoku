from __future__ import annotations

import torch

from jepa_sudoku.data.datamodule import (
    LinearMaskCurriculum,
    SudokuDataConfig,
    SudokuDataModule,
    SudokuPuzzleDataset,
)


def test_dataset_shapes_and_mask() -> None:
    dataset = SudokuPuzzleDataset(num_samples=5, num_cells_to_mask=24, seed=123)
    puzzle, solution, query = dataset[0]
    empty_count = query.shape[0]
    mask = puzzle[:, 2] != 0

    assert puzzle.shape == (81 - empty_count, 3)
    assert solution.shape == (empty_count, 3)
    assert query.shape == (empty_count, 3)

    # x and y are 1..9, z is 1..9 for clues and 0 for queries.
    assert torch.all((puzzle[:, 0] >= 1) & (puzzle[:, 0] <= 9))
    assert torch.all((puzzle[:, 1] >= 1) & (puzzle[:, 1] <= 9))
    assert torch.all((puzzle[:, 2] >= 1) & (puzzle[:, 2] <= 9))
    assert torch.all((solution[:, 2] >= 1) & (solution[:, 2] <= 9))
    assert torch.all(query[:, 2] == 0)

    assert mask.dtype == torch.bool
    assert mask.all()


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

    batch_puzzle, batch_solution, batch_query = next(iter(loader))
    n = module.current_num_cells_to_mask

    assert batch_puzzle.shape == (config.batch_size, 81 - n, 3)
    assert batch_solution.shape == (config.batch_size, n, 3)
    assert batch_query.shape == (config.batch_size, n, 3)


def test_reproducibility_across_dataset_instances() -> None:
    dataset_a = SudokuPuzzleDataset(num_samples=3, num_cells_to_mask=20, seed=555)
    dataset_b = SudokuPuzzleDataset(num_samples=3, num_cells_to_mask=20, seed=555)

    assert torch.allclose(dataset_a[2][0], dataset_b[2][0])
    assert torch.equal(dataset_a[2][2], dataset_b[2][2])


def test_dataset_can_load_precomputed_solution_file(tmp_path) -> None:
    solution_boards = torch.randint(1, 10, (6, 81), dtype=torch.uint8)
    dataset_path = tmp_path / "pretrain.pt"
    torch.save({"solution_boards": solution_boards}, dataset_path)

    config = SudokuDataConfig(
        num_samples=6,
        num_cells_to_mask=3,
        dataset_path=str(dataset_path),
        batch_size=2,
        shuffle=False,
    )
    module = SudokuDataModule(config)

    puzzle, solution, query = module.get_single(0)
    assert puzzle.shape == (78, 3)
    assert solution.shape == (3, 3)
    assert query.shape == (3, 3)


def test_curriculum_masking_is_step_driven_and_increasing() -> None:
    curriculum = LinearMaskCurriculum(start=1, max_mask=5, num_steps=4)
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
    puzzle_easy, _, query_easy = module.get_single(0)
    module.set_epoch(4)
    puzzle_hard, _, query_hard = module.get_single(0)

    assert puzzle_easy.shape[0] == 80  # one cell masked
    assert puzzle_hard.shape[0] == 76  # five cells masked
    assert not torch.equal(puzzle_easy, puzzle_hard)
    assert query_easy.shape[0] == 1
    assert query_hard.shape[0] == 5


def test_curriculum_reproducibility_across_instances() -> None:
    curriculum = LinearMaskCurriculum(start=2, max_mask=6, num_steps=3)
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

    puzzle_a, solution_a, query_a = module_a.get_single(1)
    puzzle_b, solution_b, query_b = module_b.get_single(1)

    assert torch.equal(puzzle_a, puzzle_b)
    assert torch.equal(solution_a, solution_b)
    assert torch.equal(query_a, query_b)


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
    puzzle_epoch0, _, query_epoch0 = module.get_single(0)
    module.set_epoch(3)
    puzzle_epoch3, _, query_epoch3 = module.get_single(0)

    assert puzzle_epoch0.shape == (63, 3)
    assert query_epoch0.shape[0] == 18
    assert query_epoch3.shape[0] == 18
    assert not torch.equal(puzzle_epoch0, puzzle_epoch3)


def test_randomized_masking_changes_per_access_for_same_sample() -> None:
    dataset = SudokuPuzzleDataset(num_samples=3, num_cells_to_mask=10, seed=7)

    first, _, first_query = dataset[0]
    second, _, second_query = dataset[0]

    assert first_query.shape[0] == 10
    assert second_query.shape[0] == 10
    assert not torch.equal(first, second)


def test_set_num_cells_to_mask_updates_output_shapes() -> None:
    module = SudokuDataModule(
        SudokuDataConfig(
            num_samples=4,
            num_cells_to_mask=2,
            seed=555,
            batch_size=2,
            shuffle=False,
        )
    )

    puzzle_a, solution_a, query_a = module.get_single(0)
    module.set_num_cells_to_mask(5)
    puzzle_b, solution_b, query_b = module.get_single(0)

    assert puzzle_a.shape == (79, 3)
    assert puzzle_b.shape == (76, 3)
    assert query_b.shape[0] == 5
    assert solution_b.shape[0] == 5
    assert query_a.shape[0] != query_b.shape[0]
