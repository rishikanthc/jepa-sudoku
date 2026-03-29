from __future__ import annotations

from jepa_sudoku.data.sudoku_generator import SudokuBoardGenerator


def test_reproducible_full_board_across_instances() -> None:
    seed = 123
    first = SudokuBoardGenerator(seed=seed)
    second = SudokuBoardGenerator(seed=seed)

    board1 = first.generate_full_board()
    board2 = second.generate_full_board()

    assert board1 == board2


def test_iterative_generation_consumes_rng_state() -> None:
    seed = 456
    generator = SudokuBoardGenerator(seed=seed)

    board1 = generator.generate_full_board()
    board2 = generator.generate_full_board()

    assert board1 != board2


def test_reset_replays_reproducible_sequence() -> None:
    seed = 789
    generator = SudokuBoardGenerator(seed=seed)

    first_series = [generator.generate_full_board() for _ in range(3)]

    generator.reset()
    second_series = [generator.generate_full_board() for _ in range(3)]

    assert first_series == second_series
