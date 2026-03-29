from __future__ import annotations

import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jepa_sudoku.data.sudoku_generator import SudokuBoardGenerator
from jepa_sudoku.model.ssp import ThreeAxisSSP, ThreeAxisSSPConfig


def _is_complete_valid_sudoku(board: list[list[int]]) -> bool:
    expected = set(range(1, 10))

    for row in board:
        if set(row) != expected:
            return False

    for c in range(9):
        col = {board[r][c] for r in range(9)}
        if col != expected:
            return False

    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            box = set()
            for r in range(br, br + 3):
                for c in range(bc, bc + 3):
                    box.add(board[r][c])
            if box != expected:
                return False

    return True


def _board_to_coords(board: list[list[int]]) -> torch.Tensor:
    data = []
    for r, row in enumerate(board, start=1):
        for c, value in enumerate(row, start=1):
            data.append([r, c, value])
    return torch.tensor(data, dtype=torch.float32)


def test_generate_five_complete_sudoku_boards_and_validate_rules() -> None:
    for seed in range(5):
        generator = SudokuBoardGenerator(seed=seed)
        board = generator.generate_full_board()
        assert _is_complete_valid_sudoku(board)


def test_encode_five_sudoku_boards_as_ssp_cells() -> None:
    ssp = ThreeAxisSSP(ThreeAxisSSPConfig(dim=256, seed=99)).to("cpu")

    for seed in range(5):
        generator = SudokuBoardGenerator(seed=100 + seed)
        board = generator.generate_puzzle(removed_cells=30, unique_solution=False).puzzle

        # include at least one empty cell to exercise z=0
        board[0][0] = 0

        coords = _board_to_coords(board)  # shape: (81, 3), x/y in [1, 9], z in [0, 9]
        encoded = ssp.encode(coords)
        decoded, _ = ssp.decode(encoded)

        assert decoded.shape == coords.shape
        assert torch.equal(decoded, coords)
