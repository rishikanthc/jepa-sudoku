from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple


Board = List[List[int]]

FULL_MASK = (1 << 9) - 1  # bits 0..8 -> digits 1..9


@dataclass
class SudokuPuzzle:
    """Container for a generated Sudoku puzzle and its solution."""

    puzzle: Board
    solution: Board
    removed_cells: int
    seed: Optional[int] = None


class SudokuBoardGenerator:
    """Generate complete Sudoku boards and playable puzzles with randomized backtracking."""

    SIZE = 9
    BOX_SIZE = 3

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        self._base_seed = seed
        self._rng = random.Random(seed)

    def reseed(self, seed: Optional[int]) -> None:
        """Reinitialize the generator with an explicit seed."""
        self.seed = seed
        self._base_seed = seed
        self._rng.seed(seed)

    def reset(self) -> None:
        """Reset RNG to the generator's initial seed for reproducible replay."""
        self._rng.seed(self._base_seed)

    def generate_full_board(self) -> Board:
        """Generate a full 9x9 valid Sudoku board using randomized MRV backtracking."""
        board: Board = [[0 for _ in range(self.SIZE)] for _ in range(self.SIZE)]
        row_masks, col_masks, box_masks = self._empty_masks()
        self._fill_board(board, row_masks, col_masks, box_masks)
        return board

    def generate_puzzle(
        self,
        removed_cells: int = 40,
        *,
        unique_solution: bool = True,
    ) -> SudokuPuzzle:
        """
        Generate a Sudoku puzzle with a target number of removed cells.

        Args:
            removed_cells: Target number of clues to clear (0..80).
            unique_solution: If True, keep a removal only if puzzle remains uniquely
                solvable.
        """
        if not 0 <= removed_cells <= 80:
            raise ValueError("removed_cells must be between 0 and 80 inclusive")

        solution = self.generate_full_board()
        puzzle = [row[:] for row in solution]
        target = min(removed_cells, 80)

        indices = list(range(self.SIZE * self.SIZE))
        self._rng.shuffle(indices)

        removed = 0
        for flat_idx in indices:
            if removed >= target:
                break

            r, c = divmod(flat_idx, self.SIZE)
            if puzzle[r][c] == 0:
                continue

            value = puzzle[r][c]
            puzzle[r][c] = 0

            if unique_solution:
                solutions = self._count_solutions_for_puzzle(puzzle, limit=2)
                if solutions != 1:
                    puzzle[r][c] = value
                    continue

            removed += 1

        return SudokuPuzzle(
            puzzle=puzzle,
            solution=solution,
            removed_cells=removed,
            seed=self.seed,
        )

    @staticmethod
    def is_complete(board: Board) -> bool:
        """Return True if board has no zeros."""
        return all(cell != 0 for row in board for cell in row)

    def _fill_board(
        self,
        board: Board,
        row_masks: List[int],
        col_masks: List[int],
        box_masks: List[int],
    ) -> bool:
        """Generate a full board by recursive randomized MRV backtracking."""
        state = self._select_mrv_cell(board, row_masks, col_masks, box_masks)
        if state is None:
            return True

        r, c, candidate_mask = state
        candidates = self._mask_to_digits(candidate_mask)
        self._rng.shuffle(candidates)

        for value in candidates:
            bit = 1 << (value - 1)
            self._place_value(board, row_masks, col_masks, box_masks, r, c, value, bit)
            if self._fill_board(board, row_masks, col_masks, box_masks):
                return True
            self._clear_value(board, row_masks, col_masks, box_masks, r, c, bit)

        return False

    def _count_solutions(
        self,
        board: Board,
        row_masks: List[int],
        col_masks: List[int],
        box_masks: List[int],
        *,
        limit: int = 2,
    ) -> int:
        """
        Count solutions up to ``limit`` using DFS.
        Stops search early when ``limit`` is reached.
        """
        count = 0

        def backtrack() -> None:
            nonlocal count
            if count >= limit:
                return

            state = self._select_mrv_cell(board, row_masks, col_masks, box_masks)
            if state is None:
                count += 1
                return

            r, c, candidate_mask = state
            candidates = self._mask_to_digits(candidate_mask)
            for value in candidates:
                bit = 1 << (value - 1)
                self._place_value(board, row_masks, col_masks, box_masks, r, c, value, bit)
                backtrack()
                self._clear_value(board, row_masks, col_masks, box_masks, r, c, bit)
                if count >= limit:
                    break

        backtrack()
        return count

    def _select_mrv_cell(
        self,
        board: Board,
        row_masks: List[int],
        col_masks: List[int],
        box_masks: List[int],
    ) -> Optional[Tuple[int, int, int]]:
        """
        Return (row, col, candidates) for the empty cell with the fewest options.
        MRV heuristic reduces backtracking.
        """
        best_position: Optional[Tuple[int, int, int]] = None
        best_count = 10

        for r in range(self.SIZE):
            for c in range(self.SIZE):
                if board[r][c] != 0:
                    continue

                candidates_mask = self._candidate_mask(row_masks, col_masks, box_masks, r, c)
                if candidates_mask == 0:
                    return (r, c, 0)

                candidate_count = candidates_mask.bit_count()
                if candidate_count < best_count:
                    best_count = candidate_count
                    best_position = (r, c, candidates_mask)
                    if best_count == 1:
                        return best_position

        return best_position

    def _candidate_mask(
        self,
        row_masks: List[int],
        col_masks: List[int],
        box_masks: List[int],
        r: int,
        c: int,
    ) -> int:
        """Return a 9-bit mask of allowed digits for the cell."""
        if r < 0 or r >= self.SIZE or c < 0 or c >= self.SIZE:
            raise IndexError("cell index out of bounds")
        if row_masks[r] | col_masks[c] | box_masks[self._box_index(r, c)] == FULL_MASK:
            return 0
        used = row_masks[r] | col_masks[c] | box_masks[self._box_index(r, c)]
        return FULL_MASK & ~used

    @staticmethod
    def _mask_to_digits(mask: int) -> List[int]:
        values: List[int] = []
        while mask:
            bit = mask & -mask
            values.append((bit.bit_length() - 1) + 1)
            mask ^= bit
        return values

    @staticmethod
    def _box_index(r: int, c: int) -> int:
        return (r // 3) * 3 + (c // 3)

    @staticmethod
    def _empty_masks() -> Tuple[List[int], List[int], List[int]]:
        return [0] * 9, [0] * 9, [0] * 9

    def _place_value(
        self,
        board: Board,
        row_masks: List[int],
        col_masks: List[int],
        box_masks: List[int],
        r: int,
        c: int,
        value: int,
        bit: int,
    ) -> None:
        board[r][c] = value
        row_masks[r] |= bit
        col_masks[c] |= bit
        box_masks[self._box_index(r, c)] |= bit

    def _clear_value(
        self,
        board: Board,
        row_masks: List[int],
        col_masks: List[int],
        box_masks: List[int],
        r: int,
        c: int,
        bit: int,
    ) -> None:
        if board[r][c] != 0:
            board[r][c] = 0
        if bit:
            row_masks[r] ^= bit
            col_masks[c] ^= bit
            box_masks[self._box_index(r, c)] ^= bit

    def solve(self, board: Board) -> Optional[Board]:
        """
        Solve a Sudoku board if a solution exists.
        Returns a solved board or None.
        """
        working = [row[:] for row in board]
        row_masks, col_masks, box_masks = self._masks_from_board(working)
        if self._solve_in_place(working, row_masks, col_masks, box_masks):
            return working
        return None

    def _solve_in_place(
        self,
        board: Board,
        row_masks: List[int],
        col_masks: List[int],
        box_masks: List[int],
    ) -> bool:
        state = self._select_mrv_cell(board, row_masks, col_masks, box_masks)
        if state is None:
            return True

        r, c, candidate_mask = state
        candidates = self._mask_to_digits(candidate_mask)
        for value in candidates:
            bit = 1 << (value - 1)
            self._place_value(board, row_masks, col_masks, box_masks, r, c, value, bit)
            if self._solve_in_place(board, row_masks, col_masks, box_masks):
                return True
            self._clear_value(board, row_masks, col_masks, box_masks, r, c, bit)

        return False

    def _count_solutions_for_puzzle(self, puzzle: Board, *, limit: int = 2) -> int:
        """Count solutions for a partially-filled puzzle."""
        row_masks, col_masks, box_masks = self._masks_from_board(puzzle)
        return self._count_solutions(
            puzzle,
            row_masks,
            col_masks,
            box_masks,
            limit=limit,
        )

    def _masks_from_board(self, board: Board) -> Tuple[List[int], List[int], List[int]]:
        """
        Build row/column/box masks from an existing board.
        Raises ValueError if board has duplicates.
        """
        row_masks, col_masks, box_masks = self._empty_masks()
        for r in range(self.SIZE):
            for c in range(self.SIZE):
                value = board[r][c]
                if not 0 <= value <= 9:
                    raise ValueError("Invalid board: cell values must be in [0, 9]")
                if value == 0:
                    continue
                bit = 1 << (value - 1)
                b = self._box_index(r, c)
                row_mask = row_masks[r]
                col_mask = col_masks[c]
                box_mask = box_masks[b]
                if (row_mask & bit) or (col_mask & bit) or (box_mask & bit):
                    raise ValueError("Invalid board: conflicting fixed digits")
                row_masks[r] |= bit
                col_masks[c] |= bit
                box_masks[b] |= bit
        return row_masks, col_masks, box_masks
