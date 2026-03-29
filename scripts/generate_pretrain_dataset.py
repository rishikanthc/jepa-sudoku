from __future__ import annotations

import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time

import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jepa_sudoku.data.sudoku_generator import SudokuBoardGenerator


def _generate_solution_values_from_seed(seed: int) -> list[int]:
    generator = SudokuBoardGenerator(seed=seed)
    solution_board = generator.generate_full_board()
    return [int(cell) for row in solution_board for cell in row]


def generate_dataset(
    *,
    output_path: Path,
    num_samples: int,
    seed: int,
    workers: int,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Pass overwrite=true in the dataset config to replace it."
        )
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    worker_count = max(1, workers)
    seeds = [(seed * 1_000_003) + index for index in range(num_samples)]
    boards = torch.empty((num_samples, 81), dtype=torch.uint8)

    print(
        f"Generating Sudoku dataset -> {output_path} "
        f"(num_samples={num_samples}, workers={worker_count})"
    )
    start_time = time.perf_counter()
    progress = tqdm(total=num_samples, desc="Generating", unit="board")
    if worker_count == 1:
        for index, sample_seed in enumerate(seeds):
            boards[index] = torch.tensor(
                _generate_solution_values_from_seed(sample_seed),
                dtype=torch.uint8,
            )
            progress.update(1)
    else:
        # Keep chunks small enough that tqdm updates quickly, but large enough to
        # avoid excessive scheduling overhead on multi-million board runs.
        chunk_size = min(2048, max(64, num_samples // (worker_count * 256)))
        context_name = "fork" if sys.platform != "win32" else "spawn"
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp.get_context(context_name),
        ) as executor:
            for index, values in enumerate(
                executor.map(_generate_solution_values_from_seed, seeds, chunksize=chunk_size)
            ):
                boards[index] = torch.tensor(values, dtype=torch.uint8)
                progress.update(1)
    progress.close()
    payload = {
        "format": "jepa_sudoku_pretrain_dataset",
        "version": 1,
        "num_samples": num_samples,
        "seed": seed,
        "solution_boards": boards,
    }
    torch.save(payload, output_path)
    elapsed = time.perf_counter() - start_time
    print(f"Saved dataset to {output_path} in {elapsed:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and save a Sudoku pre-training dataset.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset/pretrain_default.yaml"),
        help="Path to the dataset generation config file.",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    generate_dataset(
        output_path=Path(config.output_path),
        num_samples=int(config.num_samples),
        seed=int(config.seed),
        workers=int(config.workers),
        overwrite=bool(config.overwrite),
    )


if __name__ == "__main__":
    main()
