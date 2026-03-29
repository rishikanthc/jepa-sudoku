from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return values.copy()
    padded = np.pad(values, (window - 1, 0), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _moving_std(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return np.zeros_like(values)
    mean = _moving_average(values, window)
    mean_sq = _moving_average(values**2, window)
    variance = np.clip(mean_sq - (mean**2), a_min=0.0, a_max=None)
    return np.sqrt(variance)


def load_metrics_csv(
    csv_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    steps: list[int] = []
    losses: list[float] = []
    cosine_similarities: list[float] = []
    empty_cells: list[float] = []
    has_empty_cells = False
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            steps.append(int(row["global_step"]))
            losses.append(float(row["train_loss"]))
            cosine_similarities.append(float(row["train_cosine_similarity"]))
            if "empty_cells" in row and row["empty_cells"] not in {None, ""}:
                has_empty_cells = True
                empty_cells.append(float(row["empty_cells"]))
    return (
        np.asarray(steps),
        np.asarray(losses),
        np.asarray(cosine_similarities),
        np.asarray(empty_cells) if has_empty_cells else None,
    )


def plot_metrics_curve(
    csv_path: Path,
    output_path: Path,
    *,
    window: int,
    title: str | None,
) -> None:
    steps, losses, cosine_similarities, empty_cells = load_metrics_csv(csv_path)
    if len(steps) == 0:
        raise ValueError(f"No rows found in {csv_path}")

    loss_smooth = _moving_average(losses, window)
    loss_spread = _moving_std(losses, window)
    cosine_smooth = _moving_average(cosine_similarities, window)
    empty_cells_smooth = (
        _moving_average(empty_cells, window) if empty_cells is not None else None
    )

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#2c2c2c",
            "axes.labelcolor": "#2c2c2c",
            "xtick.color": "#2c2c2c",
            "ytick.color": "#2c2c2c",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=180)
    ax2 = ax.twinx()

    loss_band = ax.fill_between(
        steps,
        loss_smooth - loss_spread,
        loss_smooth + loss_spread,
        color="#8b8b8b",
        alpha=0.18,
        linewidth=0,
    )
    loss_raw_line = ax.plot(
        steps,
        losses,
        color="#f26a9a",
        linewidth=1.2,
        alpha=0.35,
        label="Step loss",
    )[0]
    loss_line = ax.plot(
        steps,
        loss_smooth,
        color="#2e2e2e",
        linewidth=2.0,
        label="Smoothed loss",
    )[0]
    cosine_line = ax.plot(
        steps,
        cosine_smooth,
        color="#f26a9a",
        linewidth=1.8,
        alpha=0.85,
        label="Smoothed cosine similarity",
    )[0]
    empty_cells_line = None
    if empty_cells_smooth is not None:
        empty_cells_line = ax2.plot(
            steps,
            empty_cells_smooth,
            color="#6f6f6f",
            linewidth=1.6,
            linestyle="--",
            label="Empty cells",
        )[0]

    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss / cosine similarity")
    if empty_cells_line is not None:
        ax2.set_ylabel("# empty cells", color="#6f6f6f")
        ax2.tick_params(axis="y", colors="#6f6f6f")
    else:
        ax2.set_yticks([])
        ax2.set_ylabel("")

    if title:
        ax.set_title(title, fontsize=13, color="#2c2c2c", pad=14)

    ax.grid(False)
    ax2.grid(False)
    ax.margins(x=0.02)

    handles = [loss_line, loss_raw_line, cosine_line]
    if empty_cells_line is not None:
        handles.append(empty_cells_line)
    labels = [handle.get_label() for handle in handles]
    ax.legend(handles, labels, frameon=False, loc="upper right", handlelength=1.8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot pre-training metrics from CSV logs.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("logs/pretrain_metrics.csv"),
        help="Path to the CSV file written during training.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("logs/pretrain_metrics_curve.png"),
        help="Output path for the plot image.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=25,
        help="Smoothing window for the plotted curves.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Pre-Training Metrics",
        help="Optional plot title.",
    )
    args = parser.parse_args()
    plot_metrics_curve(args.csv, args.out, window=max(1, args.window), title=args.title)


if __name__ == "__main__":
    main()
