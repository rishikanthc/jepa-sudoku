from __future__ import annotations

import csv
from pathlib import Path
from typing import TextIO

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT


class StepMetricCSVLogger(pl.Callback):
    def __init__(self, csv_path: str) -> None:
        super().__init__()
        self.csv_path = Path(csv_path)
        self._handle: TextIO | None = None
        self._writer: csv.writer | None = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._handle)
        self._writer.writerow(
            [
                "epoch",
                "batch_idx",
                "global_step",
                "train_loss",
                "train_cosine_similarity",
                "empty_cells",
            ]
        )
        self._handle.flush()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: object,
        batch_idx: int,
    ) -> None:
        if not trainer.is_global_zero or self._writer is None or self._handle is None:
            return
        if not isinstance(outputs, dict):
            return

        loss = outputs.get("train_loss")
        cosine = outputs.get("train_cosine_similarity")
        empty_cells = outputs.get("empty_cells")
        if loss is None or cosine is None or empty_cells is None:
            return

        self._writer.writerow(
            [
                int(pl_module.current_epoch),
                int(batch_idx),
                int(trainer.global_step),
                float(loss.detach().cpu().item()),
                float(cosine.detach().cpu().item()),
                float(empty_cells.detach().cpu().item()),
            ]
        )
        self._handle.flush()

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._writer = None
