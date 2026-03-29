from __future__ import annotations

from dataclasses import dataclass

import lightning.pytorch as pl
import torch
from torch import Tensor
from torch.optim import Adam

from jepa_sudoku.model.losses import cosine_contrastive_loss
from jepa_sudoku.model.models import Encoder, Predictor, SudokuRepresentation


@dataclass
class LightningTrainConfig:
    learning_rate: float = 1e-3
    non_target_weight: float = 1.0
    non_target_margin: float = 0.2
    curriculum_enabled: bool = False
    curriculum_mode: str = "adaptive"
    curriculum_step: int = 1
    curriculum_patience: int = 3
    curriculum_min_delta: float = 1.0e-6
    curriculum_max_mask: int = 81


class SudokuLightningModule(pl.LightningModule):
    def __init__(
        self,
        *,
        encoder: Encoder,
        predictor: Predictor,
        representation: SudokuRepresentation,
        config: LightningTrainConfig,
    ) -> None:
        super().__init__()
        if encoder.representation is not predictor.representation:
            raise ValueError("Encoder and predictor must share the same representation instance.")
        if encoder.representation is not representation:
            raise ValueError("Pass the same representation instance used by encoder/predictor.")

        self.encoder = encoder
        self.predictor = predictor
        self.representation = representation
        self.config = config
        self.history: list[float] = []

        self._curriculum_plateau_counter = 0
        self._curriculum_best_loss = float("inf")

    def configure_optimizers(self) -> Adam:
        return Adam(
            list(self.encoder.parameters()) + list(self.predictor.parameters()),
            lr=self.config.learning_rate,
        )

    def _shared_step(self, batch: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        puzzle, solution, query = batch
        encoder_out = self.encoder(puzzle)
        pred_vectors = self.predictor(query, encoder_out)
        target_vectors = self.representation.encode_targets(solution)
        loss = cosine_contrastive_loss(
            pred_vectors,
            target_vectors,
            solution[..., 2],
            self.representation.digit_prototypes(),
            non_target_weight=self.config.non_target_weight,
            margin=self.config.non_target_margin,
        )

        pred_norm = torch.nn.functional.normalize(pred_vectors, dim=-1)
        target_norm = torch.nn.functional.normalize(target_vectors, dim=-1)
        cosine_sim = (pred_norm * target_norm).sum(dim=-1).mean()
        return loss, cosine_sim

    def training_step(
        self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> dict[str, Tensor]:
        loss, cosine_sim = self._shared_step(batch)
        data_module = self.trainer.datamodule
        empty_cells = (
            float(data_module.current_num_cells_to_mask)
            if data_module is not None
            else float(batch[2].shape[1])
        )
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train_cosine_similarity",
            cosine_sim,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "empty_cells",
            empty_cells,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
            sync_dist=True,
            rank_zero_only=True,
        )
        return {
            "loss": loss,
            "train_loss": loss.detach(),
            "train_cosine_similarity": cosine_sim.detach(),
            "empty_cells": torch.tensor(empty_cells, device=loss.device),
        }

    def on_fit_start(self) -> None:
        data_module = self.trainer.datamodule
        if data_module is None:
            return
        if self.config.curriculum_enabled and self.config.curriculum_mode == "adaptive":
            data_module.set_num_cells_to_mask(data_module.train_config.num_cells_to_mask)

    def on_train_epoch_start(self) -> None:
        data_module = self.trainer.datamodule
        if data_module is None:
            return
        data_module.set_epoch(self.current_epoch)
        self.log(
            "curriculum_empty_cells",
            float(data_module.current_num_cells_to_mask),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=False,
            sync_dist=True,
            rank_zero_only=True,
        )

    def on_train_batch_start(
        self,
        batch: tuple[Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> None:
        data_module = self.trainer.datamodule
        if data_module is None:
            return
        if self.config.curriculum_enabled and self.config.curriculum_mode == "linear":
            data_module.set_epoch(self.global_step)

    def _metric_to_float(self, metric_name: str) -> float | None:
        value = self.trainer.callback_metrics.get(metric_name)
        if value is None:
            return None
        return float(value.detach().cpu().item())

    def _maybe_increase_difficulty(self, monitored_loss: float) -> None:
        if self.config.curriculum_mode != "adaptive" or not self.config.curriculum_enabled:
            return

        data_module = self.trainer.datamodule
        if data_module is None or data_module.current_num_cells_to_mask >= self.config.curriculum_max_mask:
            return

        if monitored_loss < (self._curriculum_best_loss - self.config.curriculum_min_delta):
            self._curriculum_best_loss = monitored_loss
            self._curriculum_plateau_counter = 0
        else:
            self._curriculum_plateau_counter += 1

        if self._curriculum_plateau_counter < self.config.curriculum_patience:
            return

        new_num_cells = min(
            self.config.curriculum_max_mask,
            data_module.current_num_cells_to_mask + self.config.curriculum_step,
        )
        if new_num_cells == data_module.current_num_cells_to_mask:
            return

        data_module.set_num_cells_to_mask(new_num_cells)
        self._curriculum_plateau_counter = 0
        self._curriculum_best_loss = float("inf")
        trainer = getattr(self, "_trainer", None)
        if trainer is None or getattr(trainer, "is_global_zero", True):
            print(f"Curriculum update -> empty_cells={new_num_cells}")

    def on_train_epoch_end(self) -> None:
        train_loss = self._metric_to_float("train_loss")
        if train_loss is None:
            return
        self.history.append(train_loss)
        self._maybe_increase_difficulty(train_loss)
