from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightning.pytorch as pl
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam

from jepa_sudoku.model.models import Encoder, Predictor, SudokuRepresentation


@dataclass
class SFTTrainConfig:
    learning_rate: float = 1e-4
    curriculum_enabled: bool = False
    curriculum_mode: str = "adaptive"
    curriculum_monitor: str = "val_loss"
    curriculum_step: int = 1
    curriculum_patience: int = 3
    curriculum_min_delta: float = 1.0e-4
    curriculum_max_mask: int = 81


class FrozenEncoderSFTModule(pl.LightningModule):
    def __init__(
        self,
        *,
        encoder: Encoder,
        decoder: Predictor,
        representation: SudokuRepresentation,
        config: SFTTrainConfig,
        val_difficulties: list[int],
        test_difficulties: list[int],
        checkpoint_monitor: str,
        checkpoint_mode: str,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.representation = representation
        self.config = config
        self.val_difficulties = val_difficulties
        self.test_difficulties = test_difficulties
        self.history: list[float] = []
        self.test_metrics: dict[str, float] = {}
        self.latest_val_metrics: dict[str, float] | None = None
        self.classifier_head = nn.Linear(self.decoder.config.d_model, 9)
        self.checkpoint_monitor = checkpoint_monitor
        self.checkpoint_mode = checkpoint_mode
        if checkpoint_mode not in {"min", "max"}:
            raise ValueError("checkpoint_mode must be 'min' or 'max'")
        self.best_monitor_score = float("-inf") if checkpoint_mode == "max" else float("inf")
        self.best_trainable_state_dict: dict[str, Tensor] | None = None
        self._curriculum_monitor_modes = {
            "train_loss": "min",
            "val_loss": "min",
            "val_cell_accuracy": "max",
            "val_board_accuracy": "max",
        }
        if self.config.curriculum_monitor not in self._curriculum_monitor_modes:
            raise ValueError(
                "Unsupported curriculum_monitor="
                f"{self.config.curriculum_monitor!r}. "
                "Use one of: train_loss, val_loss, val_cell_accuracy, val_board_accuracy."
            )

        for module in (self.encoder, self.representation):
            for parameter in module.parameters():
                parameter.requires_grad = False

        self._curriculum_plateau_counter = 0
        self._curriculum_best_metric = self._initial_curriculum_best_metric()
        self._stage_totals: dict[str, dict[str, Tensor]] = {}

    def _initial_curriculum_best_metric(self) -> float:
        mode = self._curriculum_monitor_modes[self.config.curriculum_monitor]
        return float("inf") if mode == "min" else float("-inf")

    def _trainable_state_dict(self) -> dict[str, dict[str, Tensor]]:
        return {
            "decoder_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in self.decoder.state_dict().items()
                if not key.startswith("representation.") and not key.startswith("embedding.")
            },
            "classifier_head_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in self.classifier_head.state_dict().items()
            },
        }

    def configure_optimizers(self) -> Adam:
        return Adam(
            list(self.decoder.parameters()) + list(self.classifier_head.parameters()),
            lr=self.config.learning_rate,
        )

    def _compute_logits_and_targets(
        self, batch: tuple[Tensor, Tensor, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        puzzle, solution, query = batch
        self.encoder.eval()
        self.representation.eval()
        with torch.no_grad():
            encoder_out = self.encoder(puzzle)
        pred_vectors = self.decoder(query, encoder_out)
        logits = self.classifier_head(pred_vectors)
        targets = solution[..., 2].to(dtype=torch.long) - 1
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )
        predictions = logits.argmax(dim=-1)
        return loss, predictions, targets, logits

    def _board_correct(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return predictions.eq(targets).all(dim=-1)

    def training_step(
        self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> dict[str, Tensor]:
        loss, predictions, targets, _ = self._compute_logits_and_targets(batch)
        cell_accuracy = predictions.eq(targets).float().mean()
        board_accuracy = self._board_correct(predictions, targets).float().mean()
        empty_cells = torch.tensor(float(batch[2].shape[1]), device=loss.device)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train_cell_accuracy",
            cell_accuracy,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train_board_accuracy",
            board_accuracy,
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
            "train_cell_accuracy": cell_accuracy.detach(),
            "train_board_accuracy": board_accuracy.detach(),
            "empty_cells": empty_cells.detach(),
        }

    def on_fit_start(self) -> None:
        data_module = self.trainer.datamodule
        if data_module is None:
            return
        if self.config.curriculum_enabled and self.config.curriculum_mode == "adaptive":
            data_module.set_num_cells_to_mask(data_module.config.train_num_cells_to_mask)

    def on_train_epoch_start(self) -> None:
        data_module = self.trainer.datamodule
        if data_module is None:
            return
        if self.config.curriculum_enabled and self.config.curriculum_mode == "linear":
            data_module.set_epoch(self.global_step)
        else:
            data_module.set_epoch(self.current_epoch)

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

    def on_train_epoch_end(self) -> None:
        return

    def on_train_batch_end(
        self,
        outputs: dict[str, Tensor] | Tensor | None,
        batch: tuple[Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> None:
        if self.config.curriculum_monitor != "train_loss":
            return
        if not isinstance(outputs, dict):
            return
        train_loss = outputs.get("train_loss")
        if train_loss is None:
            return
        self._maybe_increase_difficulty(float(train_loss.detach().cpu().item()))

    def _reset_stage_totals(self, stage: str) -> None:
        self._stage_totals[stage] = {}

    def on_validation_epoch_start(self) -> None:
        self._reset_stage_totals("val")

    def on_test_epoch_start(self) -> None:
        self._reset_stage_totals("test")

    def _accumulate_stage_metrics(
        self,
        *,
        stage: str,
        bucket: str,
        loss: Tensor,
        predictions: Tensor,
        targets: Tensor,
    ) -> None:
        bucket_metrics = self._stage_totals.setdefault(stage, {}).setdefault(
            bucket,
            {
                "loss_sum": torch.zeros((), device=loss.device),
                "cell_correct": torch.zeros((), device=loss.device),
                "cell_total": torch.zeros((), device=loss.device),
                "board_correct": torch.zeros((), device=loss.device),
                "board_total": torch.zeros((), device=loss.device),
            },
        )
        board_correct = self._board_correct(predictions, targets)
        bucket_metrics["loss_sum"] += loss.detach() * targets.numel()
        bucket_metrics["cell_correct"] += predictions.eq(targets).sum()
        bucket_metrics["cell_total"] += torch.tensor(float(targets.numel()), device=loss.device)
        bucket_metrics["board_correct"] += board_correct.sum()
        bucket_metrics["board_total"] += torch.tensor(float(targets.shape[0]), device=loss.device)

    def validation_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        loss, predictions, targets, _ = self._compute_logits_and_targets(batch)
        difficulty = self.val_difficulties[dataloader_idx]
        self._accumulate_stage_metrics(
            stage="val",
            bucket="overall",
            loss=loss,
            predictions=predictions,
            targets=targets,
        )
        self._accumulate_stage_metrics(
            stage="val",
            bucket=f"difficulty_{difficulty}",
            loss=loss,
            predictions=predictions,
            targets=targets,
        )

    def test_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        loss, predictions, targets, _ = self._compute_logits_and_targets(batch)
        difficulty = self.test_difficulties[dataloader_idx]
        self._accumulate_stage_metrics(
            stage="test",
            bucket="overall",
            loss=loss,
            predictions=predictions,
            targets=targets,
        )
        self._accumulate_stage_metrics(
            stage="test",
            bucket=f"difficulty_{difficulty}",
            loss=loss,
            predictions=predictions,
            targets=targets,
        )

    def _all_reduce(self, tensor: Tensor) -> Tensor:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def _finalize_stage_metrics(self, stage: str) -> dict[str, float]:
        finalized: dict[str, float] = {}
        for bucket, totals in self._stage_totals.get(stage, {}).items():
            reduced = {name: self._all_reduce(value.clone()) for name, value in totals.items()}
            cell_total = reduced["cell_total"].clamp_min(1.0)
            board_total = reduced["board_total"].clamp_min(1.0)
            finalized[f"{stage}_loss_{bucket}"] = float(
                (reduced["loss_sum"] / cell_total).detach().cpu().item()
            )
            finalized[f"{stage}_cell_accuracy_{bucket}"] = float(
                (reduced["cell_correct"] / cell_total).detach().cpu().item()
            )
            finalized[f"{stage}_board_accuracy_{bucket}"] = float(
                (reduced["board_correct"] / board_total).detach().cpu().item()
            )
        return finalized

    def _metric_to_float(self, metric_name: str) -> float | None:
        metric = self.trainer.callback_metrics.get(metric_name)
        if metric is None:
            return None
        return float(metric.detach().cpu().item())

    def _maybe_increase_difficulty(self, monitored_metric: float) -> None:
        if not self.config.curriculum_enabled or self.config.curriculum_mode != "adaptive":
            return
        data_module = self.trainer.datamodule
        if data_module is None:
            return
        if data_module.current_num_cells_to_mask >= self.config.curriculum_max_mask:
            return

        mode = self._curriculum_monitor_modes[self.config.curriculum_monitor]
        improved = (
            monitored_metric < (self._curriculum_best_metric - self.config.curriculum_min_delta)
            if mode == "min"
            else monitored_metric > (self._curriculum_best_metric + self.config.curriculum_min_delta)
        )
        if improved:
            self._curriculum_best_metric = monitored_metric
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
        self._curriculum_best_metric = self._initial_curriculum_best_metric()
        if getattr(self.trainer, "is_global_zero", True):
            print(f"SFT curriculum update -> empty_cells={new_num_cells}")

    def on_validation_epoch_end(self) -> None:
        metrics = self._finalize_stage_metrics("val")
        if not metrics:
            return
        overall_loss = metrics["val_loss_overall"]
        overall_cell_accuracy = metrics["val_cell_accuracy_overall"]
        overall_board_accuracy = metrics["val_board_accuracy_overall"]
        self.log("val_loss", overall_loss, prog_bar=True, logger=False, sync_dist=True)
        self.log(
            "val_cell_accuracy",
            overall_cell_accuracy,
            prog_bar=True,
            logger=False,
            sync_dist=True,
        )
        self.log(
            "val_board_accuracy",
            overall_board_accuracy,
            prog_bar=True,
            logger=False,
            sync_dist=True,
        )
        self.history.append(overall_loss)
        self.latest_val_metrics = {
            "val_loss": overall_loss,
            "val_cell_accuracy": overall_cell_accuracy,
            "val_board_accuracy": overall_board_accuracy,
        }
        for difficulty in self.val_difficulties:
            bucket = f"difficulty_{difficulty}"
            loss_key = f"val_loss_{bucket}"
            cell_key = f"val_cell_accuracy_{bucket}"
            board_key = f"val_board_accuracy_{bucket}"
            if loss_key not in metrics:
                continue
            self.latest_val_metrics[loss_key] = metrics[loss_key]
            self.latest_val_metrics[cell_key] = metrics[cell_key]
            self.latest_val_metrics[board_key] = metrics[board_key]
        self._maybe_update_best_decoder(
            self.latest_val_metrics
        )
        if self.config.curriculum_monitor == "val_loss":
            self._maybe_increase_difficulty(overall_loss)
        elif self.config.curriculum_monitor == "val_cell_accuracy":
            self._maybe_increase_difficulty(overall_cell_accuracy)
        elif self.config.curriculum_monitor == "val_board_accuracy":
            self._maybe_increase_difficulty(overall_board_accuracy)

    def on_test_epoch_end(self) -> None:
        metrics = self._finalize_stage_metrics("test")
        self.test_metrics = metrics
        if not metrics:
            return
        self.log(
            "test_loss",
            metrics["test_loss_overall"],
            prog_bar=True,
            logger=False,
            sync_dist=True,
        )
        self.log(
            "test_cell_accuracy",
            metrics["test_cell_accuracy_overall"],
            prog_bar=True,
            logger=False,
            sync_dist=True,
        )
        self.log(
            "test_board_accuracy",
            metrics["test_board_accuracy_overall"],
            prog_bar=True,
            logger=False,
            sync_dist=True,
        )
        for difficulty in self.test_difficulties:
            bucket = f"difficulty_{difficulty}"
            loss_key = f"test_loss_{bucket}"
            cell_key = f"test_cell_accuracy_{bucket}"
            board_key = f"test_board_accuracy_{bucket}"
            if loss_key not in metrics:
                continue
            self.log(loss_key, metrics[loss_key], logger=False, sync_dist=True)
            self.log(cell_key, metrics[cell_key], logger=False, sync_dist=True)
            self.log(board_key, metrics[board_key], logger=False, sync_dist=True)

    def _maybe_update_best_decoder(self, metrics: dict[str, float]) -> None:
        score = metrics.get(self.checkpoint_monitor)
        if score is None:
            return
        improved = (
            score > self.best_monitor_score
            if self.checkpoint_mode == "max"
            else score < self.best_monitor_score
        )
        if not improved:
            return
        self.best_monitor_score = score
        self.best_trainable_state_dict = self._trainable_state_dict()

    def restore_best_weights(self) -> None:
        if self.best_trainable_state_dict is None:
            return
        self.decoder.load_state_dict(
            self.best_trainable_state_dict["decoder_state_dict"], strict=False
        )
        self.classifier_head.load_state_dict(
            self.best_trainable_state_dict["classifier_head_state_dict"], strict=True
        )

    def save_best_decoder_checkpoint(self, dirpath: str, filename: str) -> str | None:
        if self.best_trainable_state_dict is None:
            return None
        checkpoint_path = Path(dirpath) / f"{filename}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                **self.best_trainable_state_dict,
                "monitor": self.checkpoint_monitor,
                "score": self.best_monitor_score,
            },
            checkpoint_path,
        )
        return str(checkpoint_path)
