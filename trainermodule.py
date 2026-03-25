from __future__ import annotations

from dataclasses import dataclass
from itertools import chain

import torch
from torch import Tensor
from torch.optim import Adam

from datamodule import SudokuDataModule
from losses import prototype_classification_loss
from models import Encoder, Predictor, SudokuRepresentation


@dataclass
class TrainConfig:
    max_epochs: int = 100
    patience: int = 10
    learning_rate: float = 1e-3
    prototype_logit_scale: float = 1.0
    min_delta: float = 1e-6
    device: str = "cpu"
    curriculum_enabled: bool = False
    curriculum_mode: str = "adaptive"
    curriculum_step: int = 1
    curriculum_patience: int = 10
    curriculum_min_delta: float = 1e-6
    curriculum_max_mask: int = 64
    curriculum_num_epochs: int = 20


class SudokuTrainer:
    """
    Orchestrates an encoder/predictor training loop for Sudoku cell prediction.
    """

    def __init__(
        self,
        encoder: Encoder,
        predictor: Predictor,
        data_module: SudokuDataModule,
        representation: SudokuRepresentation,
        config: TrainConfig,
        val_data_module: SudokuDataModule | None = None,
    ) -> None:
        if encoder.representation is not predictor.representation:
            raise ValueError("Encoder and predictor must share the same representation instance.")
        if encoder.representation is not representation:
            raise ValueError("Pass the same representation instance used by encoder/predictor.")

        self.encoder = encoder
        self.predictor = predictor
        self.representation = representation
        self.data_module = data_module
        self.val_data_module = val_data_module
        self.config = config

        self.encoder.to(config.device)
        self.predictor.to(config.device)
        self.representation.to(config.device)

        self.optimizer = Adam(
            chain(self.encoder.parameters(), self.predictor.parameters()),
            lr=config.learning_rate,
        )
        if config.prototype_logit_scale <= 0.0:
            raise ValueError(
                f"prototype_logit_scale must be positive, got {config.prototype_logit_scale}."
            )
        self._plateau_counter = 0

    def _accuracy_from_logits(self, logits: Tensor, solution: Tensor) -> tuple[int, int]:
        """
        Compute exact-match accuracy of predicted digit values.

        Returns:
            num_correct, num_total
        """
        if logits.numel() == 0:
            return 0, 0

        pred_digits = logits.argmax(dim=-1).to(dtype=torch.long) + 1
        target_digits = solution[:, :, 2].to(dtype=torch.long)

        matches = pred_digits == target_digits
        num_correct = int(matches.sum().item())
        num_total = int(target_digits.numel())
        return num_correct, num_total

    def _maybe_increase_difficulty(self, monitor_loss: float, best_loss: float) -> bool:
        if self.config.curriculum_mode != "adaptive":
            return False

        if (
            not self.config.curriculum_enabled
            or self.data_module.current_num_cells_to_mask >= self.config.curriculum_max_mask
        ):
            return False

        improved = monitor_loss + self.config.curriculum_min_delta < best_loss
        if improved:
            self._plateau_counter = 0
            return False

        self._plateau_counter += 1
        if self._plateau_counter < self.config.curriculum_patience:
            return False

        new_num_cells = min(
            self.config.curriculum_max_mask,
            self.data_module.current_num_cells_to_mask + self.config.curriculum_step,
        )
        if new_num_cells == self.data_module.current_num_cells_to_mask:
            return False

        self.data_module.set_num_cells_to_mask(new_num_cells)
        if self.val_data_module is not None:
            self.val_data_module.set_num_cells_to_mask(new_num_cells)
        self._plateau_counter = 0
        return True

    def _run_epoch(self, *, train: bool) -> tuple[float, float]:
        if train:
            self.encoder.train()
            self.predictor.train()
        else:
            self.encoder.eval()
            self.predictor.eval()

        dataloader = (
            self.data_module.train_dataloader()
            if train
            else self.val_data_module.train_dataloader()
        )
        if dataloader is None:
            raise RuntimeError("Validation dataloader is not configured.")

        total_loss = 0.0
        total_correct = 0
        total_targets = 0
        num_batches = 0

        for batch in dataloader:
            num_batches += 1
            puzzle, solution, query = batch
            puzzle = puzzle.to(self.config.device)
            solution = solution.to(self.config.device)
            query = query.to(self.config.device)

            with torch.set_grad_enabled(train):
                encoder_out = self.encoder(puzzle)
                pred_vectors = self.predictor(query, encoder_out)
                logits = self.representation.logits_from_predictions(
                    pred_vectors,
                    logit_scale=self.config.prototype_logit_scale,
                )
                loss = prototype_classification_loss(logits, solution[:, :, 2])

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()
            with torch.no_grad():
                correct, num_targets = self._accuracy_from_logits(logits, solution)
                total_correct += correct
                total_targets += num_targets

        if num_batches == 0:
            return 0.0, 0.0

        avg_accuracy = total_correct / total_targets if total_targets > 0 else 0.0
        return total_loss / num_batches, avg_accuracy

    def train(self) -> list[tuple[float, float | None]]:
        """
        Runs training with simple early stopping.
        """
        history: list[tuple[float, float | None]] = []
        best_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(self.config.max_epochs):
            self.data_module.set_epoch(epoch)
            train_loss, train_acc = self._run_epoch(train=True)

            val_loss = None
            val_acc = None
            if self.val_data_module is not None:
                self.val_data_module.set_epoch(epoch)
                val_loss, val_acc = self._run_epoch(train=False)

            current_empty_cells = self.data_module.current_num_cells_to_mask
            if val_loss is not None:
                print(
                    f"Epoch {epoch+1:03d}/{self.config.max_epochs:03d} | "
                    f"empty_cells={current_empty_cells:02d} | "
                    f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                    f"train_acc={train_acc:.6f} | val_acc={val_acc:.6f}"
                )
            else:
                print(
                    f"Epoch {epoch+1:03d}/{self.config.max_epochs:03d} | "
                    f"empty_cells={current_empty_cells:02d} | "
                    f"train_loss={train_loss:.6f} | train_acc={train_acc:.6f}"
                )

            monitor_loss = val_loss if val_loss is not None else train_loss
            history.append((train_loss, val_loss))

            if monitor_loss + self.config.min_delta < best_loss:
                best_loss = monitor_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if self._maybe_increase_difficulty(monitor_loss, best_loss):
                print(
                    f"Curriculum update -> empty_cells={self.data_module.current_num_cells_to_mask}"
                )
                epochs_without_improvement = 0

            if epochs_without_improvement >= self.config.patience:
                break

        return history
