from __future__ import annotations

from dataclasses import dataclass
from itertools import chain

import torch
from torch import Tensor
from torch.optim import Adam

from datamodule import SudokuDataModule
from losses import cosine_loss
from models import Encoder, Predictor
from ssp import ThreeAxisSSP


@dataclass
class TrainConfig:
    max_epochs: int = 100
    patience: int = 10
    learning_rate: float = 1e-3
    min_delta: float = 1e-6
    device: str = "cpu"


class SudokuTrainer:
    """
    Orchestrates an encoder/predictor training loop for Sudoku cell prediction.
    """

    def __init__(
        self,
        encoder: Encoder,
        predictor: Predictor,
        data_module: SudokuDataModule,
        embedding: ThreeAxisSSP,
        config: TrainConfig,
        val_data_module: SudokuDataModule | None = None,
    ) -> None:
        if encoder.embedding is not predictor.embedding:
            raise ValueError("Encoder and predictor must share the same embedding instance.")
        if encoder.embedding is not embedding:
            raise ValueError("Pass the same embedding instance used by encoder/predictor.")

        self.encoder = encoder
        self.predictor = predictor
        self.embedding = embedding
        self.data_module = data_module
        self.val_data_module = val_data_module
        self.config = config

        self.encoder.to(config.device)
        self.predictor.to(config.device)
        self.embedding.to(config.device)

        self.optimizer = Adam(
            chain(self.encoder.parameters(), self.predictor.parameters()),
            lr=config.learning_rate,
        )

    @staticmethod
    def _build_value_only_coords(solution: Tensor) -> Tensor:
        """
        Build z-only coords for target encoding:
        set x,y = 0 and keep z from solution.
        """
        coords = torch.zeros_like(solution)
        coords[:, :, 2] = solution[:, :, 2]
        return coords

    def _run_epoch(self, *, train: bool) -> float:
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
        num_batches = 0

        for batch in dataloader:
            num_batches += 1
            puzzle, solution, query, _mask = batch
            puzzle = puzzle.to(self.config.device)
            solution = solution.to(self.config.device)
            query = query.to(self.config.device)

            with torch.set_grad_enabled(train):
                encoder_out = self.encoder(puzzle)
                pred_vectors = self.predictor(query, encoder_out)

                target_coords = self._build_value_only_coords(solution)
                target_vectors = self.embedding.encode(target_coords)
                loss = cosine_loss(pred_vectors, target_vectors)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()

        if num_batches == 0:
            return 0.0
        return total_loss / num_batches

    def train(self) -> list[tuple[float, float | None]]:
        """
        Runs training with simple early stopping.
        """
        history: list[tuple[float, float | None]] = []
        best_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(self.config.max_epochs):
            self.data_module.set_epoch(epoch)
            train_loss = self._run_epoch(train=True)

            val_loss = None
            if self.val_data_module is not None:
                self.val_data_module.set_epoch(epoch)
                val_loss = self._run_epoch(train=False)

            current_empty_cells = self.data_module.current_num_cells_to_mask
            if val_loss is not None:
                print(
                    f"Epoch {epoch+1:03d}/{self.config.max_epochs:03d} | "
                    f"empty_cells={current_empty_cells:02d} | "
                    f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
                )
            else:
                print(
                    f"Epoch {epoch+1:03d}/{self.config.max_epochs:03d} | "
                    f"empty_cells={current_empty_cells:02d} | "
                    f"train_loss={train_loss:.6f}"
                )

            monitor_loss = val_loss if val_loss is not None else train_loss
            history.append((train_loss, val_loss))

            if monitor_loss + self.config.min_delta < best_loss:
                best_loss = monitor_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.config.patience:
                break

        return history
