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
    ) -> None:
        if encoder.embedding is not predictor.embedding:
            raise ValueError("Encoder and predictor must share the same embedding instance.")
        if encoder.embedding is not embedding:
            raise ValueError("Pass the same embedding instance used by encoder/predictor.")

        self.encoder = encoder
        self.predictor = predictor
        self.embedding = embedding
        self.data_module = data_module
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

    def _train_single_epoch(self) -> float:
        self.encoder.train()
        self.predictor.train()

        total_loss = 0.0
        num_batches = 0

        for batch in self.data_module.train_dataloader():
            num_batches += 1
            puzzle, solution, query, _mask = batch
            puzzle = puzzle.to(self.config.device)
            solution = solution.to(self.config.device)
            query = query.to(self.config.device)

            encoder_out = self.encoder(puzzle)
            pred_vectors = self.predictor(query, encoder_out)

            target_coords = self._build_value_only_coords(solution)
            target_vectors = self.embedding.encode(target_coords)
            loss = cosine_loss(pred_vectors, target_vectors)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        if num_batches == 0:
            return 0.0
        return total_loss / num_batches

    def train(self) -> list[float]:
        """
        Runs training with simple early stopping.
        """
        history: list[float] = []
        best_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(self.config.max_epochs):
            self.data_module.set_epoch(epoch)
            epoch_loss = self._train_single_epoch()
            history.append(epoch_loss)

            if epoch_loss + self.config.min_delta < best_loss:
                best_loss = epoch_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.config.patience:
                break

        return history
