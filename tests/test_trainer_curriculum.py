from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from trainermodule import SudokuTrainer, TrainConfig


class _FakeRepresentation(nn.Module):
    pass


class _FakeEncoder(nn.Module):
    def __init__(self, representation: nn.Module) -> None:
        super().__init__()
        self.representation = representation
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _FakePredictor(nn.Module):
    def __init__(self, representation: nn.Module) -> None:
        super().__init__()
        self.representation = representation
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, query: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        return query


class _FakeDataModule:
    def __init__(self, num_cells_to_mask: int) -> None:
        self.current_num_cells_to_mask = num_cells_to_mask
        self.epoch_history: list[int] = []
        self.mask_updates: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self.epoch_history.append(epoch)

    def set_num_cells_to_mask(self, num_cells_to_mask: int) -> None:
        self.current_num_cells_to_mask = num_cells_to_mask
        self.mask_updates.append(num_cells_to_mask)


class _ScriptedTrainer(SudokuTrainer):
    def __init__(
        self,
        train_losses: Sequence[float],
        val_losses: Sequence[float],
        *,
        curriculum_patience: int,
        starting_empty_cells: int = 4,
    ) -> None:
        representation = _FakeRepresentation()
        encoder = _FakeEncoder(representation)
        predictor = _FakePredictor(representation)
        data_module = _FakeDataModule(starting_empty_cells)
        val_data_module = _FakeDataModule(starting_empty_cells)
        super().__init__(
            encoder=encoder,
            predictor=predictor,
            data_module=data_module,
            val_data_module=val_data_module,
            representation=representation,
            config=TrainConfig(
                max_epochs=len(train_losses),
                patience=10_000,
                curriculum_enabled=True,
                curriculum_mode="adaptive",
                curriculum_step=1,
                curriculum_patience=curriculum_patience,
                curriculum_min_delta=1e-6,
                curriculum_max_mask=64,
            ),
        )
        self._train_losses = list(train_losses)
        self._val_losses = list(val_losses)
        self._epoch_idx = 0

    def _run_epoch(self, data_module: _FakeDataModule, *, train: bool) -> tuple[float, float]:
        idx = self._epoch_idx
        loss = self._train_losses[idx] if train else self._val_losses[idx]
        if not train:
            self._epoch_idx += 1
        return loss, 0.0


def test_adaptive_curriculum_does_not_advance_while_validation_keeps_improving() -> None:
    trainer = _ScriptedTrainer(
        train_losses=[1.0, 0.9, 0.8, 0.7, 0.6],
        val_losses=[0.50, 0.40, 0.30, 0.20, 0.10],
        curriculum_patience=2,
    )

    trainer.train()

    assert trainer.data_module.mask_updates == []
    assert trainer.val_data_module.mask_updates == []


def test_adaptive_curriculum_resets_plateau_tracking_after_difficulty_increase() -> None:
    trainer = _ScriptedTrainer(
        train_losses=[1.0] * 7,
        val_losses=[0.50, 0.50, 0.70, 0.60, 0.55, 0.54, 0.53],
        curriculum_patience=2,
    )

    trainer.train()

    assert trainer.data_module.mask_updates == [5]
    assert trainer.val_data_module.mask_updates == [5]
