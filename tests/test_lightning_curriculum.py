from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from jepa_sudoku.training.lightning_module import LightningTrainConfig, SudokuLightningModule


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
    def __init__(self, empty_cells: int) -> None:
        self.current_num_cells_to_mask = empty_cells
        self.updated_to: list[int] = []
        self.epochs_set: list[int] = []

    def set_num_cells_to_mask(self, num_cells_to_mask: int) -> None:
        self.current_num_cells_to_mask = num_cells_to_mask
        self.updated_to.append(num_cells_to_mask)

    def set_epoch(self, epoch: int) -> None:
        self.epochs_set.append(epoch)


def _build_module(empty_cells: int = 4) -> tuple[SudokuLightningModule, _FakeDataModule]:
    representation = _FakeRepresentation()
    module = SudokuLightningModule(
        encoder=_FakeEncoder(representation),
        predictor=_FakePredictor(representation),
        representation=representation,
        config=LightningTrainConfig(
            curriculum_enabled=True,
            curriculum_mode="adaptive",
            curriculum_step=1,
            curriculum_patience=3,
            curriculum_min_delta=1.0e-3,
            curriculum_max_mask=8,
        ),
    )
    data_module = _FakeDataModule(empty_cells)
    module._trainer = SimpleNamespace(datamodule=data_module)
    return module, data_module


def test_adaptive_curriculum_does_not_advance_while_validation_loss_improves() -> None:
    module, data_module = _build_module()

    module._maybe_increase_difficulty(1.0)
    module._maybe_increase_difficulty(0.9)
    module._maybe_increase_difficulty(0.8)
    module._maybe_increase_difficulty(0.7)

    assert data_module.updated_to == []


def test_adaptive_curriculum_advances_after_validation_loss_plateaus() -> None:
    module, data_module = _build_module()

    module._maybe_increase_difficulty(1.0)
    module._maybe_increase_difficulty(1.0)
    module._maybe_increase_difficulty(1.0)
    assert data_module.updated_to == []

    module._maybe_increase_difficulty(1.0)
    assert data_module.updated_to == [5]


def test_adaptive_curriculum_resets_best_loss_after_difficulty_bump() -> None:
    module, data_module = _build_module()

    module._maybe_increase_difficulty(1.0)
    module._maybe_increase_difficulty(1.0)
    module._maybe_increase_difficulty(1.0)
    module._maybe_increase_difficulty(1.0)
    assert data_module.updated_to == [5]

    module._maybe_increase_difficulty(0.8)
    module._maybe_increase_difficulty(0.8)
    assert data_module.updated_to == [5]


def test_linear_curriculum_uses_global_step_on_train_batch_start() -> None:
    representation = _FakeRepresentation()
    module = SudokuLightningModule(
        encoder=_FakeEncoder(representation),
        predictor=_FakePredictor(representation),
        representation=representation,
        config=LightningTrainConfig(
            curriculum_enabled=True,
            curriculum_mode="linear",
            curriculum_max_mask=8,
        ),
    )
    data_module = _FakeDataModule(empty_cells=1)
    module._trainer = SimpleNamespace(datamodule=data_module, global_step=17)

    module.on_train_batch_start((torch.empty(0), torch.empty(0), torch.empty(0)), 0)

    assert data_module.epochs_set == [17]
