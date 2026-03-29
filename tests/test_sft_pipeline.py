from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from hydra import compose, initialize_config_dir

from jepa_sudoku.training.sft_data import SFTLightningDataModule
from jepa_sudoku.training.sft_experiments import (
    CurriculumAwareEarlyStopping,
    _load_pretrain_config,
    build_sft_data_config,
    build_sft_trainer,
    run_sft_experiment,
)
from jepa_sudoku.training.lightning_module import LightningTrainConfig, SudokuLightningModule
from jepa_sudoku.model.models import Encoder, Predictor, SudokuRepresentation, TransformerConfig


def _write_pretrain_checkpoint(path: Path) -> None:
    transformer_config = TransformerConfig(
        context_size=81,
        n_heads=2,
        head_dim=256,
        n_layers=2,
        d_ff=1024,
        dropout=0.1,
    )
    representation = SudokuRepresentation(d_model=transformer_config.d_model, seed=42)
    encoder = Encoder(transformer_config, representation=representation)
    predictor = Predictor(transformer_config, representation=representation)
    module = SudokuLightningModule(
        encoder=encoder,
        predictor=predictor,
        representation=representation,
        config=LightningTrainConfig(learning_rate=1e-4),
    )
    torch.save({"state_dict": module.state_dict()}, path)


def test_sft_datamodule_stratifies_val_and_test_by_difficulty(tmp_path) -> None:
    dataset_path = tmp_path / "solutions.pt"
    solution_boards = torch.randint(1, 10, (30, 81), dtype=torch.uint8)
    torch.save({"solution_boards": solution_boards}, dataset_path)

    config_dir = str(Path(__file__).resolve().parent.parent / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(
            config_name="sft",
            overrides=[
                f"data.dataset_path={dataset_path}",
                "data.num_samples=30",
                "data.batch_size=4",
                "data.eval_batch_size=2",
                "data.num_workers=0",
                "data.train_num_cells_to_mask=6",
                "data.difficulty_range.start=8",
                "data.difficulty_range.end=12",
                "data.difficulty_range.step=2",
                "curriculum.enabled=false",
            ],
        )

    data_module = SFTLightningDataModule(config=build_sft_data_config(config))
    data_module.setup("fit")

    assert data_module.val_difficulties == [8, 10, 12]
    assert data_module.test_difficulties == [8, 10, 12]
    assert [dataset.num_cells_to_mask for dataset in data_module.val_datasets] == [8, 10, 12]
    assert [dataset.num_cells_to_mask for dataset in data_module.test_datasets] == [8, 10, 12]
    assert sum(len(dataset) for dataset in data_module.val_datasets) == 3
    assert sum(len(dataset) for dataset in data_module.test_datasets) == 6


def test_sft_curriculum_forces_in_process_train_loading(tmp_path) -> None:
    dataset_path = tmp_path / "solutions.pt"
    solution_boards = torch.randint(1, 10, (30, 81), dtype=torch.uint8)
    torch.save({"solution_boards": solution_boards}, dataset_path)

    config_dir = str(Path(__file__).resolve().parent.parent / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(
            config_name="sft",
            overrides=[
                f"data.dataset_path={dataset_path}",
                "data.num_samples=30",
                "data.batch_size=4",
                "data.num_workers=4",
                "data.difficulty_range.start=8",
                "data.difficulty_range.end=12",
                "data.difficulty_range.step=2",
                "curriculum.enabled=true",
                "curriculum.mode=linear",
                "curriculum.num_steps=4",
            ],
        )

    data_module = SFTLightningDataModule(
        config=build_sft_data_config(config),
        curriculum_enabled=config.curriculum.enabled,
        curriculum_mode=config.curriculum.mode,
        curriculum_max_mask=config.curriculum.max_mask,
        curriculum_num_steps=config.curriculum.num_steps,
    )
    data_module.setup("fit")
    train_loader = data_module.train_dataloader()

    assert train_loader.num_workers == 4
    data_module.set_epoch(4)
    assert data_module.current_num_cells_to_mask == config.curriculum.max_mask


def test_load_pretrain_config_reuses_existing_hydra_context() -> None:
    config_dir = str(Path(__file__).resolve().parent.parent / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        _ = compose(config_name="sft")
        pretrain_config = _load_pretrain_config("lightning_ddp_2gpu")

    assert pretrain_config.ssp.dim == 512
    assert pretrain_config.model.n_heads == 2
    assert pretrain_config.model.head_dim == 256


def test_sft_trainer_accepts_step_based_validation_interval() -> None:
    config_dir = str(Path(__file__).resolve().parent.parent / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(
            config_name="sft",
            overrides=[
                "trainer.accelerator=cpu",
                "trainer.devices=1",
                "trainer.strategy=auto",
                "trainer.val_check_interval=25",
                "trainer.check_val_every_n_epoch=null",
                "logging.csv_path=logs/test-sft-metrics.csv",
                "checkpoint.dirpath=checkpoints",
            ],
        )

    trainer = build_sft_trainer(config)

    assert trainer.val_check_interval == 25
    assert trainer.check_val_every_n_epoch is None


def test_curriculum_aware_early_stopping_waits_for_max_difficulty() -> None:
    callback = CurriculumAwareEarlyStopping(
        curriculum_enabled=True,
        curriculum_max_mask=32,
        monitor="val_board_accuracy",
        mode="max",
        patience=3,
        min_delta=1.0e-4,
        check_on_train_epoch_end=False,
    )

    trainer_before_max = SimpleNamespace(
        datamodule=SimpleNamespace(current_num_cells_to_mask=16)
    )
    trainer_at_max = SimpleNamespace(
        datamodule=SimpleNamespace(current_num_cells_to_mask=32)
    )

    assert callback._curriculum_at_max_difficulty(trainer_before_max) is False
    assert callback._curriculum_at_max_difficulty(trainer_at_max) is True


def test_sft_pipeline_saves_decoder_checkpoint_and_logs_metrics(tmp_path) -> None:
    dataset_path = tmp_path / "solutions.pt"
    checkpoint_path = tmp_path / "pretrain.ckpt"
    csv_path = tmp_path / "sft-metrics.csv"
    solution_boards = torch.randint(1, 10, (20, 81), dtype=torch.uint8)
    torch.save({"solution_boards": solution_boards}, dataset_path)
    _write_pretrain_checkpoint(checkpoint_path)

    config_dir = str(Path(__file__).resolve().parent.parent / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(
            config_name="sft",
            overrides=[
                f"pretrained.checkpoint_path={checkpoint_path}",
                "pretrained.config_name=default",
                "decoder_model.n_heads=2",
                "decoder_model.head_dim=256",
                "decoder_model.n_layers=1",
                "decoder_model.d_ff=512",
                "decoder_model.dropout=0.0",
                f"data.dataset_path={dataset_path}",
                "data.num_samples=20",
                "data.batch_size=2",
                "data.eval_batch_size=2",
                "data.num_workers=0",
                "data.train_num_cells_to_mask=4",
                "data.difficulty_range.start=2",
                "data.difficulty_range.end=2",
                "data.difficulty_range.step=1",
                "curriculum.enabled=false",
                "training.max_epochs=1",
                "trainer.accelerator=cpu",
                "trainer.devices=1",
                "trainer.strategy=auto",
                "trainer.fast_dev_run=false",
                "trainer.limit_train_batches=1",
                "trainer.limit_val_batches=1",
                "trainer.limit_test_batches=1",
                "trainer.val_check_interval=1",
                "trainer.check_val_every_n_epoch=null",
                f"checkpoint.dirpath={tmp_path}",
                "checkpoint.filename=sft-decoder-test",
                f"logging.csv_path={csv_path}",
                "early_stopping.patience=3",
            ],
        )

    result = run_sft_experiment(config)

    assert result.checkpoint_path is not None
    saved = torch.load(result.checkpoint_path, map_location="cpu")
    assert "decoder_state_dict" in saved
    assert "classifier_head_state_dict" in saved
    assert all(key.startswith("blocks.") or key.startswith("drop") or key.startswith("out_ln") or key.startswith("head") for key in saved["decoder_state_dict"])
    assert set(saved["classifier_head_state_dict"]) == {"weight", "bias"}
    assert result.test_metrics is not None
    assert "test_cell_accuracy_overall" in result.test_metrics
    assert "test_board_accuracy_overall" in result.test_metrics
    assert csv_path.exists()
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "stage,epoch,batch_idx,global_step,difficulty,train_loss,train_cell_accuracy,train_board_accuracy,empty_cells,val_loss,val_cell_accuracy,val_board_accuracy"
    assert len(lines) >= 3
    assert any(",overall," in line for line in lines[1:])
    assert any(",2," in line for line in lines[1:])
