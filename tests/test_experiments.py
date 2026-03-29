from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from jepa_sudoku.data.datamodule import SudokuPuzzleDataset
from jepa_sudoku.training.experiments import build_data_module, build_one_batch_overfit_config
from jepa_sudoku.training.lightning_data import LightningSudokuDataModule


def test_build_one_batch_overfit_config_applies_debug_preset() -> None:
    base_config = OmegaConf.load("configs/default.yaml")

    config = build_one_batch_overfit_config(base_config)

    assert config.data.num_samples == 24
    assert config.data.batch_size == 24
    assert config.data.randomize_mask_per_access is False
    assert config.data.shuffle is False
    assert config.curriculum.enabled is False


def test_lightning_ddp_experiment_config_enables_two_gpu_trainer() -> None:
    config_dir = str(Path(__file__).resolve().parent.parent / "configs")

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(config_name="default", overrides=["experiment=lightning_ddp_2gpu"])

    assert config.trainer.accelerator == "gpu"
    assert config.trainer.devices == 2
    assert config.trainer.strategy == "ddp"
    assert config.trainer.precision == "16-mixed"
    assert config.trainer.matmul_precision == "high"
    assert config.data.dataset_path.endswith("datasets/pretrain_default.pt")
    assert config.data.num_cells_to_mask == 20
    assert config.curriculum.enabled is False


def test_build_data_module_returns_lightning_datamodule() -> None:
    base_config = OmegaConf.load("configs/default.yaml")

    data_module = build_data_module(base_config)

    assert isinstance(data_module, LightningSudokuDataModule)


def test_curriculum_updates_change_train_mask_count_only() -> None:
    config_dir = str(Path(__file__).resolve().parent.parent / "configs")

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(
            config_name="default",
            overrides=[
                "data.num_cells_to_mask=2",
                "curriculum.enabled=true",
            ],
        )

    data_module = build_data_module(config)
    data_module.setup()
    data_module.set_num_cells_to_mask(5)

    assert data_module.current_num_cells_to_mask == 5


def test_adjacent_seeds_do_not_overlap_board_templates() -> None:
    train_dataset = SudokuPuzzleDataset(
        num_samples=32,
        num_cells_to_mask=1,
        seed=123,
        unique_solution=False,
        randomize_mask_per_access=False,
    )
    next_dataset = SudokuPuzzleDataset(
        num_samples=32,
        num_cells_to_mask=1,
        seed=124,
        unique_solution=False,
        randomize_mask_per_access=False,
    )

    train_templates = {
        tuple(int(x) for x in train_dataset._get_or_build_template(index).tolist())
        for index in range(32)
    }
    next_templates = {
        tuple(int(x) for x in next_dataset._get_or_build_template(index).tolist())
        for index in range(32)
    }

    assert train_templates.isdisjoint(next_templates)
