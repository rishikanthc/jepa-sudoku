from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from experiments import build_one_batch_overfit_config


def test_build_one_batch_overfit_config_applies_debug_preset() -> None:
    base_config = OmegaConf.load("configs/default.yaml")

    config = build_one_batch_overfit_config(base_config)

    assert config.data.num_samples == 24
    assert config.data.batch_size == 24
    assert config.data.randomize_mask_per_access is False
    assert config.data.shuffle is False
    assert config.validation.enabled is False
    assert config.training.device == "cpu"


def test_realistic_training_experiment_config_builds_train_val_test_split() -> None:
    config_dir = str(Path(__file__).resolve().parent.parent / "configs")

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(config_name="default", overrides=["experiment=realistic_training"])

    assert config.data.num_samples == 10000
    assert config.validation.enabled is True
    assert config.validation.num_samples == 5000
    assert config.test.enabled is True
    assert config.test.num_samples == 5000
