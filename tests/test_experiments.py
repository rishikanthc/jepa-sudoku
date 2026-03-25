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
