"""Unit tests for load_config (utils/config.py)."""

import pytest
import yaml
from pathlib import Path

from utils.config import Config, load_config


def _write_yaml(path: Path, content: dict) -> Path:
    config_path = path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(content, f)
    return config_path


class TestLoadConfigValid:
    def test_minimal_config(self, tmp_path):
        (tmp_path / "data" / "synthetic2").mkdir(parents=True)
        config_path = _write_yaml(
            tmp_path,
            {
                "data": {"input_path": "data/synthetic2"},
            },
        )
        config = load_config(tmp_path, config_path)

        assert isinstance(config, Config)
        assert config.per_destination_forecasting is None

    def test_with_per_destination_section(self, tmp_path):
        (tmp_path / "data" / "synthetic2").mkdir(parents=True)
        config_path = _write_yaml(
            tmp_path,
            {
                "data": {"input_path": "data/synthetic2"},
                "per_destination_forecasting": {"model_names": ["naive_forecaster"]},
            },
        )
        config = load_config(tmp_path, config_path)

        assert config.per_destination_forecasting is not None
        assert config.per_destination_forecasting.model_names == ["naive_forecaster"]


class TestLoadConfigFileErrors:
    def test_missing_config_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_config(tmp_path, tmp_path / "nonexistent.yaml")

    def test_malformed_yaml(self, tmp_path):
        config_path = tmp_path / "bad.yaml"
        config_path.write_text("{{invalid: yaml: [")
        with pytest.raises(yaml.YAMLError):
            load_config(tmp_path, config_path)

    def test_yaml_not_a_mapping(self, tmp_path):
        config_path = tmp_path / "list.yaml"
        config_path.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="must contain a YAML mapping"):
            load_config(tmp_path, config_path)
