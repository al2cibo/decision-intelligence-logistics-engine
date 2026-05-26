"""Unit tests for PerDestinationConfig and load_config extensions."""

import pytest
import yaml
from pathlib import Path

from utils.config import (
    PerDestinationConfig,
    Config,
    load_config,
    _validate_per_destination_config,
    KNOWN_METRICS,
)


# --- Fixtures ---


@pytest.fixture
def tmp_config_dir(tmp_path):
    """Create a temporary directory with a data subdirectory."""
    (tmp_path / "data" / "synthetic2").mkdir(parents=True)
    return tmp_path


def _write_yaml(path: Path, content: dict) -> Path:
    """Write a YAML config file and return its path."""
    config_path = path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(content, f)
    return config_path


# --- Tests for valid configuration ---


class TestPerDestinationConfigValid:
    def test_minimal_valid_config(self, tmp_config_dir):
        """Minimal valid config with just model_names and data section."""
        config_data = {
            "data": {"input_path": "data/synthetic2"},
            "per_destination_forecasting": {
                "model_names": ["naive_forecaster"],
            },
        }
        config_path = _write_yaml(tmp_config_dir, config_data)
        config = load_config(tmp_config_dir, config_path)

        assert config.per_destination_forecasting is not None
        pdc = config.per_destination_forecasting
        assert pdc.model_names == ["naive_forecaster"]
        assert pdc.train_ratio == 0.8
        assert pdc.selection_metric == "wape"
        assert pdc.max_workers == 1
        assert pdc.minimum_history_length == 2
        assert pdc.random_seed == 42
        assert pdc.model_params == {}

    def test_full_valid_config(self, tmp_config_dir):
        """Full config with all fields specified."""
        config_data = {
            "data": {"input_path": "data/synthetic2"},
            "per_destination_forecasting": {
                "model_names": ["naive_forecaster", "seasonal_forecaster"],
                "train_ratio": 0.7,
                "selection_metric": "mae",
                "max_workers": 4,
                "minimum_history_length": 14,
                "random_seed": 123,
                "model_params": {
                    "seasonal_forecaster": {"lag_value": 7},
                },
            },
        }
        config_path = _write_yaml(tmp_config_dir, config_data)
        config = load_config(tmp_config_dir, config_path)

        pdc = config.per_destination_forecasting
        assert pdc.model_names == ["naive_forecaster", "seasonal_forecaster"]
        assert pdc.train_ratio == 0.7
        assert pdc.selection_metric == "mae"
        assert pdc.max_workers == 4
        assert pdc.minimum_history_length == 14
        assert pdc.random_seed == 123
        assert pdc.model_params == {"seasonal_forecaster": {"lag_value": 7}}

    def test_config_without_per_destination_section(self, tmp_config_dir):
        """Config without per_destination_forecasting section returns None."""
        config_data = {
            "data": {"input_path": "data/synthetic2"},
            "forecasting": {"metric": "wape", "train_ratio": 0.8},
        }
        config_path = _write_yaml(tmp_config_dir, config_data)
        config = load_config(tmp_config_dir, config_path)

        assert config.per_destination_forecasting is None

    def test_twenty_model_names(self, tmp_config_dir):
        """Maximum of 20 model names is accepted."""
        names = [f"model_{i}" for i in range(20)]
        config_data = {
            "data": {"input_path": "data/synthetic2"},
            "per_destination_forecasting": {
                "model_names": names,
            },
        }
        config_path = _write_yaml(tmp_config_dir, config_data)
        config = load_config(tmp_config_dir, config_path)
        assert len(config.per_destination_forecasting.model_names) == 20

    def test_all_known_metrics_accepted(self, tmp_config_dir):
        """All known metrics are accepted as selection_metric."""
        for metric in KNOWN_METRICS:
            config_data = {
                "data": {"input_path": "data/synthetic2"},
                "per_destination_forecasting": {
                    "model_names": ["naive_forecaster"],
                    "selection_metric": metric,
                },
            }
            config_path = _write_yaml(tmp_config_dir, config_data)
            config = load_config(tmp_config_dir, config_path)
            assert config.per_destination_forecasting.selection_metric == metric

    def test_max_workers_boundary_values(self, tmp_config_dir):
        """max_workers at boundaries 1 and 128 are accepted."""
        for workers in [1, 128]:
            config_data = {
                "data": {"input_path": "data/synthetic2"},
                "per_destination_forecasting": {
                    "model_names": ["naive_forecaster"],
                    "max_workers": workers,
                },
            }
            config_path = _write_yaml(tmp_config_dir, config_data)
            config = load_config(tmp_config_dir, config_path)
            assert config.per_destination_forecasting.max_workers == workers


# --- Tests for invalid configuration ---


class TestPerDestinationConfigInvalid:
    def test_missing_model_names(self):
        """Missing model_names raises ValueError."""
        with pytest.raises(ValueError, match="model_names is required"):
            _validate_per_destination_config({})

    def test_empty_model_names(self):
        """Empty model_names list raises ValueError."""
        with pytest.raises(ValueError, match="must contain 1-20 model names"):
            _validate_per_destination_config({"model_names": []})

    def test_too_many_model_names(self):
        """More than 20 model names raises ValueError."""
        names = [f"model_{i}" for i in range(21)]
        with pytest.raises(ValueError, match="must contain 1-20 model names"):
            _validate_per_destination_config({"model_names": names})

    def test_non_string_model_name(self):
        """Non-string model name raises ValueError."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            _validate_per_destination_config({"model_names": [123]})

    def test_empty_string_model_name(self):
        """Empty string model name raises ValueError."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            _validate_per_destination_config({"model_names": [""]})

    def test_whitespace_only_model_name(self):
        """Whitespace-only model name raises ValueError."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            _validate_per_destination_config({"model_names": ["   "]})

    def test_train_ratio_zero(self):
        """train_ratio of 0 raises ValueError."""
        with pytest.raises(ValueError, match="train_ratio must be in \\(0, 1\\)"):
            _validate_per_destination_config(
                {"model_names": ["m"], "train_ratio": 0.0}
            )

    def test_train_ratio_one(self):
        """train_ratio of 1 raises ValueError."""
        with pytest.raises(ValueError, match="train_ratio must be in \\(0, 1\\)"):
            _validate_per_destination_config(
                {"model_names": ["m"], "train_ratio": 1.0}
            )

    def test_train_ratio_negative(self):
        """Negative train_ratio raises ValueError."""
        with pytest.raises(ValueError, match="train_ratio must be in \\(0, 1\\)"):
            _validate_per_destination_config(
                {"model_names": ["m"], "train_ratio": -0.5}
            )

    def test_train_ratio_greater_than_one(self):
        """train_ratio > 1 raises ValueError."""
        with pytest.raises(ValueError, match="train_ratio must be in \\(0, 1\\)"):
            _validate_per_destination_config(
                {"model_names": ["m"], "train_ratio": 1.5}
            )

    def test_invalid_selection_metric(self):
        """Unknown selection_metric raises ValueError."""
        with pytest.raises(ValueError, match="not recognised"):
            _validate_per_destination_config(
                {"model_names": ["m"], "selection_metric": "r2"}
            )

    def test_max_workers_zero(self):
        """max_workers of 0 raises ValueError."""
        with pytest.raises(ValueError, match="must be between 1 and 128"):
            _validate_per_destination_config(
                {"model_names": ["m"], "max_workers": 0}
            )

    def test_max_workers_too_large(self):
        """max_workers > 128 raises ValueError."""
        with pytest.raises(ValueError, match="must be between 1 and 128"):
            _validate_per_destination_config(
                {"model_names": ["m"], "max_workers": 129}
            )

    def test_minimum_history_length_zero(self):
        """minimum_history_length of 0 raises ValueError."""
        with pytest.raises(ValueError, match="minimum_history_length must be > 0"):
            _validate_per_destination_config(
                {"model_names": ["m"], "minimum_history_length": 0}
            )

    def test_minimum_history_length_negative(self):
        """Negative minimum_history_length raises ValueError."""
        with pytest.raises(ValueError, match="minimum_history_length must be > 0"):
            _validate_per_destination_config(
                {"model_names": ["m"], "minimum_history_length": -1}
            )

    def test_model_params_key_not_in_model_names(self):
        """model_params key not in model_names raises ValueError."""
        with pytest.raises(ValueError, match="not in model_names"):
            _validate_per_destination_config(
                {
                    "model_names": ["naive_forecaster"],
                    "model_params": {"unknown_model": {"param": 1}},
                }
            )

    def test_model_params_value_not_dict(self):
        """model_params value that is not a dict raises ValueError."""
        with pytest.raises(ValueError, match="must be a dict"):
            _validate_per_destination_config(
                {
                    "model_names": ["naive_forecaster"],
                    "model_params": {"naive_forecaster": "not_a_dict"},
                }
            )


# --- Tests for file-level error handling ---


class TestLoadConfigFileErrors:
    def test_missing_config_file(self, tmp_path):
        """Missing config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_config(tmp_path, tmp_path / "nonexistent.yaml")

    def test_malformed_yaml(self, tmp_path):
        """Malformed YAML raises yaml.YAMLError."""
        config_path = tmp_path / "bad.yaml"
        config_path.write_text("{{invalid: yaml: [")
        with pytest.raises(yaml.YAMLError):
            load_config(tmp_path, config_path)

    def test_yaml_not_a_mapping(self, tmp_path):
        """YAML that is not a mapping raises ValueError."""
        config_path = tmp_path / "list.yaml"
        config_path.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="must contain a YAML mapping"):
            load_config(tmp_path, config_path)
