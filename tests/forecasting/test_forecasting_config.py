"""Unit tests for PerDestinationConfig and _validate_per_destination_config."""

import pytest

from forecasting.config import (
    PerDestinationConfig,
    _validate_per_destination_config,
    KNOWN_METRICS,
)


class TestPerDestinationConfigValid:
    def test_minimal_valid_config(self):
        config = _validate_per_destination_config({"model_names": ["naive_forecaster"]})

        assert config.model_names == ["naive_forecaster"]
        assert config.train_ratio == 0.8
        assert config.selection_metric == "wape"
        assert config.max_workers == 1
        assert config.minimum_history_length == 2
        assert config.random_seed == 42
        assert config.model_params == {}

    def test_full_valid_config(self):
        config = _validate_per_destination_config(
            {
                "model_names": ["naive_forecaster", "seasonal_forecaster"],
                "train_ratio": 0.7,
                "selection_metric": "mae",
                "max_workers": 4,
                "minimum_history_length": 14,
                "random_seed": 123,
                "model_params": {"seasonal_forecaster": {"lag_value": 7}},
            }
        )

        assert config.model_names == ["naive_forecaster", "seasonal_forecaster"]
        assert config.train_ratio == 0.7
        assert config.selection_metric == "mae"
        assert config.max_workers == 4
        assert config.minimum_history_length == 14
        assert config.random_seed == 123
        assert config.model_params == {"seasonal_forecaster": {"lag_value": 7}}

    def test_twenty_model_names(self):
        names = [f"model_{i}" for i in range(20)]
        config = _validate_per_destination_config({"model_names": names})
        assert len(config.model_names) == 20

    def test_all_known_metrics_accepted(self):
        for metric in KNOWN_METRICS:
            config = _validate_per_destination_config(
                {
                    "model_names": ["naive_forecaster"],
                    "selection_metric": metric,
                }
            )
            assert config.selection_metric == metric

    def test_max_workers_boundary_values(self):
        for workers in [1, 128]:
            config = _validate_per_destination_config(
                {
                    "model_names": ["naive_forecaster"],
                    "max_workers": workers,
                }
            )
            assert config.max_workers == workers


class TestPerDestinationConfigInvalid:
    def test_missing_model_names(self):
        with pytest.raises(ValueError, match="model_names is required"):
            _validate_per_destination_config({})

    def test_empty_model_names(self):
        with pytest.raises(ValueError, match="must contain 1-20 model names"):
            _validate_per_destination_config({"model_names": []})

    def test_too_many_model_names(self):
        names = [f"model_{i}" for i in range(21)]
        with pytest.raises(ValueError, match="must contain 1-20 model names"):
            _validate_per_destination_config({"model_names": names})

    def test_non_string_model_name(self):
        with pytest.raises(ValueError, match="must be a non-empty string"):
            _validate_per_destination_config({"model_names": [123]})

    def test_empty_string_model_name(self):
        with pytest.raises(ValueError, match="must be a non-empty string"):
            _validate_per_destination_config({"model_names": [""]})

    def test_whitespace_only_model_name(self):
        with pytest.raises(ValueError, match="must be a non-empty string"):
            _validate_per_destination_config({"model_names": ["   "]})

    def test_train_ratio_zero(self):
        with pytest.raises(ValueError, match="train_ratio must be in \\(0, 1\\)"):
            _validate_per_destination_config({"model_names": ["m"], "train_ratio": 0.0})

    def test_train_ratio_one(self):
        with pytest.raises(ValueError, match="train_ratio must be in \\(0, 1\\)"):
            _validate_per_destination_config({"model_names": ["m"], "train_ratio": 1.0})

    def test_train_ratio_negative(self):
        with pytest.raises(ValueError, match="train_ratio must be in \\(0, 1\\)"):
            _validate_per_destination_config(
                {"model_names": ["m"], "train_ratio": -0.5}
            )

    def test_train_ratio_greater_than_one(self):
        with pytest.raises(ValueError, match="train_ratio must be in \\(0, 1\\)"):
            _validate_per_destination_config({"model_names": ["m"], "train_ratio": 1.5})

    def test_invalid_selection_metric(self):
        with pytest.raises(ValueError, match="not recognised"):
            _validate_per_destination_config(
                {
                    "model_names": ["m"],
                    "selection_metric": "r2",
                }
            )

    def test_max_workers_zero(self):
        with pytest.raises(ValueError, match="must be between 1 and 128"):
            _validate_per_destination_config({"model_names": ["m"], "max_workers": 0})

    def test_max_workers_too_large(self):
        with pytest.raises(ValueError, match="must be between 1 and 128"):
            _validate_per_destination_config({"model_names": ["m"], "max_workers": 129})

    def test_minimum_history_length_zero(self):
        with pytest.raises(ValueError, match="minimum_history_length must be > 0"):
            _validate_per_destination_config(
                {
                    "model_names": ["m"],
                    "minimum_history_length": 0,
                }
            )

    def test_minimum_history_length_negative(self):
        with pytest.raises(ValueError, match="minimum_history_length must be > 0"):
            _validate_per_destination_config(
                {
                    "model_names": ["m"],
                    "minimum_history_length": -1,
                }
            )

    def test_model_params_key_not_in_model_names(self):
        with pytest.raises(ValueError, match="not in model_names"):
            _validate_per_destination_config(
                {
                    "model_names": ["naive_forecaster"],
                    "model_params": {"unknown_model": {"param": 1}},
                }
            )

    def test_model_params_value_not_dict(self):
        with pytest.raises(ValueError, match="must be a dict"):
            _validate_per_destination_config(
                {
                    "model_names": ["naive_forecaster"],
                    "model_params": {"naive_forecaster": "not_a_dict"},
                }
            )
