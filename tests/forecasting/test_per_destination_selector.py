"""Unit tests for PerDestinationModelSelector and SelectionResult."""

import math

import pytest

from forecasting.evaluation.per_destination_model_selector import (
    PerDestinationModelSelector,
    SelectionResult,
)


class TestSelectionResult:
    """Tests for the SelectionResult frozen dataclass."""

    def test_creation(self):
        result = SelectionResult(
            destination_id="dest_1",
            model_name="naive",
            metrics={"wape": 0.1, "mae": 2.0, "rmse": 3.0, "mape": 0.05, "mse": 9.0},
        )
        assert result.destination_id == "dest_1"
        assert result.model_name == "naive"
        assert result.metrics["wape"] == 0.1

    def test_frozen(self):
        result = SelectionResult(
            destination_id="dest_1",
            model_name="naive",
            metrics={"wape": 0.1},
        )
        with pytest.raises(AttributeError):
            result.model_name = "other"


class TestPerDestinationModelSelectorInit:
    """Tests for PerDestinationModelSelector initialization."""

    def test_default_metric(self):
        selector = PerDestinationModelSelector()
        assert selector.metric == "wape"

    @pytest.mark.parametrize("metric", ["wape", "mae", "rmse", "mape", "mse"])
    def test_valid_metrics(self, metric):
        selector = PerDestinationModelSelector(metric=metric)
        assert selector.metric == metric

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="not recognised"):
            PerDestinationModelSelector(metric="invalid")

    def test_invalid_metric_message_lists_available(self):
        with pytest.raises(ValueError, match="Available: wape, mae, rmse, mape, mse"):
            PerDestinationModelSelector(metric="r2")


class TestPerDestinationModelSelectorSelect:
    """Tests for the select method."""

    def test_selects_minimum_metric(self):
        selector = PerDestinationModelSelector(metric="wape")
        model_metrics = [
            ("model_a", {"wape": 0.5, "mae": 10.0}),
            ("model_b", {"wape": 0.2, "mae": 8.0}),
            ("model_c", {"wape": 0.8, "mae": 5.0}),
        ]
        result = selector.select("dest_1", model_metrics)
        assert result.model_name == "model_b"
        assert result.destination_id == "dest_1"
        assert result.metrics["wape"] == 0.2

    def test_tiebreak_first_in_list_wins(self):
        selector = PerDestinationModelSelector(metric="mae")
        model_metrics = [
            ("model_a", {"mae": 5.0}),
            ("model_b", {"mae": 5.0}),
            ("model_c", {"mae": 5.0}),
        ]
        result = selector.select("dest_1", model_metrics)
        assert result.model_name == "model_a"

    def test_skips_nan_values(self):
        selector = PerDestinationModelSelector(metric="wape")
        model_metrics = [
            ("model_a", {"wape": float("nan")}),
            ("model_b", {"wape": 0.3}),
        ]
        result = selector.select("dest_1", model_metrics)
        assert result.model_name == "model_b"

    def test_all_nan_raises_valueerror(self):
        selector = PerDestinationModelSelector(metric="wape")
        model_metrics = [
            ("model_a", {"wape": float("nan")}),
            ("model_b", {"wape": float("nan")}),
        ]
        with pytest.raises(ValueError, match="No valid"):
            selector.select("dest_1", model_metrics)

    def test_metric_not_in_dict_raises_valueerror(self):
        selector = PerDestinationModelSelector(metric="wape")
        model_metrics = [
            ("model_a", {"mae": 5.0}),
        ]
        with pytest.raises(ValueError, match="not found in metrics"):
            selector.select("dest_1", model_metrics)

    def test_single_model(self):
        selector = PerDestinationModelSelector(metric="rmse")
        model_metrics = [
            ("only_model", {"rmse": 2.5, "mae": 1.0}),
        ]
        result = selector.select("dest_x", model_metrics)
        assert result.model_name == "only_model"
        assert result.metrics["rmse"] == 2.5

    def test_nan_skipped_minimum_still_found(self):
        selector = PerDestinationModelSelector(metric="mse")
        model_metrics = [
            ("model_a", {"mse": float("nan")}),
            ("model_b", {"mse": 10.0}),
            ("model_c", {"mse": 5.0}),
        ]
        result = selector.select("dest_1", model_metrics)
        assert result.model_name == "model_c"
        assert result.metrics["mse"] == 5.0

    def test_returns_full_metrics_dict(self):
        selector = PerDestinationModelSelector(metric="wape")
        metrics = {"wape": 0.1, "mae": 2.0, "rmse": 3.0, "mape": 0.05, "mse": 9.0}
        model_metrics = [("best_model", metrics)]
        result = selector.select("dest_1", model_metrics)
        assert result.metrics == metrics

    def test_empty_model_metrics_raises(self):
        selector = PerDestinationModelSelector(metric="wape")
        with pytest.raises(ValueError, match="No valid"):
            selector.select("dest_1", [])
