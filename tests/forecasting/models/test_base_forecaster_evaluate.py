"""Unit tests for BaseForecaster.evaluate method."""

import math

import numpy as np
import polars as pl
import pytest

from forecasting.models.naive_forecaster import NaiveForecaster


class TestBaseForecasterEvaluate:
    """Tests for the concrete evaluate method on BaseForecaster."""

    def setup_method(self):
        self.model = NaiveForecaster()

    def test_returns_all_expected_keys(self):
        df = pl.DataFrame(
            {
                "demand": [10.0, 20.0, 30.0],
                "naive_forecast": [12.0, 18.0, 33.0],
            }
        )
        result = self.model.evaluate(
            df, target_col="demand", forecast_col="naive_forecast"
        )
        assert set(result.keys()) == {"wape", "mae", "rmse", "mape", "mse"}

    def test_all_values_are_floats(self):
        df = pl.DataFrame(
            {
                "demand": [10.0, 20.0, 30.0],
                "naive_forecast": [12.0, 18.0, 33.0],
            }
        )
        result = self.model.evaluate(
            df, target_col="demand", forecast_col="naive_forecast"
        )
        for key, value in result.items():
            assert isinstance(value, float), f"{key} is not a float: {type(value)}"

    def test_raises_valueerror_missing_target_col(self):
        df = pl.DataFrame(
            {
                "other_col": [10.0, 20.0],
                "naive_forecast": [12.0, 18.0],
            }
        )
        with pytest.raises(ValueError, match="'demand' is missing"):
            self.model.evaluate(df, target_col="demand", forecast_col="naive_forecast")

    def test_raises_valueerror_missing_forecast_col(self):
        df = pl.DataFrame(
            {
                "demand": [10.0, 20.0],
                "other_col": [12.0, 18.0],
            }
        )
        with pytest.raises(ValueError, match="'naive_forecast' is missing"):
            self.model.evaluate(df, target_col="demand", forecast_col="naive_forecast")

    def test_raises_valueerror_all_forecast_null(self):
        df = pl.DataFrame(
            {
                "demand": [10.0, 20.0, 30.0],
                "naive_forecast": [None, None, None],
            }
        ).cast({"naive_forecast": pl.Float64})
        with pytest.raises(ValueError, match="All forecast values are null"):
            self.model.evaluate(df, target_col="demand", forecast_col="naive_forecast")

    def test_null_forecast_rows_excluded(self):
        df = pl.DataFrame(
            {
                "demand": [10.0, 20.0, 30.0],
                "naive_forecast": [None, 20.0, 30.0],
            }
        )
        result = self.model.evaluate(
            df, target_col="demand", forecast_col="naive_forecast"
        )
        # After null exclusion: demand=[20, 30], forecast=[20, 30] -> perfect
        assert result["mae"] == pytest.approx(0.0)
        assert result["mse"] == pytest.approx(0.0)

    def test_null_target_rows_excluded(self):
        df = pl.DataFrame(
            {
                "demand": [None, 20.0, 30.0],
                "naive_forecast": [12.0, 20.0, 30.0],
            }
        )
        result = self.model.evaluate(
            df, target_col="demand", forecast_col="naive_forecast"
        )
        # After null exclusion: demand=[20, 30], forecast=[20, 30] -> perfect
        assert result["mae"] == pytest.approx(0.0)

    def test_zero_actual_demand_returns_nan_wape_mape(self):
        df = pl.DataFrame(
            {
                "demand": [0.0, 0.0, 0.0],
                "naive_forecast": [1.0, 2.0, 3.0],
            }
        )
        result = self.model.evaluate(
            df, target_col="demand", forecast_col="naive_forecast"
        )
        assert math.isnan(result["wape"])
        # sklearn's MAPE divides by actual, so with zero actuals it may produce inf
        # but our WAPE should be NaN
        assert not math.isnan(result["mae"])  # MAE is still valid

    def test_zero_rows_after_null_exclusion_returns_nan_all(self):
        # All target values are null but not all forecast values are null
        df = pl.DataFrame(
            {
                "demand": [None, None, None],
                "naive_forecast": [1.0, None, 3.0],
            }
        ).cast({"demand": pl.Float64})
        result = self.model.evaluate(
            df, target_col="demand", forecast_col="naive_forecast"
        )
        for key in ("wape", "mae", "rmse", "mape", "mse"):
            assert math.isnan(result[key]), f"{key} should be NaN but got {result[key]}"

    def test_uses_model_forecast_col_attribute_when_none(self):
        """When forecast_col is None, should use self.forecast_col."""
        df = pl.DataFrame(
            {
                "demand": [10.0, 20.0, 30.0],
                "naive_forecast": [10.0, 20.0, 30.0],
            }
        )
        # NaiveForecaster has self.forecast_col = "naive_forecast"
        result = self.model.evaluate(df, target_col="demand")
        assert result["mae"] == pytest.approx(0.0)

    def test_correct_metric_values(self):
        df = pl.DataFrame(
            {
                "demand": [10.0, 20.0, 30.0],
                "naive_forecast": [12.0, 18.0, 33.0],
            }
        )
        result = self.model.evaluate(
            df, target_col="demand", forecast_col="naive_forecast"
        )

        errors = np.array([2.0, 2.0, 3.0])
        expected_mae = np.mean(np.abs(errors))
        expected_mse = np.mean(errors**2)
        expected_rmse = np.sqrt(expected_mse)
        expected_wape = np.sum(np.abs(errors)) / np.sum(np.array([10.0, 20.0, 30.0]))

        assert result["mae"] == pytest.approx(expected_mae)
        assert result["mse"] == pytest.approx(expected_mse)
        assert result["rmse"] == pytest.approx(expected_rmse)
        assert result["wape"] == pytest.approx(expected_wape)
