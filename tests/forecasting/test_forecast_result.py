"""Unit tests for ForecastResult and TimePeriod frozen dataclasses."""

from datetime import date

import polars as pl
import pytest

from forecasting.results.forecast_result import ForecastResult, TimePeriod


class TestTimePeriod:
    """Tests for TimePeriod dataclass."""

    def test_valid_time_period(self):
        tp = TimePeriod(start_date=date(2024, 1, 1), end_date=date(2024, 6, 30))
        assert tp.start_date == date(2024, 1, 1)
        assert tp.end_date == date(2024, 6, 30)

    def test_start_equals_end_raises(self):
        with pytest.raises(ValueError, match="must be before"):
            TimePeriod(start_date=date(2024, 1, 1), end_date=date(2024, 1, 1))

    def test_start_after_end_raises(self):
        with pytest.raises(ValueError, match="must be before"):
            TimePeriod(start_date=date(2024, 6, 30), end_date=date(2024, 1, 1))

    def test_frozen_rejects_assignment(self):
        tp = TimePeriod(start_date=date(2024, 1, 1), end_date=date(2024, 6, 30))
        with pytest.raises(AttributeError):
            tp.start_date = date(2024, 2, 1)  # type: ignore[misc]


class TestForecastResult:
    """Tests for ForecastResult dataclass."""

    @pytest.fixture
    def valid_forecast_values(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "forecast": [10.0, 20.0],
            },
            schema={"date": pl.Date, "forecast": pl.Float64},
        )

    @pytest.fixture
    def valid_result(self, valid_forecast_values: pl.DataFrame) -> ForecastResult:
        return ForecastResult(
            destination_id="dest_001",
            model_name="naive_forecaster",
            train_period=TimePeriod(date(2023, 1, 1), date(2023, 12, 31)),
            validation_period=TimePeriod(date(2024, 1, 1), date(2024, 1, 2)),
            forecast_values=valid_forecast_values,
            metrics={"wape": 0.1, "mae": 2.0, "mse": 5.0, "rmse": 2.24, "mape": 0.15},
            execution_time_seconds=1.5,
            model_parameters={"lag": 7},
        )

    def test_valid_result_creation(self, valid_result: ForecastResult):
        assert valid_result.destination_id == "dest_001"
        assert valid_result.model_name == "naive_forecaster"
        assert valid_result.execution_time_seconds == 1.5

    def test_frozen_rejects_assignment(self, valid_result: ForecastResult):
        with pytest.raises(AttributeError):
            valid_result.destination_id = "other"  # type: ignore[misc]

    def test_empty_destination_id_raises(self, valid_forecast_values: pl.DataFrame):
        with pytest.raises(ValueError, match="destination_id must be non-empty"):
            ForecastResult(
                destination_id="",
                model_name="naive_forecaster",
                train_period=TimePeriod(date(2023, 1, 1), date(2023, 12, 31)),
                validation_period=TimePeriod(date(2024, 1, 1), date(2024, 1, 2)),
                forecast_values=valid_forecast_values,
                metrics={"wape": 0.1},
                execution_time_seconds=1.0,
                model_parameters={},
            )

    def test_empty_model_name_raises(self, valid_forecast_values: pl.DataFrame):
        with pytest.raises(ValueError, match="model_name must be non-empty"):
            ForecastResult(
                destination_id="dest_001",
                model_name="",
                train_period=TimePeriod(date(2023, 1, 1), date(2023, 12, 31)),
                validation_period=TimePeriod(date(2024, 1, 1), date(2024, 1, 2)),
                forecast_values=valid_forecast_values,
                metrics={"wape": 0.1},
                execution_time_seconds=1.0,
                model_parameters={},
            )

    def test_negative_execution_time_raises(self, valid_forecast_values: pl.DataFrame):
        with pytest.raises(
            ValueError, match="execution_time_seconds must be non-negative"
        ):
            ForecastResult(
                destination_id="dest_001",
                model_name="naive_forecaster",
                train_period=TimePeriod(date(2023, 1, 1), date(2023, 12, 31)),
                validation_period=TimePeriod(date(2024, 1, 1), date(2024, 1, 2)),
                forecast_values=valid_forecast_values,
                metrics={"wape": 0.1},
                execution_time_seconds=-0.5,
                model_parameters={},
            )

    def test_zero_execution_time_is_valid(self, valid_forecast_values: pl.DataFrame):
        result = ForecastResult(
            destination_id="dest_001",
            model_name="naive_forecaster",
            train_period=TimePeriod(date(2023, 1, 1), date(2023, 12, 31)),
            validation_period=TimePeriod(date(2024, 1, 1), date(2024, 1, 2)),
            forecast_values=valid_forecast_values,
            metrics={"wape": 0.1},
            execution_time_seconds=0.0,
            model_parameters={},
        )
        assert result.execution_time_seconds == 0.0

    def test_wrong_columns_raises(self):
        bad_df = pl.DataFrame(
            {"date": [date(2024, 1, 1)], "prediction": [10.0]},
            schema={"date": pl.Date, "prediction": pl.Float64},
        )
        with pytest.raises(ValueError, match="forecast_values must have columns"):
            ForecastResult(
                destination_id="dest_001",
                model_name="naive_forecaster",
                train_period=TimePeriod(date(2023, 1, 1), date(2023, 12, 31)),
                validation_period=TimePeriod(date(2024, 1, 1), date(2024, 1, 2)),
                forecast_values=bad_df,
                metrics={"wape": 0.1},
                execution_time_seconds=1.0,
                model_parameters={},
            )

    def test_wrong_date_type_raises(self):
        bad_df = pl.DataFrame(
            {"date": ["2024-01-01"], "forecast": [10.0]},
            schema={"date": pl.Utf8, "forecast": pl.Float64},
        )
        with pytest.raises(ValueError, match="'date' column must be pl.Date"):
            ForecastResult(
                destination_id="dest_001",
                model_name="naive_forecaster",
                train_period=TimePeriod(date(2023, 1, 1), date(2023, 12, 31)),
                validation_period=TimePeriod(date(2024, 1, 1), date(2024, 1, 2)),
                forecast_values=bad_df,
                metrics={"wape": 0.1},
                execution_time_seconds=1.0,
                model_parameters={},
            )

    def test_wrong_forecast_type_raises(self):
        bad_df = pl.DataFrame(
            {"date": [date(2024, 1, 1)], "forecast": [10]},
            schema={"date": pl.Date, "forecast": pl.Int64},
        )
        with pytest.raises(ValueError, match="'forecast' column must be pl.Float64"):
            ForecastResult(
                destination_id="dest_001",
                model_name="naive_forecaster",
                train_period=TimePeriod(date(2023, 1, 1), date(2023, 12, 31)),
                validation_period=TimePeriod(date(2024, 1, 1), date(2024, 1, 2)),
                forecast_values=bad_df,
                metrics={"wape": 0.1},
                execution_time_seconds=1.0,
                model_parameters={},
            )

    def test_metrics_without_known_keys_raises(
        self, valid_forecast_values: pl.DataFrame
    ):
        with pytest.raises(ValueError, match="metrics must contain at least one key"):
            ForecastResult(
                destination_id="dest_001",
                model_name="naive_forecaster",
                train_period=TimePeriod(date(2023, 1, 1), date(2023, 12, 31)),
                validation_period=TimePeriod(date(2024, 1, 1), date(2024, 1, 2)),
                forecast_values=valid_forecast_values,
                metrics={"unknown_metric": 0.5},
                execution_time_seconds=1.0,
                model_parameters={},
            )

    def test_metrics_with_subset_of_known_keys_is_valid(
        self, valid_forecast_values: pl.DataFrame
    ):
        result = ForecastResult(
            destination_id="dest_001",
            model_name="naive_forecaster",
            train_period=TimePeriod(date(2023, 1, 1), date(2023, 12, 31)),
            validation_period=TimePeriod(date(2024, 1, 1), date(2024, 1, 2)),
            forecast_values=valid_forecast_values,
            metrics={"mae": 2.0, "rmse": 3.0},
            execution_time_seconds=1.0,
            model_parameters={},
        )
        assert result.metrics == {"mae": 2.0, "rmse": 3.0}
