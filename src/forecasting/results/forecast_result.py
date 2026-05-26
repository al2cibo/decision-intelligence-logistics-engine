"""Frozen dataclasses for forecast results and time periods."""

from dataclasses import dataclass
from datetime import date
from typing import Any

import polars as pl


@dataclass(frozen=True)
class TimePeriod:
    """An immutable time period with start and end dates.

    Parameters
    ----------
    start_date : date
        The start of the period (inclusive).
    end_date : date
        The end of the period (inclusive).

    Raises
    ------
    ValueError
        If start_date is not strictly before end_date.
    """

    start_date: date
    end_date: date

    def __post_init__(self) -> None:
        if self.start_date >= self.end_date:
            raise ValueError(
                f"start_date ({self.start_date}) must be before end_date ({self.end_date})"
            )


@dataclass(frozen=True)
class ForecastResult:
    """Immutable result for a single destination-model combination.

    Parameters
    ----------
    destination_id : str
        Non-empty identifier for the destination.
    model_name : str
        Non-empty name of the model that produced this result.
    train_period : TimePeriod
        The time period used for training.
    validation_period : TimePeriod
        The time period used for validation/testing.
    forecast_values : pl.DataFrame
        DataFrame with exactly columns [date, forecast] where date is pl.Date
        and forecast is pl.Float64.
    metrics : dict[str, float]
        Evaluation metrics. Must contain at least one key from
        {"wape", "mae", "mse", "rmse", "mape"}.
    execution_time_seconds : float
        Non-negative wall-clock time for model execution.
    model_parameters : dict[str, Any]
        Parameters used to configure the model.

    Raises
    ------
    ValueError
        If any field constraint is violated.
    """

    destination_id: str
    model_name: str
    train_period: TimePeriod
    validation_period: TimePeriod
    forecast_values: pl.DataFrame
    metrics: dict[str, float]
    execution_time_seconds: float
    model_parameters: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.destination_id:
            raise ValueError("destination_id must be non-empty")
        if not self.model_name:
            raise ValueError("model_name must be non-empty")
        if self.execution_time_seconds < 0:
            raise ValueError("execution_time_seconds must be non-negative")

        # Validate forecast_values schema
        expected_cols = {"date", "forecast"}
        if set(self.forecast_values.columns) != expected_cols:
            raise ValueError(
                f"forecast_values must have columns {expected_cols}, "
                f"got {set(self.forecast_values.columns)}"
            )

        # Validate column types
        schema = self.forecast_values.schema
        if schema["date"] != pl.Date:
            raise ValueError(
                f"forecast_values 'date' column must be pl.Date, got {schema['date']}"
            )
        if schema["forecast"] != pl.Float64:
            raise ValueError(
                f"forecast_values 'forecast' column must be pl.Float64, "
                f"got {schema['forecast']}"
            )

        # Validate metrics contains at least one known key
        valid_metric_keys = {"wape", "mae", "mse", "rmse", "mape"}
        if not valid_metric_keys.intersection(self.metrics.keys()):
            raise ValueError(
                f"metrics must contain at least one key from {valid_metric_keys}, "
                f"got keys {set(self.metrics.keys())}"
            )
