"""Naive lag-1 forecaster: uses yesterday's demand as today's forecast."""

import polars as pl

from forecasting.models.base_forecaster import BaseForecaster


class NaiveForecaster(BaseForecaster):
    """Forecasts demand using the previous period's observed value (lag-1 shift).

    This is the simplest possible forecasting strategy and serves as a lower-bound
    baseline. A model that cannot beat it on WAPE has no practical value.

    Parameters
    ----------
    target_col : str
        Name of the demand column. Defaults to ``"demand"``.
    """

    def __init__(self, target_col: str = "demand"):
        self.target_col = target_col
        self.forecast_col = "naive_forecast"

    @property
    def name(self) -> str:
        return "naive_forecaster"

    def fit(self, df: pl.DataFrame) -> None:
        """No-op: naive forecasting requires no training."""

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return df with a forecast column equal to the lagged demand (shift by 1)."""
        if self.target_col not in df.columns:
            raise ValueError(f"Column '{self.target_col}' not found in DataFrame")
        return df.with_columns(
            pl.col(self.target_col).shift(1).alias(self.forecast_col)
        )
