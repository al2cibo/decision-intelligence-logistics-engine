"""Seasonal lag forecaster: repeats demand from N periods ago."""

import polars as pl

from forecasting.models.base_forecaster import BaseForecaster


class SeasonalForecaster(BaseForecaster):
    """Forecasts demand using the value from ``lag_value`` periods ago.

    Captures weekly seasonality when ``lag_value=7`` (default): Monday's
    forecast is last Monday's demand, and so on.

    Parameters
    ----------
    target_col : str
        Name of the demand column. Defaults to ``"demand"``.
    lag_value : int
        Number of periods to look back. Defaults to ``7`` (one week).
    """

    def __init__(self, target_col: str = "demand", lag_value: int = 7):
        self.target_col = target_col
        self.lag_value = lag_value
        self.forecast_col = f"seasonal_forecast_lag_{lag_value}"

    @property
    def name(self) -> str:
        return f"seasonal_forecaster_lag_{self.lag_value}"

    def fit(self, df: pl.DataFrame) -> None:
        """Store the last lag_value training rows for use as context in predict."""
        self._tail = df.tail(self.lag_value)

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return df with a forecast column equal to demand shifted by lag_value.

        The last lag_value training rows (stored in fit) are prepended so that
        the first lag_value rows of the test window get training-derived values
        instead of nulls.
        """
        if not hasattr(self, "_tail"):
            raise RuntimeError(
                f"{self.__class__.__name__}.predict() called before fit()"
            )
        combined = pl.concat([self._tail, df])
        return combined.with_columns(
            pl.col(self.target_col).shift(self.lag_value).alias(self.forecast_col)
        ).tail(len(df))
