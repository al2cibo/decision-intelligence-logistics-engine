"""Rolling window (moving average) forecaster."""

import polars as pl

from forecasting.models.base_forecaster import BaseForecaster


class RollingWindowForecaster(BaseForecaster):
    """Forecasts demand using a rolling mean over the previous ``rolling_window`` periods.

    The rolling mean is computed on the lag-1 shifted demand, so the forecast
    for period t is the mean of demand in periods [t-window, t-1]. This avoids
    look-ahead leakage during evaluation.

    Parameters
    ----------
    target_col : str
        Name of the demand column. Defaults to ``"demand"``.
    rolling_window : int
        Number of past periods to average. Defaults to ``1`` (reduces to naive).
    """

    def __init__(self, target_col: str = "demand", rolling_window: int = 1):
        self.target_col = target_col
        self.rolling_window = rolling_window
        self.forecast_col = f"ma_{rolling_window}_forecast"

    @property
    def name(self) -> str:
        return f"ma_{self.rolling_window}_forecaster"

    def fit(self, df: pl.DataFrame) -> None:
        """Store the last rolling_window training rows for use as context in predict."""
        self._tail = df.tail(self.rolling_window)

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return df with a forecast column equal to the rolling mean of lagged demand."""
        if not hasattr(self, "_tail"):
            raise RuntimeError(
                f"{self.__class__.__name__}.predict() called before fit()"
            )
        combined = pl.concat([self._tail, df])
        return combined.with_columns(
            pl.col(self.target_col)
            .shift(1)
            .rolling_mean(window_size=self.rolling_window)
            .alias(self.forecast_col)
        ).tail(len(df))
