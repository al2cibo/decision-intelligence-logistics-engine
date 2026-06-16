"""ETS (Exponential Smoothing) forecaster wrapping statsmodels Holt-Winters.

Supports configurable trend and seasonal components. When neither is set the
model reduces to simple exponential smoothing.
"""

import numpy as np
import polars as pl
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from forecasting.models.base_forecaster import BaseForecaster


class ETSForecaster(BaseForecaster):
    """Holt-Winters exponential smoothing forecaster.

    Fits on the training series and produces in-sample fitted values for the
    test window. If the test window extends beyond the training length, the
    model appends out-of-sample forecasts.

    Parameters
    ----------
    target_col : str
        Name of the demand column. Defaults to ``"demand"``.
    trend : str | None
        Trend component: ``"add"``, ``"mul"``, or ``None``. Defaults to ``"add"``.
    seasonal : str | None
        Seasonal component: ``"add"``, ``"mul"``, or ``None``. Defaults to ``None``.
    seasonal_periods : int | None
        Number of periods in a seasonal cycle. Required when ``seasonal`` is set.
        Defaults to ``None``.
    """

    def __init__(
        self,
        target_col: str = "demand",
        trend: str | None = "add",
        seasonal: str | None = None,
        seasonal_periods: int | None = None,
    ):
        self.target_col = target_col
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.forecast_col = "ets_forecast"
        self._fitted_result = None
        self._train_length = 0

    @property
    def name(self) -> str:
        return "ets_forecaster"

    def fit(self, df: pl.DataFrame) -> None:
        if self.target_col not in df.columns:
            raise ValueError(f"Column '{self.target_col}' not found in DataFrame")

        y = df[self.target_col].to_numpy().astype(float)
        self._train_length = len(y)

        model = ExponentialSmoothing(
            y,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
        )
        self._fitted_result = model.fit()

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        if self._fitted_result is None:
            raise RuntimeError("fit() must be called before predict()")
        if self.target_col not in df.columns:
            raise ValueError(f"Column '{self.target_col}' not found in DataFrame")

        n = len(df)
        fitted_vals = self._fitted_result.fittedvalues

        if n <= self._train_length:
            forecast_values = fitted_vals[:n]
        else:
            oos_steps = n - self._train_length
            oos_forecast = self._fitted_result.forecast(steps=oos_steps)
            forecast_values = np.concatenate([fitted_vals, oos_forecast])

        return df.with_columns(
            pl.Series(name=self.forecast_col, values=forecast_values)
        )
