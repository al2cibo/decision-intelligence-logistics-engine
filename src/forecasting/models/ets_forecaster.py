"""ETS (Exponential Smoothing) forecaster wrapping statsmodels Holt-Winters.

Supports configurable trend and seasonal components. When neither is set the
model reduces to simple exponential smoothing.
"""

import polars as pl
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from forecasting.models.base_forecaster import BaseForecaster


class ETSForecaster(BaseForecaster):
    """Holt-Winters exponential smoothing forecaster.

    Fits on the training series and produces genuine out-of-sample forecasts
    for whatever window is passed to ``predict`` — the pipeline always calls
    ``predict`` on data chronologically after the training window, so the
    forecast is always ``forecast(steps=len(df))`` starting right after the
    end of training.

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

    @property
    def name(self) -> str:
        return "ets_forecaster"

    def fit(self, df: pl.DataFrame) -> None:
        if self.target_col not in df.columns:
            raise ValueError(f"Column '{self.target_col}' not found in DataFrame")

        y = df[self.target_col].to_numpy().astype(float)

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
        forecast_values = self._fitted_result.forecast(steps=n)

        return df.with_columns(
            pl.Series(name=self.forecast_col, values=forecast_values)
        )
