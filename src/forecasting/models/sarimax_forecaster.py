"""SARIMAX forecaster wrapping statsmodels SARIMAX.

Supports configurable ARIMA order (p, d, q) and optional seasonal order (P, D, Q, s).
When ``seasonal_order`` is None the model reduces to a plain ARIMA.
"""

import polars as pl
from statsmodels.tsa.statespace.sarimax import SARIMAX

from forecasting.models.base_forecaster import BaseForecaster


class SARIMAXForecaster(BaseForecaster):
    """ARIMA/SARIMA forecaster built on statsmodels SARIMAX.

    Fits on the training series and produces genuine out-of-sample forecasts
    for whatever window is passed to ``predict`` — the pipeline always calls
    ``predict`` on data chronologically after the training window, so the
    forecast is always ``forecast(steps=len(df))`` starting right after the
    end of training.

    Parameters
    ----------
    target_col : str
        Name of the demand column. Defaults to ``"demand"``.
    order : tuple[int, int, int]
        ARIMA order ``(p, d, q)``. Defaults to ``(1, 1, 0)``.
    seasonal_order : tuple[int, int, int, int] | None
        Seasonal ARIMA order ``(P, D, Q, s)``. When ``None`` the seasonal
        component is suppressed. Defaults to ``None``.
    """

    def __init__(
        self,
        target_col: str = "demand",
        order: tuple[int, int, int] = (1, 1, 0),
        seasonal_order: tuple[int, int, int, int] | None = None,
    ):
        self.target_col = target_col
        self.order = order
        self.seasonal_order = seasonal_order
        self.forecast_col = "sarimax_forecast"
        self._fitted_result = None

    @property
    def name(self) -> str:
        return "sarimax_forecaster"

    def fit(self, df: pl.DataFrame) -> None:
        if self.target_col not in df.columns:
            raise ValueError(f"Column '{self.target_col}' not found in DataFrame")

        y = df[self.target_col].to_numpy().astype(float)

        model = SARIMAX(
            y,
            order=self.order,
            seasonal_order=(
                self.seasonal_order if self.seasonal_order is not None else (0, 0, 0, 0)
            ),
        )
        self._fitted_result = model.fit(disp=False)

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
