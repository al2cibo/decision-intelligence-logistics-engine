"""Abstract base class for all forecasting models."""

from abc import ABC, abstractmethod

import polars as pl

from forecasting.evaluation.evaluator import Evaluator


class BaseForecaster(ABC):
    """Common interface that every forecasting model must implement.

    Subclasses must define:
    - ``name`` property returning a unique string identifier
    - ``fit(df)`` to train on a DataFrame
    - ``predict(df)`` to produce a forecast column

    The concrete ``evaluate`` method is provided here and delegates metric
    computation to :class:`~forecasting.evaluation.evaluator.Evaluator`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique string identifier for this model (used as registry key)."""

    @abstractmethod
    def fit(self, df: pl.DataFrame) -> None:
        """Train the model on the given DataFrame."""

    @abstractmethod
    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return df augmented with a forecast column."""

    def evaluate(
        self,
        df: pl.DataFrame,
        target_col: str = "demand",
        forecast_col: str | None = None,
    ) -> dict[str, float]:
        """Compute accuracy metrics comparing forecast to target.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame containing both the target and forecast columns.
        target_col : str
            Column of actual demand values. Defaults to ``"demand"``.
        forecast_col : str | None
            Column of forecast values. Defaults to ``self.forecast_col`` when None.

        Returns
        -------
        dict[str, float]
            Keys: ``wape``, ``mae``, ``rmse``, ``mape``, ``mse``.
            All values are NaN when no non-null rows remain after filtering.

        Raises
        ------
        ValueError
            If the forecast column cannot be resolved, required columns are
            missing, or all forecast values are null.
        """
        if forecast_col is None:
            forecast_col = getattr(self, "forecast_col", None)
            if forecast_col is None:
                raise ValueError(
                    "forecast_col must be provided or set as self.forecast_col"
                )

        if target_col not in df.columns:
            raise ValueError(
                f"Required column '{target_col}' is missing from the DataFrame"
            )
        if forecast_col not in df.columns:
            raise ValueError(
                f"Required column '{forecast_col}' is missing from the DataFrame"
            )

        if df[forecast_col].is_null().all():
            raise ValueError(
                "All forecast values are null — no valid forecast values available for evaluation"
            )

        filtered_df = df.drop_nulls(subset=[target_col, forecast_col])

        if filtered_df.height == 0:
            return {k: float("nan") for k in ("wape", "mae", "rmse", "mape", "mse")}

        metrics = Evaluator(filtered_df, target_col).compute_metrics(forecast_col)
        return {k: float(metrics[k]) for k in ("wape", "mae", "rmse", "mape", "mse")}
