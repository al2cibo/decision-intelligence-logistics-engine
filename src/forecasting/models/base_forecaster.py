"""
Base forecaster model made to be considered as an abstract interface for the
other main forecaster module.
"""

import polars as pl
from abc import ABC, abstractmethod

from forecasting.evaluation.evaluator import Evaluator


class BaseForecaster(ABC):

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def fit(self, df: pl.DataFrame):
        """
        Main class designed for machine learning models that require training.
        """
        pass

    @abstractmethod
    def predict(self, df: pl.DataFrame):
        """
        Main designed to perform prediction given the input
        """
        pass

    def evaluate(
        self,
        df: pl.DataFrame,
        target_col: str = "demand",
        forecast_col: str | None = None,
    ) -> dict[str, float]:
        """Compute metrics for this model's predictions.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame containing target and forecast columns.
        target_col : str
            Name of the actual values column.
        forecast_col : str | None
            Name of the forecast column. Defaults to self.forecast_col if available.

        Returns
        -------
        dict[str, float]
            Dictionary with keys: "wape", "mae", "rmse", "mape", "mse".

        Raises
        ------
        ValueError
            If required columns are missing or all forecast values are null.
        """
        # Resolve forecast column name
        if forecast_col is None:
            forecast_col = getattr(self, "forecast_col", None)
            if forecast_col is None:
                raise ValueError(
                    "forecast_col must be provided or set as an attribute on the model"
                )

        # Validate required columns exist
        if target_col not in df.columns:
            raise ValueError(
                f"Required column '{target_col}' is missing from the DataFrame"
            )
        if forecast_col not in df.columns:
            raise ValueError(
                f"Required column '{forecast_col}' is missing from the DataFrame"
            )

        # Check if all forecast values are null
        if df[forecast_col].is_null().all():
            raise ValueError(
                "All forecast values are null — no valid forecast values available for evaluation"
            )

        # Exclude rows where either target or forecast is null
        filtered_df = df.drop_nulls(subset=[target_col, forecast_col])

        # If zero rows remain after null exclusion, return NaN for all metrics
        if filtered_df.height == 0:
            return {
                "wape": float("nan"),
                "mae": float("nan"),
                "rmse": float("nan"),
                "mape": float("nan"),
                "mse": float("nan"),
            }

        # Delegate metric computation to the Evaluator
        evaluator = Evaluator(filtered_df, target_col)
        metrics = evaluator.compute_metrics(forecast_col)

        # Ensure all expected keys are present with float values
        return {
            "wape": float(metrics["wape"]),
            "mae": float(metrics["mae"]),
            "rmse": float(metrics["rmse"]),
            "mape": float(metrics["mape"]),
            "mse": float(metrics["mse"]),
        }
