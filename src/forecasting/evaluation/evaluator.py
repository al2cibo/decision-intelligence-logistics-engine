"""Forecasting accuracy metrics computation.

Computes MAE, MSE, RMSE, MAPE, and WAPE given a DataFrame with
target and forecast columns.
"""

import logging

import numpy as np
import polars as pl
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """Compute forecast accuracy metrics against a target column.

    The Evaluator holds a reference DataFrame and target column name,
    then computes standard regression metrics for any forecast column
    in that DataFrame. The instance DataFrame is never mutated after
    construction, ensuring repeatable metric computation.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame containing at least the target column and one or more
        forecast columns.
    target_col_name : str
        Name of the column holding actual/target values.

    Examples
    --------
    >>> evaluator = Evaluator(df, "demand")
    >>> metrics = evaluator.compute_metrics("forecast_model_a")
    """

    def __init__(self, df: pl.DataFrame, target_col_name: str) -> None:
        self.target_col_name = target_col_name
        self.df = df

    def compute_metrics(self, forecast_col_name: str) -> dict[str, float]:
        """Compute accuracy metrics for a given forecast column.

        Filters null values from the forecast column into a local
        variable without modifying the instance DataFrame. If the
        forecast column contains only null values after filtering,
        returns a metrics dictionary with NaN values.

        Parameters
        ----------
        forecast_col_name : str
            Name of the column holding forecast/predicted values.

        Returns
        -------
        dict[str, float]
            Dictionary with keys 'mae', 'mse', 'rmse', 'mape', 'wape'
            and their corresponding metric values.
        """
        filtered_df = self._delete_null_values(
            self.df, forecast_col_name=forecast_col_name
        )

        if filtered_df.is_empty():
            return {
                "mae": float("nan"),
                "mse": float("nan"),
                "rmse": float("nan"),
                "mape": float("nan"),
                "wape": float("nan"),
            }

        target = filtered_df[self.target_col_name].to_numpy()
        forecast = filtered_df[forecast_col_name].to_numpy()

        mae = mean_absolute_error(target, forecast)
        mse = mean_squared_error(target, forecast)
        rmse = root_mean_squared_error(target, forecast)
        mape = mean_absolute_percentage_error(target, forecast)
        wape = self._wape(target, forecast)

        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "mape": mape,
            "wape": wape,
        }

    @staticmethod
    def _wape(target_col: np.ndarray, forecast_col: np.ndarray) -> float:
        """Compute Weighted Absolute Percentage Error.

        Parameters
        ----------
        target_col : np.ndarray
            Array of actual/target values.
        forecast_col : np.ndarray
            Array of forecast/predicted values.

        Returns
        -------
        float
            WAPE value, or NaN if the sum of absolute targets is zero.
        """
        denominator = np.sum(np.abs(target_col))
        if denominator == 0:
            return np.nan

        return np.sum(np.abs(target_col - forecast_col)) / denominator

    @staticmethod
    def _delete_null_values(df: pl.DataFrame, forecast_col_name: str) -> pl.DataFrame:
        """Filter rows with null values in the forecast column.

        Parameters
        ----------
        df : pl.DataFrame
            Input DataFrame to filter.
        forecast_col_name : str
            Name of the forecast column to check for nulls.

        Returns
        -------
        pl.DataFrame
            DataFrame with null forecast rows removed.
        """
        original_length = df.height

        filtered_df = df.drop_nulls(subset=[forecast_col_name])
        if original_length != filtered_df.height:
            logger.warning(
                "Removed %d null values from '%s' column before evaluation.",
                original_length - filtered_df.height,
                forecast_col_name,
            )

        return filtered_df
