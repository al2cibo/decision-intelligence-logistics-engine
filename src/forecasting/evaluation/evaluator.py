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

    def __init__(self, df: pl.DataFrame, target_col_name: str):
        self.target_col_name = target_col_name
        self.df = df

    def compute_metrics(self, forecast_col_name):

        self.df = self._delete_null_values(self.df, forecast_col_name=forecast_col_name)

        target = self.df[self.target_col_name].to_numpy()
        forecast = self.df[forecast_col_name].to_numpy()

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
    def _wape(target_col, forecast_col):
        denominator = np.sum(np.abs(target_col))
        if denominator == 0:
            return np.nan

        return np.sum(np.abs(target_col - forecast_col)) / denominator

    @staticmethod
    def _delete_null_values(df: pl.DataFrame, forecast_col_name: str):
        original_length = df.height

        filtered_df = df.drop_nulls(subset=[forecast_col_name])
        if original_length != filtered_df.height:
            logger.warning(
                "Removed %d null values from '%s' column before evaluation.",
                original_length - filtered_df.height,
                forecast_col_name,
            )

        return filtered_df
