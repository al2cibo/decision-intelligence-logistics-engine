"""Unified model selection for choosing the best forecasting model.

Provides a single ``ModelSelector`` class that supports two input modes:

- **DataFrame mode** (legacy pipeline): accepts a ``pl.DataFrame`` with a
  ``model_name`` column and metric columns.
- **Tuple-list mode** (per-destination pipeline): accepts a list of
  ``(model_name, metrics_dict)`` tuples.

Both modes select the model with the lowest value for a configurable metric,
skip NaN/null entries, and break ties by first-in-order.
"""

import logging
import math

import polars as pl

logger = logging.getLogger(__name__)


class ModelSelector:
    """Select the best forecasting model by minimising a configurable metric.

    Supports two input modes:

    - DataFrame mode (legacy pipeline): accepts ``pl.DataFrame`` with a
      ``model_name`` column.
    - Tuple-list mode (per-destination pipeline): accepts a list of
      ``(name, metrics)`` tuples.

    Parameters
    ----------
    metric : str, optional
        The metric name to minimise. Must be one of ``VALID_METRICS``.
        Defaults to ``"wape"``.

    Raises
    ------
    ValueError
        If *metric* is not in ``VALID_METRICS``.

    Examples
    --------
    >>> selector = ModelSelector(metric="mae")
    >>> selector.select_best_from_tuples([("a", {"mae": 3.0}), ("b", {"mae": 1.5})])
    ('b', {'mae': 1.5})
    """

    VALID_METRICS: set[str] = {"wape", "mae", "rmse", "mape", "mse"}

    def __init__(self, metric: str = "wape") -> None:
        if metric not in self.VALID_METRICS:
            raise ValueError(
                f"Metric '{metric}' not recognised. "
                f"Available: wape, mae, rmse, mape, mse"
            )
        self.metric = metric

    def select_best_from_dataframe(self, metrics_df: pl.DataFrame) -> str:
        """Select the best model from a metrics DataFrame.

        Parameters
        ----------
        metrics_df : pl.DataFrame
            A DataFrame containing at least a ``model_name`` column and the
            configured metric column. Rows with null values for the metric
            are skipped.

        Returns
        -------
        str
            The ``model_name`` of the best-performing model (lowest metric
            value). Ties are broken by first-in-order.

        Raises
        ------
        ValueError
            If *metrics_df* is empty, the metric column is missing, or all
            metric values are null/NaN.
        """
        if metrics_df.is_empty():
            raise ValueError("No model results available")

        if self.metric not in metrics_df.columns:
            available = [c for c in metrics_df.columns if c != "model_name"]
            raise ValueError(
                f"Metric '{self.metric}' not found. "
                f"Available metric columns: {available}"
            )

        # Filter out null/NaN values for the metric column
        valid_df = metrics_df.filter(pl.col(self.metric).is_not_null())
        valid_df = valid_df.filter(~pl.col(self.metric).is_nan())

        if valid_df.is_empty():
            raise ValueError(
                f"No valid (non-null, non-NaN) values for metric "
                f"'{self.metric}' across all models. "
                f"Cannot perform model selection."
            )

        # Sort ascending — first row is the minimum; stable sort preserves
        # input order for ties (first-in-order wins).
        sorted_df = valid_df.sort(self.metric)
        best_row = sorted_df[0]
        model_name: str = best_row["model_name"].item()
        metric_value = best_row[self.metric].item()

        logger.info(
            "Selected model '%s' with %s = %s",
            model_name,
            self.metric,
            metric_value,
        )

        return model_name

    def select_best_from_tuples(
        self,
        model_metrics: list[tuple[str, dict[str, float]]],
    ) -> tuple[str, dict[str, float]]:
        """Select the best model from an ordered list of (name, metrics) tuples.

        Parameters
        ----------
        model_metrics : list[tuple[str, dict[str, float]]]
            Ordered list of ``(model_name, metrics_dict)`` pairs. Input order
            determines tiebreaking (first wins).

        Returns
        -------
        tuple[str, dict[str, float]]
            A tuple of ``(model_name, metrics_dict)`` for the best model.

        Raises
        ------
        ValueError
            If no valid (non-null, non-NaN) metric values exist for the
            configured metric across all models.
        """
        best_name: str | None = None
        best_value: float | None = None
        best_metrics: dict[str, float] | None = None

        for model_name, metrics in model_metrics:
            value = metrics.get(self.metric)

            # Skip None values
            if value is None:
                continue

            # Skip NaN values
            if isinstance(value, float) and math.isnan(value):
                continue

            if best_value is None or value < best_value:
                best_name = model_name
                best_value = value
                best_metrics = metrics

        if best_name is None:
            raise ValueError(
                f"No valid (non-null, non-NaN) values for metric "
                f"'{self.metric}' across all models. "
                f"Cannot perform model selection."
            )

        return (best_name, best_metrics)

    @staticmethod
    def select_best(metrics_df: pl.DataFrame, metric: str = "wape") -> str:
        """Return the model_name with the lowest value for *metric*.

        Legacy static interface — delegates to the instance method
        :meth:`select_best_from_dataframe`.

        Parameters
        ----------
        metrics_df : pl.DataFrame
            A DataFrame containing at least ``model_name`` and the requested
            *metric* column.
        metric : str, optional
            Column name to minimise (default ``"wape"``).

        Returns
        -------
        str
            The ``model_name`` of the best-performing model.

        Raises
        ------
        ValueError
            If *metrics_df* is empty, *metric* is not a column, or all values
            are null/NaN.
        """
        selector = ModelSelector(metric=metric)
        return selector.select_best_from_dataframe(metrics_df)

    @staticmethod
    def rank_models(metrics_df: pl.DataFrame, metric: str = "wape") -> pl.DataFrame:
        """Return *metrics_df* sorted ascending by *metric*.

        Parameters
        ----------
        metrics_df : pl.DataFrame
            A metrics summary DataFrame.
        metric : str, optional
            Column name to sort by (default ``"wape"``).

        Returns
        -------
        pl.DataFrame
            The input DataFrame sorted in ascending order by *metric*.

        Raises
        ------
        ValueError
            If *metric* is not a column in *metrics_df*.
        """
        if metric not in metrics_df.columns:
            available = [c for c in metrics_df.columns if c != "model_name"]
            raise ValueError(
                f"Metric '{metric}' not found. "
                f"Available metric columns: {available}"
            )

        return metrics_df.sort(metric)
