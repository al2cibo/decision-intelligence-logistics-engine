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

