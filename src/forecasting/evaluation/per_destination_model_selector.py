"""Per-destination model selection for choosing the best forecasting model."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SelectionResult:
    """Result of model selection for a single destination."""

    destination_id: str
    model_name: str
    metrics: dict[str, float]


class PerDestinationModelSelector:
    """Selects the best model for a destination based on a configurable metric."""

    VALID_METRICS = {"wape", "mae", "rmse", "mape", "mse"}

    def __init__(self, metric: str = "wape"):
        if metric not in self.VALID_METRICS:
            raise ValueError(
                f"Metric '{metric}' not recognised. "
                f"Available: wape, mae, rmse, mape, mse"
            )
        self.metric = metric

    def select(
        self,
        destination_id: str,
        model_metrics: list[tuple[str, dict[str, float]]],
    ) -> SelectionResult:
        """Select best model from ordered list of (model_name, metrics) tuples.

        Parameters
        ----------
        destination_id : str
            The destination being evaluated.
        model_metrics : list[tuple[str, dict[str, float]]]
            Ordered list of (model_name, metrics_dict) pairs.
            Order determines tiebreaking (first wins).

        Returns
        -------
        SelectionResult

        Raises
        ------
        ValueError
            If the metric name is not present in the metrics dicts, or if no
            valid (non-null, non-NaN) metric values exist for selection.
        """
        best_name: str | None = None
        best_value: float | None = None
        best_metrics: dict[str, float] | None = None

        for model_name, metrics in model_metrics:
            if self.metric not in metrics:
                raise ValueError(
                    f"Metric '{self.metric}' not found in metrics for model "
                    f"'{model_name}'. Available: {sorted(metrics.keys())}"
                )

            value = metrics[self.metric]

            # Skip NaN values
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue

            if best_value is None or value < best_value:
                best_name = model_name
                best_value = value
                best_metrics = metrics

        if best_name is None:
            raise ValueError(
                f"No valid (non-null, non-NaN) values for metric '{self.metric}' "
                f"across all models for destination '{destination_id}'. "
                f"Cannot perform model selection."
            )

        return SelectionResult(
            destination_id=destination_id,
            model_name=best_name,
            metrics=best_metrics,
        )
