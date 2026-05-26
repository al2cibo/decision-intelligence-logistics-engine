"""Per-destination model selection for choosing the best forecasting model."""

from dataclasses import dataclass

from forecasting.evaluation.model_selector import ModelSelector


@dataclass(frozen=True)
class SelectionResult:
    """Result of model selection for a single destination."""

    destination_id: str
    model_name: str
    metrics: dict[str, float]


class PerDestinationModelSelector:
    """Selects the best model for a destination based on a configurable metric.

    Delegates core selection logic to the unified
    :class:`~forecasting.evaluation.model_selector.ModelSelector`.

    Parameters
    ----------
    metric : str, optional
        The metric name to minimise. Must be one of ``VALID_METRICS``.
        Defaults to ``"wape"``.

    Raises
    ------
    ValueError
        If *metric* is not in ``VALID_METRICS``.
    """

    VALID_METRICS = {"wape", "mae", "rmse", "mape", "mse"}

    def __init__(self, metric: str = "wape"):
        self._selector = ModelSelector(metric=metric)
        self.metric = self._selector.metric

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
            The selection outcome containing destination_id, model_name,
            and the full metrics dictionary for the winning model.

        Raises
        ------
        ValueError
            If the metric name is not present in the metrics dicts, or if no
            valid (non-null, non-NaN) metric values exist for selection.
        """
        # Pre-validate that the metric key exists in each model's metrics dict.
        # The unified selector uses .get() which silently skips missing keys,
        # but the per-destination contract requires an explicit error.
        for model_name, metrics in model_metrics:
            if self.metric not in metrics:
                raise ValueError(
                    f"Metric '{self.metric}' not found in metrics for model "
                    f"'{model_name}'. Available: {sorted(metrics.keys())}"
                )

        try:
            best_name, best_metrics = self._selector.select_best_from_tuples(
                model_metrics
            )
        except ValueError:
            # Re-raise with destination-specific message to preserve the
            # original error contract.
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
