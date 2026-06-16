"""Per-destination model selection wrapper."""

from dataclasses import dataclass

from forecasting.evaluation.model_selector import ModelSelector


@dataclass(frozen=True)
class SelectionResult:
    """Immutable result of model selection for a single destination."""

    destination_id: str
    model_name: str
    metrics: dict[str, float]


class PerDestinationModelSelector:
    """Selects the best model for a given destination.

    Wraps :class:`~forecasting.evaluation.model_selector.ModelSelector` with
    an explicit pre-validation step that raises a destination-specific error
    when the configured metric key is absent from any model's metrics dict.
    
    TLDR: ModelSelector is the core class that, given a list of 
    ``(model_name, metrics_dict)`` tuples, returns the model with the 
    lowest value for the configured metric. PerDestinationModelSelector
    works as a wrapper, tying the model selection to each single destination.

    Parameters
    ----------
    metric : str
        Metric to minimise. Must be one of ``ModelSelector.VALID_METRICS``.
        Defaults to ``"wape"``.
    """

    def __init__(self, metric: str = "wape") -> None:
        self._selector = ModelSelector(metric=metric)
        self.metric = self._selector.metric

    def select(
        self,
        destination_id: str,
        model_metrics: list[tuple[str, dict[str, float]]],
    ) -> SelectionResult:
        """Select the best model for a destination.

        Parameters
        ----------
        destination_id : str
            Identifier for the destination being evaluated.
        model_metrics : list[tuple[str, dict[str, float]]]
            Ordered list of ``(model_name, metrics_dict)`` pairs.

        Returns
        -------
        SelectionResult
            The winning model and its full metrics dict.

        Raises
        ------
        ValueError
            If the metric key is absent from any model's dict, or if no
            valid metric values exist for selection.
        """
        for model_name, metrics in model_metrics:
            if self.metric not in metrics:
                raise ValueError(
                    f"Metric '{self.metric}' not found in metrics for model "
                    f"'{model_name}' at destination '{destination_id}'. "
                    f"Available: {sorted(metrics.keys())}"
                )

        try:
            best_name, best_metrics = self._selector.select_best_from_tuples(model_metrics)
        except ValueError:
            raise ValueError(
                f"No valid (non-null, non-NaN) values for metric '{self.metric}' "
                f"across all models for destination '{destination_id}'."
            )

        return SelectionResult(
            destination_id=destination_id,
            model_name=best_name,
            metrics=best_metrics,
        )
