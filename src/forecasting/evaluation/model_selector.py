"""Model selection: choose the best forecasting model by minimising a metric."""

import logging
import math

logger = logging.getLogger(__name__)


class ModelSelector:
    """Selects the best forecasting model from an ordered list of candidates.

    Given a list of ``(model_name, metrics_dict)`` tuples, returns the model
    with the lowest value for the configured metric. NaN and None values are
    skipped. Ties are broken by insertion order (first wins).

    Parameters
    ----------
    metric : str
        Metric to minimise. Must be one of ``VALID_METRICS``. Defaults to ``"wape"``.

    Raises
    ------
    ValueError
        If ``metric`` is not in ``VALID_METRICS``.

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
            Ordered list of ``(model_name, metrics_dict)`` pairs.

        Returns
        -------
        tuple[str, dict[str, float]]
            ``(model_name, metrics_dict)`` for the winning model.

        Raises
        ------
        ValueError
            If no valid (non-null, non-NaN) metric values exist.
        """
        best_name: str | None = None
        best_value: float | None = None
        best_metrics: dict[str, float] | None = None

        for model_name, metrics in model_metrics:
            value = metrics.get(self.metric)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            if best_value is None or value < best_value:
                best_name = model_name
                best_value = value
                best_metrics = metrics

        if best_name is None:
            raise ValueError(
                f"No valid (non-null, non-NaN) values for metric '{self.metric}' "
                f"across all models. Cannot perform model selection."
            )

        return best_name, best_metrics
