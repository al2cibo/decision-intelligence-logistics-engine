"""In-memory implementation of PersistenceInterface for testing."""

from typing import Any

from forecasting.results.forecast_result import ForecastResult
from forecasting.persistence.persistence import PersistenceInterface


class InMemoryPersistence(PersistenceInterface):
    """Concrete in-memory persistence using dictionaries.

    Storage is keyed by (destination_id, model_name) tuples for results
    and model artifacts. Metrics are stored per destination_id with nested
    model_name -> metric dictionaries.

    This implementation is intended for testing and development use.
    """

    def __init__(self) -> None:
        self._results: dict[tuple[str, str], ForecastResult] = {}
        self._artifacts: dict[tuple[str, str], Any] = {}
        self._metrics: dict[str, dict[str, dict[str, float]]] = {}

    def save_result(self, result: ForecastResult) -> None:
        """Save a single ForecastResult, keyed by (destination_id, model_name)."""
        key = (result.destination_id, result.model_name)
        self._results[key] = result

    def load_result(self, destination_id: str, model_name: str) -> ForecastResult:
        """Load a ForecastResult. Raises KeyError if not found."""
        key = (destination_id, model_name)
        if key not in self._results:
            raise KeyError(
                f"No result found for destination_id='{destination_id}', "
                f"model_name='{model_name}'"
            )
        return self._results[key]

    def save_results_batch(
        self, destination_id: str, results: list[ForecastResult]
    ) -> None:
        """Save all results for a destination."""
        for result in results:
            key = (destination_id, result.model_name)
            self._results[key] = result

    def load_results_batch(self, destination_id: str) -> list[ForecastResult]:
        """Load all results for a destination. Raises KeyError if not found."""
        batch = [
            result
            for (dest_id, _), result in self._results.items()
            if dest_id == destination_id
        ]
        if not batch:
            raise KeyError(
                f"No results found for destination_id='{destination_id}'"
            )
        return batch

    def save_model_artifact(
        self, destination_id: str, model_name: str, artifact: Any
    ) -> None:
        """Save a trained model artifact."""
        key = (destination_id, model_name)
        self._artifacts[key] = artifact

    def load_model_artifact(self, destination_id: str, model_name: str) -> Any:
        """Load a trained model artifact. Raises KeyError if not found."""
        key = (destination_id, model_name)
        if key not in self._artifacts:
            raise KeyError(
                f"No model artifact found for destination_id='{destination_id}', "
                f"model_name='{model_name}'"
            )
        return self._artifacts[key]

    def save_metrics(
        self, destination_id: str, metrics: dict[str, dict[str, float]]
    ) -> None:
        """Save metrics for a destination: {model_name: {metric: value}}."""
        self._metrics[destination_id] = metrics

    def load_metrics(self, destination_id: str) -> dict[str, dict[str, float]]:
        """Load metrics for a destination. Raises KeyError if not found."""
        if destination_id not in self._metrics:
            raise KeyError(
                f"No metrics found for destination_id='{destination_id}'"
            )
        return self._metrics[destination_id]

    def load_all_metrics(self) -> dict[str, dict[str, dict[str, float]]]:
        """Load metrics for all destinations: {dest_id: {model: {metric: val}}}."""
        return dict(self._metrics)
