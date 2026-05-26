"""Abstract persistence interface for forecast artifacts."""

from abc import ABC, abstractmethod
from typing import Any

from forecasting.results.forecast_result import ForecastResult


class PersistenceInterface(ABC):
    """Abstract interface for saving/loading forecast artifacts.

    All load methods raise KeyError with a descriptive message if the
    requested composite key (destination_id, model_name) does not exist.
    """

    @abstractmethod
    def save_result(self, result: ForecastResult) -> None:
        """Save a single ForecastResult, keyed by (destination_id, model_name)."""
        ...

    @abstractmethod
    def load_result(self, destination_id: str, model_name: str) -> ForecastResult:
        """Load a ForecastResult. Raises KeyError if not found."""
        ...

    @abstractmethod
    def save_results_batch(
        self, destination_id: str, results: list[ForecastResult]
    ) -> None:
        """Save all results for a destination."""
        ...

    @abstractmethod
    def load_results_batch(self, destination_id: str) -> list[ForecastResult]:
        """Load all results for a destination. Raises KeyError if not found."""
        ...

    @abstractmethod
    def save_model_artifact(
        self, destination_id: str, model_name: str, artifact: Any
    ) -> None:
        """Save a trained model artifact."""
        ...

    @abstractmethod
    def load_model_artifact(self, destination_id: str, model_name: str) -> Any:
        """Load a trained model artifact. Raises KeyError if not found."""
        ...

    @abstractmethod
    def save_metrics(
        self, destination_id: str, metrics: dict[str, dict[str, float]]
    ) -> None:
        """Save metrics for a destination: {model_name: {metric: value}}."""
        ...

    @abstractmethod
    def load_metrics(self, destination_id: str) -> dict[str, dict[str, float]]:
        """Load metrics for a destination. Raises KeyError if not found."""
        ...

    @abstractmethod
    def load_all_metrics(self) -> dict[str, dict[str, dict[str, float]]]:
        """Load metrics for all destinations: {dest_id: {model: {metric: val}}}."""
        ...
