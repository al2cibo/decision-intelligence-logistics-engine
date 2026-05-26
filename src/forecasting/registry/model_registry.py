"""Model registry that maps model names to factory callables."""

from typing import Any, Callable

from forecasting.models.base_forecaster import BaseForecaster


class ModelRegistry:
    """Maps model names to factory callables. Preserves insertion order."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., BaseForecaster]] = {}

    def register(self, name: str, factory: Callable[..., BaseForecaster]) -> None:
        """Register a model factory. Overwrites if name already exists."""
        self._factories[name] = factory

    def create(self, name: str, **kwargs: Any) -> BaseForecaster:
        """Create a new model instance.

        Parameters
        ----------
        name : str
            Registered model name.
        **kwargs
            Model-specific parameters passed to the factory.

        Returns
        -------
        BaseForecaster
            A new, independent model instance.

        Raises
        ------
        KeyError
            If name is not registered. Message lists available models.
        """
        if name not in self._factories:
            available = list(self._factories.keys())
            raise KeyError(
                f"Model '{name}' is not registered. "
                f"Available models: {available}"
            )
        return self._factories[name](**kwargs)

    def list_models(self) -> list[str]:
        """Return registered model names in insertion order."""
        return list(self._factories.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._factories

    def __len__(self) -> int:
        return len(self._factories)
