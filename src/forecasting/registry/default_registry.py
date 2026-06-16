"""Default model registry pre-populated with all built-in forecasters."""

from forecasting.registry.model_registry import ModelRegistry
from forecasting.models.naive_forecaster import NaiveForecaster
from forecasting.models.seasonal_forecaster import SeasonalForecaster
from forecasting.models.rolling_window_forecaster import RollingWindowForecaster
from forecasting.models.ets_forecaster import ETSForecaster
from forecasting.models.sarimax_forecaster import SARIMAXForecaster


def create_default_registry() -> ModelRegistry:
    """Return a ModelRegistry pre-populated with all five built-in models.

    Each class is registered directly as its own factory — ``registry.create(name, **kwargs)``
    calls ``ClassName(**kwargs)``, forwarding any model-specific parameter overrides
    (e.g. ``lag_value``, ``rolling_window``, ``order``) from the pipeline config.

    Registered names
    ----------------
    - ``naive_forecaster``
    - ``seasonal_forecaster``
    - ``rolling_window_forecaster``
    - ``ets_forecaster``
    - ``sarimax_forecaster``
    """
    registry = ModelRegistry()
    registry.register("naive_forecaster", NaiveForecaster)
    registry.register("seasonal_forecaster", SeasonalForecaster)
    registry.register("rolling_window_forecaster", RollingWindowForecaster)
    registry.register("ets_forecaster", ETSForecaster)
    registry.register("sarimax_forecaster", SARIMAXForecaster)
    return registry
