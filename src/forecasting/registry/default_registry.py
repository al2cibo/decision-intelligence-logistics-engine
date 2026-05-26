"""Default model registry with all built-in forecasters registered.

Provides a `create_default_registry()` function that returns a ModelRegistry
pre-populated with factory functions for:
  - naive_forecaster
  - seasonal_forecaster
  - rolling_window_forecaster
  - ets_forecaster
  - sarimax_forecaster

Each factory accepts optional **kwargs that are forwarded to the model constructor,
allowing per-model parameter overrides (e.g., lag_value, rolling_window, order).
"""

from forecasting.registry.model_registry import ModelRegistry
from forecasting.models.naive_forecaster import NaiveForecaster
from forecasting.models.seasonal_forecaster import SeasonalForecaster
from forecasting.models.rolling_window_forecaster import RollingWindowForecaster
from forecasting.models.ets_forecaster import ETSForecaster
from forecasting.models.sarimax_forecaster import ARIMAForecaster


def _naive_factory(**kwargs) -> NaiveForecaster:
    """Create a NaiveForecaster instance.

    Parameters
    ----------
    target_col : str, optional
        Column name for the target variable (default: "demand").
    """
    return NaiveForecaster(**kwargs)


def _seasonal_factory(**kwargs) -> SeasonalForecaster:
    """Create a SeasonalForecaster instance.

    Parameters
    ----------
    target_col : str, optional
        Column name for the target variable (default: "demand").
    lag_value : int, optional
        Number of periods to lag (default: 7).
    """
    return SeasonalForecaster(**kwargs)


def _rolling_window_factory(**kwargs) -> RollingWindowForecaster:
    """Create a RollingWindowForecaster instance.

    Parameters
    ----------
    target_col : str, optional
        Column name for the target variable (default: "demand").
    rolling_window : int, optional
        Window size for the rolling mean (default: 1).
    """
    return RollingWindowForecaster(**kwargs)


def _ets_factory(**kwargs) -> ETSForecaster:
    """Create an ETSForecaster instance.

    Parameters
    ----------
    target_col : str, optional
        Column name for the target variable (default: "demand").
    trend : str | None, optional
        Trend component type (default: "add").
    seasonal : str | None, optional
        Seasonal component type (default: None).
    seasonal_periods : int | None, optional
        Number of periods in a seasonal cycle (default: None).
    """
    return ETSForecaster(**kwargs)


def _sarimax_factory(**kwargs) -> ARIMAForecaster:
    """Create a SARIMAX/ARIMA forecaster instance.

    Parameters
    ----------
    target_col : str, optional
        Column name for the target variable (default: "demand").
    order : tuple[int, int, int], optional
        ARIMA order (p, d, q) (default: (1, 1, 0)).
    seasonal_order : tuple[int, int, int, int] | None, optional
        Seasonal order (P, D, Q, s) (default: None).
    """
    return ARIMAForecaster(**kwargs)


def create_default_registry() -> ModelRegistry:
    """Create and return a ModelRegistry pre-populated with all built-in models.

    Returns
    -------
    ModelRegistry
        A registry containing factories for: naive_forecaster,
        seasonal_forecaster, rolling_window_forecaster, ets_forecaster,
        sarimax_forecaster.
    """
    registry = ModelRegistry()
    registry.register("naive_forecaster", _naive_factory)
    registry.register("seasonal_forecaster", _seasonal_factory)
    registry.register("rolling_window_forecaster", _rolling_window_factory)
    registry.register("ets_forecaster", _ets_factory)
    registry.register("sarimax_forecaster", _sarimax_factory)
    return registry
