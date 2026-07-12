"""Factory function for creating a PerDestinationForecastingPipeline from configuration.

Provides `create_forecasting_pipeline` which:
1. Creates the default model registry
2. Validates all configured model names exist in the registry
3. Constructs and returns a fully-configured PerDestinationForecastingPipeline
"""

from forecasting.registry.default_registry import create_default_registry
from forecasting.pipeline.per_destination_forecasting_pipeline import (
    PerDestinationForecastingPipeline,
)
from forecasting.config import PerDestinationConfig


def create_forecasting_pipeline(
    config: PerDestinationConfig,
) -> PerDestinationForecastingPipeline:
    """Create a PerDestinationForecastingPipeline from a validated configuration.

    Parameters
    ----------
    config : PerDestinationConfig
        Configuration containing model_names, train_ratio, selection_metric,
        max_workers, minimum_history_length, random_seed, and model_params.

    Returns
    -------
    PerDestinationForecastingPipeline
        A fully-configured pipeline ready to call `.run(df)`.

    Raises
    ------
    ValueError
        If any model name in config.model_names is not registered in the
        default registry.
    """
    registry = create_default_registry()

    # Validate all model names exist in registry before pipeline starts
    available_models = registry.list_models()
    missing = [name for name in config.model_names if name not in registry]
    if missing:
        raise ValueError(
            f"The following model names from configuration are not registered "
            f"in the model registry: {missing}. "
            f"Available models: {available_models}"
        )

    return PerDestinationForecastingPipeline(
        registry=registry,
        model_names=config.model_names,
        train_ratio=config.train_ratio,
        selection_metric=config.selection_metric,
        max_workers=config.max_workers,
        minimum_history_length=config.minimum_history_length,
        random_seed=config.random_seed,
        model_params=config.model_params,
        validation_periods=config.validation_periods,
        test_periods=config.test_periods,
    )
