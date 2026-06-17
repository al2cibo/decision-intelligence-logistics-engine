"""Forecasting layer: models, pipelines, evaluation, and results."""

from .evaluation.evaluator import Evaluator
from .evaluation.model_selector import ModelSelector
from .evaluation.per_destination_model_selector import PerDestinationModelSelector, SelectionResult
from .models.base_forecaster import BaseForecaster
from .pipeline.forecasting_pipeline_protocol import ForecastingPipelineProtocol
from .pipeline.per_destination_forecasting_pipeline import (
    PerDestinationForecastingPipeline,
    AggregatedForecastingResult,
    DestinationOutcome,
)
from .pipeline.forecasting_pipeline_factory import create_forecasting_pipeline
from .results.forecast_result import ForecastResult, TimePeriod

__all__ = [
    "Evaluator",
    "ModelSelector",
    "PerDestinationModelSelector",
    "SelectionResult",
    "BaseForecaster",
    "ForecastingPipelineProtocol",
    "PerDestinationForecastingPipeline",
    "AggregatedForecastingResult",
    "DestinationOutcome",
    "create_forecasting_pipeline",
    "ForecastResult",
    "TimePeriod",
]
