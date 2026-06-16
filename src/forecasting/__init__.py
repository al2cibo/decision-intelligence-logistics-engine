"""Forecasting layer: models, pipelines, evaluation, and results."""

from .evaluation.evaluator import Evaluator
from .evaluation.model_selector import ModelSelector
from .evaluation.per_destination_model_selector import PerDestinationModelSelector, SelectionResult
from .models.base_forecaster import BaseForecaster
from .pipeline.pipeline_protocol import PipelineProtocol
from .pipeline.per_destination_pipeline import (
    PerDestinationPipeline,
    AggregatedPipelineResult,
    DestinationOutcome,
)
from .pipeline.pipeline_factory import create_per_destination_pipeline_from_config
from .results.forecast_result import ForecastResult, TimePeriod

__all__ = [
    "Evaluator",
    "ModelSelector",
    "PerDestinationModelSelector",
    "SelectionResult",
    "BaseForecaster",
    "PipelineProtocol",
    "PerDestinationPipeline",
    "AggregatedPipelineResult",
    "DestinationOutcome",
    "create_per_destination_pipeline_from_config",
    "ForecastResult",
    "TimePeriod",
]
