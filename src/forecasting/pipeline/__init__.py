"""Forecasting pipelines: orchestration of model training, evaluation, and selection."""

from .forecasting_pipeline_protocol import ForecastingPipelineProtocol
from .per_destination_forecasting_pipeline import (
    PerDestinationForecastingPipeline,
    AggregatedForecastingResult,
    DestinationOutcome,
)
from .forecasting_pipeline_factory import create_forecasting_pipeline

__all__ = [
    "ForecastingPipelineProtocol",
    "PerDestinationForecastingPipeline",
    "AggregatedForecastingResult",
    "DestinationOutcome",
    "create_forecasting_pipeline",
]
