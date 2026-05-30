"""Forecasting pipelines: orchestration of model training, evaluation, and selection."""

from .pipeline_protocol import PipelineProtocol
from .per_destination_pipeline import (
    PerDestinationPipeline,
    AggregatedPipelineResult,
    DestinationOutcome,
)
from .pipeline_factory import create_per_destination_pipeline_from_config

__all__ = [
    "PipelineProtocol",
    "PerDestinationPipeline",
    "AggregatedPipelineResult",
    "DestinationOutcome",
    "create_per_destination_pipeline_from_config",
]
