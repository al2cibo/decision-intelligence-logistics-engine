"""Forecasting registry: model name to factory callable mapping."""

from .model_registry import ModelRegistry
from .default_registry import create_default_registry

__all__ = [
    "ModelRegistry",
    "create_default_registry",
]
