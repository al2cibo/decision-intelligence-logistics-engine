"""Data layer: ingestion, generation, and processing of logistics DataFrames."""

from .input_data import InputData
from .ingestion import Reader
from .generation import generate_synthetic_logistics_data

__all__ = [
    "InputData",
    "Reader",
    "generate_synthetic_logistics_data",
]
