"""Shared validation functions for DataFrame pre-processing."""

import polars as pl


def validate_non_empty(df: pl.DataFrame) -> None:
    """Raise ValueError if the DataFrame has zero rows."""
    if df.is_empty():
        raise ValueError("Empty dataset")


def validate_no_nulls(df: pl.DataFrame) -> None:
    """Raise ValueError if the DataFrame contains any null values."""
    if df.null_count().to_numpy().sum() > 0:
        raise ValueError("Null values are not allowed.")


def validate_columns(df: pl.DataFrame, required_cols: set[str]) -> None:
    """Raise ValueError listing only the columns that are actually missing."""
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
