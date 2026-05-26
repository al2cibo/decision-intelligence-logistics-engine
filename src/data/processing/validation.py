"""Shared validation functions for data processing.

This module provides module-level validation functions that replace the
static methods previously defined in BaseProcessor. All error messages
are preserved exactly for backward compatibility.
"""

import polars as pl


def validate_non_empty(df: pl.DataFrame) -> None:
    """Raise ValueError if the DataFrame is empty.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame to validate.

    Raises
    ------
    ValueError
        If the DataFrame has zero rows.
    """
    if df.is_empty():
        raise ValueError("Empty dataset")


def validate_no_nulls(df: pl.DataFrame) -> None:
    """Raise ValueError if the DataFrame contains null values.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame to validate.

    Raises
    ------
    ValueError
        If any column in the DataFrame contains at least one null value.
    """
    if df.null_count().to_numpy().sum() > 0:
        raise ValueError("Null values are not allowed.")


def validate_columns(df: pl.DataFrame, required_cols: set[str]) -> None:
    """Raise ValueError if required columns are missing from the DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame to validate.
    required_cols : set[str]
        The set of column names that must be present.

    Raises
    ------
    ValueError
        If any column in ``required_cols`` is not present in the DataFrame.
    """
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_cols}")
