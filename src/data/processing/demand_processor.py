"""Validation and cleaning for the demand history table."""

import polars as pl

from data.processing.validation import (
    validate_columns,
    validate_no_nulls,
    validate_non_empty,
)


class DemandProcessor:
    """Validates and deduplicates the demand history DataFrame.

    Required columns: ``date``, ``destination_id``, ``demand``.
    Output is sorted by ``[destination_id, date]`` for deterministic downstream use.
    """

    REQUIRED_COLUMNS: set[str] = {"date", "destination_id", "demand"}

    @staticmethod
    def process(df: pl.DataFrame) -> pl.DataFrame:
        """Validate, deduplicate, and sort the demand history.

        Parameters
        ----------
        df : pl.DataFrame
            Raw demand history DataFrame.

        Returns
        -------
        pl.DataFrame
            Cleaned demand history, sorted by ``[destination_id, date]``.

        Raises
        ------
        ValueError
            If the DataFrame is empty, contains nulls, or is missing required columns.
        """
        validate_non_empty(df)
        validate_no_nulls(df)
        validate_columns(df, DemandProcessor.REQUIRED_COLUMNS)
        return df.unique().sort(["destination_id", "date"])
