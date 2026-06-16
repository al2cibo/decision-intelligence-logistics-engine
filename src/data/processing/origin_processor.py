"""Validation and cleaning for the origins reference table."""

import polars as pl

from data.processing.validation import validate_columns, validate_non_empty


class OriginsProcessor:
    """Validates and deduplicates the origins reference table.

    Required columns: ``origin_id``.
    Additional columns (e.g. ``daily_capacity``) are preserved as-is.
    """

    REQUIRED_COLUMNS: set[str] = {"origin_id"}

    @staticmethod
    def process(df: pl.DataFrame) -> pl.DataFrame:
        """Validate and deduplicate the origins table.

        Parameters
        ----------
        df : pl.DataFrame
            Raw origins DataFrame.

        Returns
        -------
        pl.DataFrame
            Cleaned origins table with duplicate rows removed.

        Raises
        ------
        ValueError
            If the DataFrame is empty or missing required columns.
        """
        validate_non_empty(df)
        validate_columns(df, OriginsProcessor.REQUIRED_COLUMNS)
        return df.unique()
