"""Validation and cleaning for the destinations reference table."""

import polars as pl

from data.processing.validation import validate_columns, validate_non_empty


class DestinationsProcessor:
    """Validates and deduplicates the destinations reference table.

    Required columns: ``destination_id``.
    Additional columns (e.g. ``holding_cost``) are preserved as-is.
    """

    REQUIRED_COLUMNS: set[str] = {"destination_id"}

    @staticmethod
    def process(df: pl.DataFrame) -> pl.DataFrame:
        """Validate and deduplicate the destinations table.

        Parameters
        ----------
        df : pl.DataFrame
            Raw destinations DataFrame.

        Returns
        -------
        pl.DataFrame
            Cleaned destinations table with duplicate rows removed.

        Raises
        ------
        ValueError
            If the DataFrame is empty or missing required columns.
        """
        validate_non_empty(df)
        validate_columns(df, DestinationsProcessor.REQUIRED_COLUMNS)
        return df.unique()
