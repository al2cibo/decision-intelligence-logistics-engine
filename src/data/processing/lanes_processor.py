"""Validation and cleaning for the lanes reference table."""

import polars as pl

from data.processing.validation import validate_columns, validate_non_empty


class LanesProcessor:
    """Validates and deduplicates the lanes reference table.

    Required columns: ``origin_id``, ``destination_id``, ``unit_cost``.
    """

    REQUIRED_COLUMNS: set[str] = {"origin_id", "destination_id", "unit_cost"}

    @staticmethod
    def process(df: pl.DataFrame) -> pl.DataFrame:
        """Validate and deduplicate the lanes table.

        Parameters
        ----------
        df : pl.DataFrame
            Raw lanes DataFrame.

        Returns
        -------
        pl.DataFrame
            Cleaned lanes table with duplicate rows removed.

        Raises
        ------
        ValueError
            If the DataFrame is empty or missing required columns.
        """
        validate_non_empty(df)
        validate_columns(df, LanesProcessor.REQUIRED_COLUMNS)
        return df.unique()
