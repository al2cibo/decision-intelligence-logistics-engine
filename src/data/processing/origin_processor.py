from data.processing.validation import validate_non_empty, validate_columns

from polars import DataFrame


class OriginsProcessor:

    REQUIRED_COLUMNS = {"origin_id"}

    @staticmethod
    def process(df: DataFrame) -> DataFrame:
        validate_non_empty(df)
        validate_columns(df, OriginsProcessor.REQUIRED_COLUMNS)

        return df.unique()
