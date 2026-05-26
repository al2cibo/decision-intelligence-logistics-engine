from data.processing.validation import validate_non_empty, validate_columns

from polars import DataFrame


class DestinationsProcessor:

    REQUIRED_COLUMNS = {"destination_id"}

    @staticmethod
    def process(df: DataFrame) -> DataFrame:
        validate_non_empty(df)
        validate_columns(df, DestinationsProcessor.REQUIRED_COLUMNS)

        return df.unique()
