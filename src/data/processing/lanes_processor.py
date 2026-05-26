from data.processing.validation import validate_non_empty, validate_columns

from polars import DataFrame


class LanesProcessor:

    REQUIRED_COLUMNS = {"origin_id", "destination_id", "unit_cost"}

    @staticmethod
    def process(df: DataFrame) -> DataFrame:
        validate_non_empty(df)
        validate_columns(df, LanesProcessor.REQUIRED_COLUMNS)

        df = df.unique()

        return df
