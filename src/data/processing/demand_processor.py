from data.processing.validation import validate_non_empty, validate_no_nulls, validate_columns

from polars import DataFrame


class DemandProcessor:

    REQUIRED_COLUMNS = {"date", "destination_id", "demand"}

    @staticmethod
    def process(df: DataFrame) -> DataFrame:
        validate_non_empty(df)
        validate_no_nulls(df)
        validate_columns(df, DemandProcessor.REQUIRED_COLUMNS)

        df = df.unique()
        df = df.sort(["destination_id", "date"])

        return df
