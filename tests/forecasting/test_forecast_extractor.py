# Feature: multi-period-optimization, Property 10: ForecastExtractor Schema Preservation
"""Property-based tests for ForecastExtractor.extract_demand_time_series."""

from datetime import date, timedelta

import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from forecasting.forecast_extractor import ForecastExtractor


# --- Hypothesis Strategies ---

_destination_ids = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=8,
).map(lambda s: f"D_{s}")

_dates = st.dates(min_value=date(2020, 1, 1), max_value=date(2030, 12, 31))

_demand_values = st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False)

_forecast_col_names = st.text(
    alphabet=st.characters(whitelist_categories=("L",)),
    min_size=1,
    max_size=10,
).filter(lambda s: s not in ("date", "destination_id", "demand"))


@st.composite
def forecast_results_dataframe(draw):
    """Generate a valid forecasting results DataFrame with [date, destination_id, <forecast_col>]."""
    forecast_col = draw(_forecast_col_names)
    n_rows = draw(st.integers(min_value=1, max_value=50))

    destinations = draw(
        st.lists(_destination_ids, min_size=1, max_size=5)
    )
    dates_list = draw(
        st.lists(_dates, min_size=1, max_size=10)
    )
    demands = draw(
        st.lists(_demand_values, min_size=n_rows, max_size=n_rows)
    )

    # Build rows by cycling through destinations and dates
    row_destinations = [destinations[i % len(destinations)] for i in range(n_rows)]
    row_dates = [dates_list[i % len(dates_list)] for i in range(n_rows)]

    df = pl.DataFrame({
        "date": row_dates,
        "destination_id": row_destinations,
        forecast_col: demands,
    })

    return df, forecast_col


# --- Property Test ---


@settings(max_examples=100)
@given(data=forecast_results_dataframe())
def test_extract_demand_time_series_schema_preservation(data):
    """
    Property 10: ForecastExtractor Schema Preservation

    For any valid forecasting results DataFrame with columns [date, destination_id, <forecast_col>],
    calling extract_demand_time_series SHALL return a DataFrame with schema
    [destination_id: Utf8, date: Date, demand: Float64] where the row count equals the input row
    count and the demand column values equal the original forecast column values.

    **Validates: Requirements 5.6**
    """
    results_df, forecast_col = data

    output_df = ForecastExtractor.extract_demand_time_series(results_df, forecast_col)

    # 1. Schema check: output has exactly [destination_id, date, demand]
    assert output_df.columns == ["destination_id", "date", "demand"], (
        f"Expected columns [destination_id, date, demand], got {output_df.columns}"
    )

    # 2. Type check: destination_id is Utf8, date is Date, demand is Float64
    assert output_df.schema["destination_id"] == pl.Utf8, (
        f"Expected destination_id to be Utf8, got {output_df.schema['destination_id']}"
    )
    assert output_df.schema["date"] == pl.Date, (
        f"Expected date to be Date, got {output_df.schema['date']}"
    )
    assert output_df.schema["demand"] == pl.Float64, (
        f"Expected demand to be Float64, got {output_df.schema['demand']}"
    )

    # 3. Row count preservation
    assert output_df.height == results_df.height, (
        f"Expected {results_df.height} rows, got {output_df.height}"
    )

    # 4. Demand values equal original forecast column values
    original_values = results_df[forecast_col].cast(pl.Float64).to_list()
    output_values = output_df["demand"].to_list()
    assert original_values == output_values, (
        "Demand column values do not match original forecast column values"
    )
