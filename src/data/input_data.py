"""Container for the four raw logistics DataFrames produced by the Reader."""

from dataclasses import dataclass

import polars as pl


@dataclass
class InputData:
    """Bundles the four logistics tables into a single object.

    All four DataFrames are required. The Reader populates them from parquet;
    the DataProcessor validates and cleans them in place.
    """

    demand_history: pl.DataFrame
    """Historical demand per destination per day: [date, destination_id, demand]."""

    destinations: pl.DataFrame
    """Destination reference table: [destination_id, ...]."""

    lanes: pl.DataFrame
    """Available shipping lanes with costs: [origin_id, destination_id, unit_cost]."""

    origins: pl.DataFrame
    """Origin reference table with capacity: [origin_id, daily_capacity]."""
