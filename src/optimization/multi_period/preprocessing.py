"""Demand preprocessing for the multi-period optimizer."""

from datetime import date

import polars as pl


def preprocess_demand(
    demand_ts: pl.DataFrame, planning_horizon: list[date]
) -> pl.DataFrame:
    """Preprocess demand time series before LP formulation.

    Steps:
    1. Filter out rows with dates not in planning_horizon
    2. Fill null demand values with 0
    3. Deduplicate by summing demand for duplicate (destination_id, date) pairs

    Missing (destination, period) pairs are treated as zero demand during
    LP formulation (no explicit rows added here).

    Parameters
    ----------
    demand_ts : pl.DataFrame
        Raw demand time series ``[destination_id, date, demand]``.
    planning_horizon : list[date]
        Ordered list of dates representing valid time periods.

    Returns
    -------
    pl.DataFrame
        Preprocessed demand with schema ``[destination_id, date, demand]``.
    """
    horizon_set = set(planning_horizon)

    # 1. Filter out rows with dates outside planning_horizon
    demand_ts = demand_ts.filter(pl.col("date").is_in(list(horizon_set)))

    # 2. Fill null demand values with 0
    demand_ts = demand_ts.with_columns(pl.col("demand").fill_null(0))

    # 3. Deduplicate: sum demand for duplicate (destination_id, date) pairs
    demand_ts = demand_ts.group_by("destination_id", "date").agg(pl.col("demand").sum())

    return demand_ts
