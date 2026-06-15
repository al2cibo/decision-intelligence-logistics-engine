"""Extracts flows and inventory DataFrames from a solved LP."""

from datetime import date

import polars as pl

from .model_builder import FlowVars, InventoryVars

FLOW_THRESHOLD = 1e-6

FLOWS_SCHEMA = {
    "origin_id": pl.Utf8,
    "destination_id": pl.Utf8,
    "period": pl.Date,
    "flow": pl.Float64,
}

INVENTORY_SCHEMA = {
    "destination_id": pl.Utf8,
    "period": pl.Date,
    "inventory": pl.Float64,
}


def extract_flows(flow_vars: FlowVars) -> pl.DataFrame:
    """Extract a flows DataFrame, keeping only flows above ``FLOW_THRESHOLD``."""
    flow_records = [
        {"origin_id": o_id, "destination_id": d_id, "period": t, "flow": val}
        for (o_id, d_id, t), var in flow_vars.items()
        if (val := var.solution_value()) > FLOW_THRESHOLD
    ]

    if flow_records:
        return pl.DataFrame(flow_records).cast(FLOWS_SCHEMA)
    return pl.DataFrame(schema=FLOWS_SCHEMA)


def extract_inventory(
    inv_vars: InventoryVars,
    destinations: list[str],
    planning_horizon: list[date],
) -> pl.DataFrame:
    """Extract an inventory DataFrame for every destination/period combination."""
    inv_records = [
        {
            "destination_id": d_id,
            "period": t,
            "inventory": inv_vars[(d_id, t)].solution_value(),
        }
        for d_id in destinations
        for t in planning_horizon
    ]

    return pl.DataFrame(inv_records).cast(INVENTORY_SCHEMA)
