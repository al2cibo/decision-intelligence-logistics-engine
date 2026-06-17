"""Input validation and feasibility pre-checks for the multi-period optimizer.

``validate_inputs`` performs cheap structural checks (non-empty, required
columns, non-negative costs, positive capacities, valid initial inventory,
variable-count guard). ``check_feasibility`` performs the more expensive
pre-solve feasibility checks (unreachable destinations, aggregate capacity
vs. demand). Both raise ``ValueError`` with a descriptive message on the
first violation found.
"""

from datetime import date

import polars as pl

from optimization.validation import (
    check_capacity_feasibility,
    check_unreachable_destinations,
    validate_columns,
    validate_non_negative_costs,
    validate_not_empty,
    validate_origins_in_lanes,
    validate_positive_capacities,
)

MAX_VARIABLES = 1_000_000


def validate_inputs(
    demand_ts: pl.DataFrame,
    origins_df: pl.DataFrame,
    lanes_df: pl.DataFrame,
    destinations_df: pl.DataFrame,
    planning_horizon: list[date],
    initial_inventory: dict[str, float] | None,
) -> None:
    """Run structural validation. Raises ``ValueError`` on the first violation."""
    _validate_not_empty(demand_ts, origins_df, lanes_df, planning_horizon)
    validate_columns(
        demand_ts,
        {"destination_id", "date", "demand"},
        "Demand time series",
        message_template="{df_name} missing required columns: {missing}",
    )
    validate_non_negative_costs(lanes_df, destinations_df)
    validate_positive_capacities(origins_df)
    validate_origins_in_lanes(origins_df, lanes_df)
    _validate_initial_inventory(initial_inventory)
    _validate_variable_count(lanes_df, demand_ts, planning_horizon)


def check_feasibility(
    demand_ts: pl.DataFrame,
    origins_df: pl.DataFrame,
    lanes_df: pl.DataFrame,
    planning_horizon: list[date],
) -> None:
    """Raise ``ValueError`` if the problem is structurally infeasible before solving."""
    check_unreachable_destinations(
        demand_ts,
        lanes_df,
        message_template="Unreachable destinations (no lane serves them): {unreachable}",
    )
    _check_capacity_feasibility(demand_ts, origins_df, planning_horizon)


def _validate_not_empty(
    demand_ts: pl.DataFrame,
    origins_df: pl.DataFrame,
    lanes_df: pl.DataFrame,
    planning_horizon: list[date],
) -> None:
    """Raise ``ValueError`` if any required input is empty."""
    validate_not_empty(
        **{
            "no demand data available": demand_ts,
            "Origins DataFrame is empty": origins_df,
            "Lanes DataFrame is empty": lanes_df,
        }
    )
    if len(planning_horizon) == 0:
        raise ValueError("Planning horizon contains zero periods")


def _validate_initial_inventory(initial_inventory: dict[str, float] | None) -> None:
    """Raise ``ValueError`` if any initial inventory value is negative."""
    if initial_inventory is None:
        return
    for dest_id, value in initial_inventory.items():
        if value < 0:
            raise ValueError(
                f"Negative initial inventory for destination '{dest_id}': " f"{value}"
            )


def _validate_variable_count(
    lanes_df: pl.DataFrame,
    demand_ts: pl.DataFrame,
    planning_horizon: list[date],
) -> None:
    """Raise ``ValueError`` if variable count exceeds ``MAX_VARIABLES``."""
    n_periods = len(planning_horizon)
    n_lanes = len(lanes_df)
    n_destinations = demand_ts["destination_id"].n_unique()

    flow_vars = n_lanes * n_periods
    inventory_vars = n_destinations * n_periods
    total_vars = flow_vars + inventory_vars

    if total_vars > MAX_VARIABLES:
        raise ValueError(
            f"Variable count ({total_vars}) exceeds maximum limit "
            f"({MAX_VARIABLES}). "
            f"Flow variables: {flow_vars}, "
            f"inventory variables: {inventory_vars}."
        )


def _check_capacity_feasibility(
    demand_ts: pl.DataFrame,
    origins_df: pl.DataFrame,
    planning_horizon: list[date],
) -> None:
    """Raise ``ValueError`` if total capacity < total demand.

    Total capacity is the sum of daily_capacity for all origins multiplied
    by the number of periods in the planning horizon.
    Total demand is the sum of all demand values (nulls treated as zero).
    """
    n_periods = len(planning_horizon)
    total_capacity = origins_df["daily_capacity"].sum() * n_periods
    total_demand = demand_ts["demand"].fill_null(0).sum()

    check_capacity_feasibility(
        total_demand,
        total_capacity,
        message_template=(
            "Insufficient total capacity: total demand = {total_demand}, "
            "total capacity = {total_capacity}, shortfall = {shortfall}"
        ),
    )
