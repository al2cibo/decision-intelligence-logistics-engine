"""Evaluate a completed experiment's plan against actual demand.

The optimizer only sees forecasted demand, so its plan may over- or under-ship
relative to what actually happens.  This module simulates the planned flows
against ground-truth demand to produce realized (as-executed) metrics,
both in aggregate and broken down per destination.

Usage:
    from actuals_evaluator import evaluate, save_realized_metrics
    metrics = evaluate(output_path)
    save_realized_metrics(metrics, output_path)
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

from data.ingestion import Reader
from experiment_config import load_experiment_config
from utils.system_paths import get_project_root


@dataclass
class DestinationActuals:
    transport_cost: float
    realized_holding_cost: float
    realized_total_cost: float
    total_actual_demand: float
    total_fulfilled: float
    total_shortage: float
    fill_rate: float
    cost_per_unit_demanded: float
    cost_per_unit_fulfilled: float


@dataclass
class RealizedMetrics:
    transport_cost: float
    realized_holding_cost: float
    realized_total_cost: float
    total_actual_demand: float
    total_fulfilled: float
    total_shortage: float
    fill_rate: float
    cost_per_unit_demanded: float
    cost_per_unit_fulfilled: float
    per_destination: dict[str, DestinationActuals]


def evaluate(experiment_output_path: Path) -> RealizedMetrics:
    """Simulate the experiment's planned flows against actual demand.

    Reads flows.parquet and planning_metrics.json from experiment_output_path,
    plus the original dataset referenced in config.yaml, and returns realized metrics.
    """
    project_root = get_project_root()
    config = load_experiment_config(
        project_root, experiment_output_path / "config.yaml"
    )

    flows = pl.read_parquet(experiment_output_path / "flows.parquet")

    raw = Reader(config.dataset_path).read()

    holding_cost_map: dict[str, float] = {}
    if "holding_cost" in raw.destinations.columns:
        holding_cost_map = dict(
            zip(
                raw.destinations["destination_id"].to_list(),
                raw.destinations["holding_cost"].to_list(),
            )
        )

    # Per-destination transport cost from planned flows × lane unit costs
    lane_cost_map: dict[tuple, float] = {
        (row["origin_id"], row["destination_id"]): row["unit_cost"]
        for row in raw.lanes.to_dicts()
    }
    dest_transport_cost: dict[str, float] = {}
    for row in flows.to_dicts():
        d_id = row["destination_id"]
        cost = lane_cost_map.get((row["origin_id"], d_id), 0.0) * row["flow"]
        dest_transport_cost[d_id] = dest_transport_cost.get(d_id, 0.0) + cost

    forecast_dates = sorted(flows["period"].unique().to_list())

    actual_demand = raw.demand_history.filter(pl.col("date").is_in(forecast_dates))

    # Aggregate planned inflows per (destination, period)
    inflows = (
        flows.group_by(["destination_id", "period"])
        .agg(pl.col("flow").sum().alias("inflow"))
        .rename({"period": "date"})
    )

    destinations = sorted(raw.destinations["destination_id"].to_list())

    total_actual_demand = 0.0
    total_fulfilled = 0.0
    total_shortage = 0.0
    realized_holding_cost = 0.0
    per_destination: dict[str, DestinationActuals] = {}

    for d_id in destinations:
        h_cost = holding_cost_map.get(d_id, 0.0)

        inflow_map: dict = {
            row["date"]: row["inflow"]
            for row in inflows.filter(pl.col("destination_id") == d_id).to_dicts()
        }
        demand_map: dict = {
            row["date"]: row["demand"]
            for row in actual_demand.filter(pl.col("destination_id") == d_id).to_dicts()
        }

        d_fulfilled = 0.0
        d_shortage = 0.0
        d_holding_cost = 0.0
        d_demand = 0.0
        inventory = 0.0

        for t in forecast_dates:
            inflow = inflow_map.get(t, 0.0)
            actual_d = demand_map.get(t, 0.0)

            available = inventory + inflow
            fulfilled = min(actual_d, available)
            shortage = max(0.0, actual_d - available)
            inventory = available - fulfilled

            d_demand += actual_d
            d_fulfilled += fulfilled
            d_shortage += shortage
            d_holding_cost += h_cost * inventory

        d_transport = dest_transport_cost.get(d_id, 0.0)
        d_realized_total = d_transport + d_holding_cost
        d_fill_rate = d_fulfilled / d_demand if d_demand > 0 else 1.0
        d_cpu_demanded = d_realized_total / d_demand if d_demand > 0 else 0.0
        d_cpu_fulfilled = d_realized_total / d_fulfilled if d_fulfilled > 0 else 0.0

        per_destination[d_id] = DestinationActuals(
            transport_cost=d_transport,
            realized_holding_cost=d_holding_cost,
            realized_total_cost=d_realized_total,
            total_actual_demand=d_demand,
            total_fulfilled=d_fulfilled,
            total_shortage=d_shortage,
            fill_rate=d_fill_rate,
            cost_per_unit_demanded=d_cpu_demanded,
            cost_per_unit_fulfilled=d_cpu_fulfilled,
        )

        total_actual_demand += d_demand
        total_fulfilled += d_fulfilled
        total_shortage += d_shortage
        realized_holding_cost += d_holding_cost

    transport_cost = sum(dest_transport_cost.values())
    realized_total_cost = transport_cost + realized_holding_cost
    fill_rate = (
        total_fulfilled / total_actual_demand if total_actual_demand > 0 else 1.0
    )
    cpu_demanded = (
        realized_total_cost / total_actual_demand if total_actual_demand > 0 else 0.0
    )
    cpu_fulfilled = (
        realized_total_cost / total_fulfilled if total_fulfilled > 0 else 0.0
    )

    return RealizedMetrics(
        transport_cost=transport_cost,
        realized_holding_cost=realized_holding_cost,
        realized_total_cost=realized_total_cost,
        total_actual_demand=total_actual_demand,
        total_fulfilled=total_fulfilled,
        total_shortage=total_shortage,
        fill_rate=fill_rate,
        cost_per_unit_demanded=cpu_demanded,
        cost_per_unit_fulfilled=cpu_fulfilled,
        per_destination=per_destination,
    )


def save_realized_metrics(metrics: RealizedMetrics, output_path: Path) -> None:
    with open(output_path / "realized_metrics.json", "w") as f:
        json.dump(asdict(metrics), f, indent=2)
