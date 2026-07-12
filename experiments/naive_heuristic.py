"""Naive forecasting and optimization heuristics for the 2×2 experiment baseline.

Two functions are exposed:

``compute_lag1_forecast``
    Produces the F0 (naive) demand signal for the test window by shifting actual
    demand back one period per destination.  This mirrors what ``NaiveForecaster``
    does inside the DILE pipeline, but is computed outside it so it can be injected
    directly into either the LP or the proportional heuristic.

``run_naive_heuristic``
    Implements the O0 (naive) optimization strategy: each origin distributes its
    full daily capacity proportionally across destinations according to their
    forecasted demand share.  No LP, no inventory tracking in the plan.
    Realized inventory and unmet demand are computed retroactively from actual demand.
"""

from dataclasses import dataclass
from datetime import date as Date

import polars as pl


@dataclass
class HeuristicResult:
    """Outcome of the proportional capacity heuristic.

    ``flows`` and ``inventory`` share the same schema as ``MultiPeriodResult``
    so that ``actuals_evaluator`` and ``run_experiment`` can treat both result
    types uniformly.

    Parameters
    ----------
    flows : pl.DataFrame
        Planned flows ``[origin_id, destination_id, period: Date, flow: Float64]``.
        Every (origin, destination, period) triple is present (no threshold filtering).
    inventory : pl.DataFrame
        Retroactive actual inventory ``[destination_id, period: Date, inventory: Float64]``
        computed from planned inflows and actual demand.  The heuristic does not track
        inventory during planning; this is the ground-truth execution outcome.
    transportation_cost : float
        ``Σ flow(o,d,t) × unit_cost(o,d)`` from planned flows.
    total_cost : float
        Equal to ``transportation_cost`` — the heuristic plans no inventory, so
        holding cost is absent from the planned objective.
    holding_cost : float
        Always ``0.0`` — the heuristic ignores holding costs at planning time.
        Realized holding cost is computed by ``actuals_evaluator``.
    unmet_demand_pct : float
        ``(Σ shortage / Σ actual_demand) × 100`` over the test window.
    infeasibility_rate : float
        Fraction of (origin, period) pairs where planned outflow exceeds
        ``daily_capacity``.  For the proportional heuristic this is always 0.0
        by construction; included for completeness and future heuristic variants.
    """

    flows: pl.DataFrame
    inventory: pl.DataFrame
    transportation_cost: float
    total_cost: float
    holding_cost: float
    unmet_demand_pct: float
    infeasibility_rate: float


def compute_lag1_forecast(
    demand_history: pl.DataFrame,
    test_periods: int,
) -> pl.DataFrame:
    """Return lag-1 demand forecasts for the last ``test_periods`` dates.

    For each destination ``d`` and each date ``t`` in the test window,
    ``forecast(d, t) = actual_demand(d, t-1)``.  The shift is computed over
    the full demand history (train + test) so that forecasts inside the test
    window correctly use the previous day's actual, matching the behaviour of
    ``NaiveForecaster`` inside the DILE pipeline.

    Parameters
    ----------
    demand_history : pl.DataFrame
        Full demand history ``[date, destination_id, demand]``, all 365 days.
    test_periods : int
        Number of trailing dates to return as the forecast window.

    Returns
    -------
    pl.DataFrame
        ``[destination_id, date, demand]`` with ``test_periods × n_destinations``
        rows.  Rows where the lag value is null (i.e. no prior-day observation)
        are dropped — this only affects the very first date in the history, which
        is never in the test window for reasonable ``test_periods`` values.
    """
    all_dates = sorted(demand_history["date"].unique().to_list())
    test_dates = set(all_dates[-test_periods:])

    lagged = (
        demand_history.sort(["destination_id", "date"])
        .with_columns(pl.col("demand").shift(1).over("destination_id").alias("demand"))
        .filter(pl.col("date").is_in(list(test_dates)))
        .drop_nulls("demand")
    )
    return lagged.select(["destination_id", "date", "demand"])


def _ts_to_map(df: pl.DataFrame) -> dict[tuple, float]:
    """Convert a [destination_id, date, demand] DataFrame to a (dest, date)->demand dict."""
    return {
        (r["destination_id"], r["date"]): float(r["demand"])
        for r in df.with_columns(pl.col("demand").fill_null(0.0)).to_dicts()
    }


def run_naive_allocation_heuristic(
    forecast_ts: pl.DataFrame,
    demand_history: pl.DataFrame,
    origins_df: pl.DataFrame,
    lanes_df: pl.DataFrame,
    destinations_df: pl.DataFrame,
) -> HeuristicResult:
    """Run the proportional capacity heuristic (O0 strategy).

    **Planning step** — for each period ``t``:

    .. code-block:: text

        share(d, t) = forecast(d, t) / Σ_d forecast(d, t)
        flow(o, d, t) = daily_capacity[o] × share(d, t)

    Every origin ships its full daily capacity, split proportionally by
    forecasted demand share.  Capacity constraints are satisfied by
    construction; no LP is involved.

    **Retroactive evaluation** — inventory and unmet demand are computed from
    the planned inflows and *actual* demand (not forecasts).

    Parameters
    ----------
    forecast_ts : pl.DataFrame
        Demand signal for the test window ``[destination_id, date, demand]``.
        For F0 scenarios this comes from ``compute_lag1_forecast``; for F1
        scenarios it comes from ``AggregatedForecastingResult.export_forecasts()``.
    demand_history : pl.DataFrame
        Full demand history ``[date, destination_id, demand]``. Rows outside
        the test window derived from ``forecast_ts`` are ignored.
    origins_df : pl.DataFrame
        ``[origin_id, daily_capacity]``.
    lanes_df : pl.DataFrame
        ``[origin_id, destination_id, unit_cost]``.
    destinations_df : pl.DataFrame
        ``[destination_id]`` optionally with ``holding_cost`` (used only for
        the retroactive inventory computation that feeds ``actuals_evaluator``).

    Returns
    -------
    HeuristicResult
    """
    test_dates: list[Date] = sorted(forecast_ts["date"].unique().to_list())
    origins: list[str] = sorted(origins_df["origin_id"].unique().to_list())
    destinations: list[str] = sorted(forecast_ts["destination_id"].unique().to_list())

    actual_demand_ts = demand_history.filter(pl.col("date").is_in(test_dates))

    capacity_map: dict[str, float] = dict(
        zip(origins_df["origin_id"].to_list(), origins_df["daily_capacity"].to_list())
    )
    unit_cost_map: dict[tuple[str, str], float] = {
        (r["origin_id"], r["destination_id"]): r["unit_cost"]
        for r in lanes_df.to_dicts()
    }

    forecast_map = _ts_to_map(forecast_ts)
    actual_map = _ts_to_map(actual_demand_ts)

    flow_rows: list[dict] = []
    inventory_rows: list[dict] = []

    total_transport_cost = 0.0
    total_unmet = 0.0
    total_demand = 0.0
    infeasibility_violations = 0
    total_origin_periods = len(origins) * len(test_dates)

    inv: dict[str, float] = {d: 0.0 for d in destinations}

    for t in test_dates:
        forecasts_t = {d: max(0.0, forecast_map.get((d, t), 0.0)) for d in destinations}
        total_forecast_t = sum(forecasts_t.values())

        if total_forecast_t > 0.0:
            share = {d: forecasts_t[d] / total_forecast_t for d in destinations}
        else:
            equal = 1.0 / len(destinations)
            share = {d: equal for d in destinations}

        inflow: dict[str, float] = {d: 0.0 for d in destinations}

        for o in origins:
            cap = capacity_map[o]
            outflow_o = 0.0
            for d in destinations:
                flow_val = cap * share[d]
                flow_rows.append(
                    {
                        "origin_id": o,
                        "destination_id": d,
                        "period": t,
                        "flow": flow_val,
                    }
                )
                total_transport_cost += flow_val * unit_cost_map.get((o, d), 0.0)
                inflow[d] += flow_val
                outflow_o += flow_val

            if outflow_o > cap + 1e-6:
                infeasibility_violations += 1

        for d in destinations:
            actual_d = actual_map.get((d, t), 0.0)
            available = inv[d] + inflow[d]
            shortage = max(0.0, actual_d - available)
            inv[d] = max(0.0, available - actual_d)

            total_demand += actual_d
            total_unmet += shortage

            inventory_rows.append(
                {"destination_id": d, "period": t, "inventory": inv[d]}
            )

    flows_df = pl.DataFrame(flow_rows).with_columns(
        pl.col("period").cast(pl.Date),
        pl.col("flow").cast(pl.Float64),
    )
    inventory_df = pl.DataFrame(inventory_rows).with_columns(
        pl.col("period").cast(pl.Date),
        pl.col("inventory").cast(pl.Float64),
    )

    unmet_demand_pct = (total_unmet / total_demand * 100.0) if total_demand > 0 else 0.0
    infeasibility_rate = (
        infeasibility_violations / total_origin_periods
        if total_origin_periods > 0
        else 0.0
    )

    return HeuristicResult(
        flows=flows_df,
        inventory=inventory_df,
        transportation_cost=total_transport_cost,
        total_cost=total_transport_cost,
        holding_cost=0.0,
        unmet_demand_pct=unmet_demand_pct,
        infeasibility_rate=infeasibility_rate,
    )
