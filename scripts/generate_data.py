"""Generate synthetic logistics datasets and write them to parquet files.

Usage (from any directory):
    python scripts/generate_data.py
"""

import itertools
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def generate_synthetic_logistics_data(
    output_dir: Path,
    n_origins: int,
    n_destinations: int,
    start_date: datetime,
    end_date: datetime,
    seed: int = 42,
    include_holding_cost: bool = False,
    holding_cost_range: tuple[float, float] = (0.5, 2.0),
) -> None:
    """Generate a synthetic logistics dataset and write it to parquet files.

    Produces four files in ``output_dir``:

    - ``origins.parquet``        — origin_id, daily_capacity
    - ``destinations.parquet``   — destination_id[, holding_cost]
    - ``lanes.parquet``          — origin_id, destination_id, unit_cost
    - ``demand_history.parquet`` — date, destination_id, demand

    Demand is generated with weekly seasonality (15% reduction on weekends),
    a slow upward trend (~0.15% per day), occasional promotional spikes
    (8% probability, +25% demand), and Gaussian noise.

    Parameters
    ----------
    output_dir : Path
        Directory in which to write the four parquet files.
    n_origins : int
        Number of origin warehouses.
    n_destinations : int
        Number of destination stores/DCs.
    start_date : datetime
        First date in the demand history.
    end_date : datetime
        Last date (inclusive) in the demand history.
    seed : int
        NumPy random seed for reproducibility. Defaults to ``42``.
    include_holding_cost : bool
        When ``True``, adds a ``holding_cost`` column to ``destinations.parquet``
        with per-destination costs drawn uniformly from ``holding_cost_range``.
        Defaults to ``False``.
    holding_cost_range : tuple[float, float]
        ``(min, max)`` range for per-destination holding costs (only used when
        ``include_holding_cost=True``). Defaults to ``(0.5, 2.0)``.
    """
    rng = np.random.default_rng(seed)

    origin_ids = [f"O{i:02d}" for i in range(1, n_origins + 1)]
    destination_ids = [f"D{i:02d}" for i in range(1, n_destinations + 1)]

    origins = pl.DataFrame(
        {
            "origin_id": origin_ids,
            "daily_capacity": rng.integers(80, 180, size=n_origins).tolist(),
        }
    )

    if include_holding_cost:
        lo, hi = holding_cost_range
        holding_costs = [round(float(rng.uniform(lo, hi)), 2) for _ in destination_ids]
        destinations = pl.DataFrame(
            {
                "destination_id": destination_ids,
                "holding_cost": holding_costs,
            }
        )
    else:
        destinations = pl.DataFrame({"destination_id": destination_ids})

    lanes = pl.DataFrame(
        [
            {
                "origin_id": origin_id,
                "destination_id": destination_id,
                "unit_cost": float(rng.integers(5, 25)),
            }
            for origin_id, destination_id in itertools.product(
                origin_ids, destination_ids
            )
        ]
    )

    base_demand = dict(
        zip(destination_ids, rng.integers(20, 70, size=n_destinations).tolist())
    )

    dates = pl.date_range(start=start_date, end=end_date, interval="1d", eager=True)

    demand_rows = []
    for destination_id in destination_ids:
        base = base_demand[destination_id]
        for i, date in enumerate(dates):
            is_weekend = date.weekday() >= 5
            weekly_factor = 0.85 if is_weekend else 1.0
            trend_factor = 1.0 + 0.0015 * i
            promo_factor = 1.25 if rng.random() < 0.08 else 1.0
            noise = rng.normal(0, 4)
            demand = max(
                0.0, base * weekly_factor * trend_factor * promo_factor + noise
            )
            demand_rows.append(
                {
                    "date": date,
                    "destination_id": destination_id,
                    "demand": round(float(demand), 2),
                }
            )

    demand_history = pl.DataFrame(demand_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    origins.write_parquet(output_dir / "origins.parquet")
    destinations.write_parquet(output_dir / "destinations.parquet")
    lanes.write_parquet(output_dir / "lanes.parquet")
    demand_history.write_parquet(output_dir / "demand_history.parquet")
    print(f"Data written to {output_dir.resolve()}")


def _synthetic1() -> None:
    generate_synthetic_logistics_data(
        output_dir=_PROJECT_ROOT / "data" / "synthetic1",
        n_origins=4,
        n_destinations=8,
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 6, 30),
        seed=42,
    )


def _synthetic2() -> None:
    generate_synthetic_logistics_data(
        output_dir=_PROJECT_ROOT / "data" / "synthetic2",
        n_origins=3,
        n_destinations=6,
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 6, 30),
        seed=42,
    )


def _experiment_synthetic_v1() -> None:
    """Regenerate experiments/datasets/synthetic_v1 with holding costs.

    Same random seed and size as the original synthetic_v1 (3 origins,
    6 destinations, 2025-01-01 to 2025-06-30, seed=42), but now includes
    a ``holding_cost`` column in destinations.parquet so the LP cost
    breakdown is non-degenerate in experiment results.
    """
    generate_synthetic_logistics_data(
        output_dir=_PROJECT_ROOT / "experiments" / "datasets" / "synthetic_v1",
        n_origins=3,
        n_destinations=6,
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 6, 30),
        seed=42,
        include_holding_cost=True,
        holding_cost_range=(0.5, 2.0),
    )


def _experiment_synthetic_v2() -> None:
    """Generate experiments/datasets/synthetic_v2.

    365 days of daily demand (2024-01-01 → 2024-12-31) across 6 destinations
    and 3 origins. Each destination has a distinct demand pattern so that
    per-destination model selection has a genuine reason to choose different
    models:

    - D1: strong weekly seasonality, level ~80, no trend, low noise
    - D2: flat, no seasonality, level ~60, low noise
    - D3: upward trend (+0.1/day), weak seasonality, level ~50, moderate noise
    - D4: strong weekly seasonality + upward trend, level ~90, moderate noise
    - D5: high noise, no pattern, level ~70
    - D6: weekly seasonality + mild downward trend, level ~100, low noise

    Origins, destinations, lanes, and holding costs are all hand-assigned
    (no random generation) for full reproducibility and meaningful cost
    structure. Only demand noise uses numpy.random.seed(42).
    """
    import numpy as np

    output_dir = _PROJECT_ROOT / "experiments" / "datasets" / "synthetic_v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    start = datetime(2024, 1, 1)
    dates = pl.date_range(
        start=start,
        end=datetime(2024, 12, 30),  # 2024 is a leap year; end on Dec 30 for exactly 365 days
        interval="1d",
        eager=True,
    )
    n_days = len(dates)
    day_of_week = [d.weekday() for d in dates]  # 0=Mon … 6=Sun

    # --- origins ---
    origins = pl.DataFrame({
        "origin_id": ["O1", "O2", "O3"],
        "daily_capacity": [300.0, 250.0, 200.0],
    })

    # --- destinations ---
    destinations = pl.DataFrame({
        "destination_id": ["D1", "D2", "D3", "D4", "D5", "D6"],
        "holding_cost":   [0.50, 0.75, 1.00, 0.40, 0.60, 0.90],
    })

    # --- lanes (hand-assigned, fully connected 3×6) ---
    lane_costs = {
        # O1: cheapest (2.0–4.0)
        ("O1", "D1"): 2.0, ("O1", "D2"): 2.5, ("O1", "D3"): 3.0,
        ("O1", "D4"): 2.2, ("O1", "D5"): 3.5, ("O1", "D6"): 4.0,
        # O2: mid-range (3.5–6.0)
        ("O2", "D1"): 3.5, ("O2", "D2"): 4.0, ("O2", "D3"): 4.5,
        ("O2", "D4"): 5.0, ("O2", "D5"): 5.5, ("O2", "D6"): 6.0,
        # O3: most expensive (5.0–8.0)
        ("O3", "D1"): 5.0, ("O3", "D2"): 5.5, ("O3", "D3"): 6.0,
        ("O3", "D4"): 6.5, ("O3", "D5"): 7.0, ("O3", "D6"): 8.0,
    }
    lanes = pl.DataFrame([
        {"origin_id": o, "destination_id": d, "unit_cost": c}
        for (o, d), c in lane_costs.items()
    ])

    # --- demand history ---
    def seasonal_mult(pattern: list[float]) -> np.ndarray:
        return np.array([pattern[dow] for dow in day_of_week])

    t = np.arange(n_days)

    # D1: strong weekly seasonality, level ~80, no trend, low noise
    d1 = 80.0 * seasonal_mult([1.20, 1.15, 1.00, 0.90, 0.85, 0.70, 0.75]) \
        + np.random.normal(0, 4, n_days)

    # D2: flat, no seasonality, level ~60, low noise
    d2 = 60.0 + np.random.normal(0, 3, n_days)

    # D3: upward trend, weak seasonality, level ~50, moderate noise
    d3 = (50.0 + 0.1 * t) * seasonal_mult([1.05, 1.02, 1.00, 0.98, 0.97, 0.99, 1.00]) \
        + np.random.normal(0, 7, n_days)

    # D4: strong weekly seasonality + upward trend, level ~90, moderate noise
    d4 = (90.0 + 0.08 * t) * seasonal_mult([1.25, 1.20, 1.05, 0.95, 0.85, 0.70, 0.80]) \
        + np.random.normal(0, 8, n_days)

    # D5: high noise, no pattern, level ~70
    d5 = 70.0 + np.random.normal(0, 20, n_days)

    # D6: weekly seasonality + mild downward trend, level ~100, low noise
    d6 = (100.0 - 0.05 * t) * seasonal_mult([1.10, 1.05, 1.00, 0.95, 0.90, 0.95, 1.00]) \
        + np.random.normal(0, 5, n_days)

    demand_rows = []
    for dest_id, raw in [("D1", d1), ("D2", d2), ("D3", d3),
                          ("D4", d4), ("D5", d5), ("D6", d6)]:
        clipped = np.clip(raw, 0, None).round(1)
        for date, val in zip(dates, clipped):
            demand_rows.append({"date": date, "destination_id": dest_id, "demand": float(val)})

    demand_history = pl.DataFrame(demand_rows).with_columns(
        pl.col("date").cast(pl.Date)
    )

    # --- validation ---
    assert len(demand_history) == 2190, f"Expected 2190 rows, got {len(demand_history)}"
    assert len(lanes) == 18
    assert "holding_cost" in destinations.columns
    assert demand_history["demand"].min() >= 0.0
    for dest in ["D1", "D2", "D3", "D4", "D5", "D6"]:
        n = demand_history.filter(pl.col("destination_id") == dest).height
        assert n == 365, f"{dest}: expected 365 rows, got {n}"

    # --- write ---
    origins.write_parquet(output_dir / "origins.parquet")
    destinations.write_parquet(output_dir / "destinations.parquet")
    lanes.write_parquet(output_dir / "lanes.parquet")
    demand_history.write_parquet(output_dir / "demand_history.parquet")

    # --- summary ---
    print(f"synthetic_v2 written to {output_dir.resolve()}")
    print(f"  origins.parquet        : {len(origins)} rows")
    print(f"  destinations.parquet   : {len(destinations)} rows")
    print(f"  lanes.parquet          : {len(lanes)} rows")
    print(f"  demand_history.parquet : {len(demand_history)} rows")
    print("\nholding_cost per destination:")
    for row in destinations.iter_rows(named=True):
        print(f"  {row['destination_id']}: {row['holding_cost']}")
    print("\nDemand statistics per destination:")
    stats = (
        demand_history
        .group_by("destination_id")
        .agg([
            pl.col("demand").mean().round(2).alias("mean"),
            pl.col("demand").std().round(2).alias("std"),
            pl.col("demand").min().round(2).alias("min"),
            pl.col("demand").max().round(2).alias("max"),
        ])
        .sort("destination_id")
    )
    print(stats)


if __name__ == "__main__":
    _experiment_synthetic_v1()
    _experiment_synthetic_v2()
