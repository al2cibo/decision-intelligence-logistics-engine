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
) -> None:
    """Generate a synthetic logistics dataset and write it to parquet files.

    Produces four files in ``output_dir``:

    - ``origins.parquet``        — origin_id, daily_capacity
    - ``destinations.parquet``   — destination_id
    - ``lanes.parquet``          — origin_id, destination_id, unit_cost
    - ``demand_history.parquet`` — date, destination_id, demand

    Demand is generated with weekly seasonality (15% reduction on weekends),
    a slow upward trend (~0.15% per day), occasional promotional spikes
    (8% probability, +25% demand), and Gaussian noise.
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


if __name__ == "__main__":
    _synthetic2()
