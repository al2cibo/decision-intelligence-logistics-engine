"""Synthetic data generation for pipeline demonstrations."""

from datetime import date, timedelta

import numpy as np
import polars as pl


def generate_synthetic_destination_data(
    destinations: list[str],
    n_days: int = 90,
    start_date: date | None = None,
) -> pl.DataFrame:
    """Generate synthetic demand data with distinct patterns per destination.

    Each destination gets a different demand profile:

    - D01: steady demand with weekly seasonality
    - D02: trending upward with noise
    - D03: high volatility with occasional spikes
    - D04: low, stable demand

    Parameters
    ----------
    destinations : list[str]
        List of destination identifiers to generate data for.
    n_days : int, optional
        Number of days of data to generate, by default 90.
    start_date : date or None, optional
        The first date in the generated series. Defaults to 2024-01-01
        when None.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame with columns ``date`` (pl.Date),
        ``destination_id`` (str), and ``demand`` (pl.Float64).
    """
    if start_date is None:
        start_date = date(2024, 1, 1)

    rng = np.random.default_rng(seed=42)
    rows = []

    for dest in destinations:
        for day_offset in range(n_days):
            current_date = start_date + timedelta(days=day_offset)
            day_of_week = current_date.weekday()

            if dest == "D01":
                # Weekly seasonality: higher on weekdays
                base = 100 + (20 if day_of_week < 5 else -10)
                demand = base + rng.normal(0, 5)
            elif dest == "D02":
                # Upward trend
                base = 50 + day_offset * 0.5
                demand = base + rng.normal(0, 8)
            elif dest == "D03":
                # High volatility with spikes
                base = 80
                spike = 50 if rng.random() < 0.1 else 0
                demand = base + spike + rng.normal(0, 15)
            else:
                # Low stable demand
                demand = 30 + rng.normal(0, 3)

            rows.append({
                "date": current_date,
                "destination_id": dest,
                "demand": max(0.0, float(demand)),
            })

    return pl.DataFrame(rows).cast({"date": pl.Date, "demand": pl.Float64})
