"""Multi-period optimization result dataclass."""

from dataclasses import dataclass

import polars as pl


@dataclass
class MultiPeriodResult:
    """Result of a multi-period transportation optimization solve.

    Attributes
    ----------
    flows : pl.DataFrame
        Schema [origin_id: Utf8, destination_id: Utf8, period: Date, flow: Float64]
        Only flows exceeding 1e-6 threshold.
    inventory : pl.DataFrame
        Schema [destination_id: Utf8, period: Date, inventory: Float64]
        One row per destination per period (including zero inventory).
    total_cost : float
        The minimised total cost (transportation + holding).
    """

    flows: pl.DataFrame
    inventory: pl.DataFrame
    total_cost: float
