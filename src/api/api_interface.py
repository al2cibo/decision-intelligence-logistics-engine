"""Abstract base class defining the API layer contract."""

from abc import ABC, abstractmethod
from datetime import date

import polars as pl

from forecasting.pipeline.per_destination_pipeline import AggregatedPipelineResult
from optimization import MultiPeriodResult


class APIInterface(ABC):
    """Abstract interface for the API layer.

    Subclasses must implement ``forecast`` and ``optimize`` to provide
    concrete API behaviour for the logistics engine.
    """

    @abstractmethod
    def forecast(self, input_data: pl.DataFrame) -> AggregatedPipelineResult:
        """Run per-destination forecasting on historical demand data.

        Parameters
        ----------
        input_data : pl.DataFrame
            Historical demand data with schema [date, destination_id, demand].

        Returns
        -------
        AggregatedPipelineResult
            Per-destination outcomes: selected model, metrics, and forecast values.
        """
        ...

    @abstractmethod
    def optimize(
        self,
        demand_ts: pl.DataFrame,
        origins_df: pl.DataFrame,
        lanes_df: pl.DataFrame,
        destinations_df: pl.DataFrame,
        planning_horizon: list[date],
        initial_inventory: dict[str, float] | None = None,
    ) -> MultiPeriodResult:
        """Solve a multi-period minimum-cost transportation problem.

        Parameters
        ----------
        demand_ts : pl.DataFrame
            Time-indexed demand with schema [destination_id, date, demand].
        origins_df : pl.DataFrame
            Origin data with schema [origin_id, daily_capacity].
        lanes_df : pl.DataFrame
            Lane data with schema [origin_id, destination_id, unit_cost].
        destinations_df : pl.DataFrame
            Destination data with schema [destination_id, ...].
        planning_horizon : list[date]
            Ordered list of dates to optimise over.
        initial_inventory : dict[str, float] | None
            Starting inventory per destination (defaults to zero).

        Returns
        -------
        MultiPeriodResult
            Time-indexed flows, inventory levels, and total cost.
        """
        ...
