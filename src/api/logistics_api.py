"""Concrete implementation of APIInterface for the logistics engine."""

from datetime import date

import polars as pl

from api.api_interface import APIInterface
from forecasting.pipeline.per_destination_pipeline import AggregatedPipelineResult
from forecasting.pipeline.pipeline_factory import create_per_destination_pipeline_from_config
from optimization import MultiPeriodOptimizer, MultiPeriodResult
from utils.config import PerDestinationConfig


class LogisticsAPI(APIInterface):
    """Concrete API implementation wiring PerDestinationPipeline and MultiPeriodOptimizer.

    Parameters
    ----------
    config : PerDestinationConfig
        Forecasting pipeline configuration (model names, train ratio, etc.).
    solver_name : str
        OR-Tools backend solver (default ``"GLOP"``).
    """

    def __init__(
        self,
        config: PerDestinationConfig,
        solver_name: str = "GLOP",
    ) -> None:
        self._config = config
        self._solver_name = solver_name

    def forecast(self, input_data: pl.DataFrame) -> AggregatedPipelineResult:
        """Run per-destination forecasting on historical demand data."""
        pipeline = create_per_destination_pipeline_from_config(self._config)
        return pipeline.run(input_data)

    def optimize(
        self,
        demand_ts: pl.DataFrame,
        origins_df: pl.DataFrame,
        lanes_df: pl.DataFrame,
        destinations_df: pl.DataFrame,
        planning_horizon: list[date],
        initial_inventory: dict[str, float] | None = None,
    ) -> MultiPeriodResult:
        """Solve the multi-period transportation LP."""
        optimizer = MultiPeriodOptimizer(solver_name=self._solver_name)
        return optimizer.solve(
            demand_ts=demand_ts,
            origins_df=origins_df,
            lanes_df=lanes_df,
            destinations_df=destinations_df,
            planning_horizon=planning_horizon,
            initial_inventory=initial_inventory,
        )
