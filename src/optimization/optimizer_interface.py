"""Unified entry point for single-period and multi-period optimization."""

from __future__ import annotations

from datetime import date

import polars as pl

from .multi_period_optimizer import MultiPeriodOptimizer
from .multi_period_result import MultiPeriodResult
from .optimizer import Optimizer, OptimizationResult


class OptimizerInterface:
    """Unified entry point for single-period and multi-period optimization.

    Dispatches to either the existing single-period ``Optimizer`` or the
    ``MultiPeriodOptimizer`` based on the configured mode.

    Parameters
    ----------
    mode : str, optional
        Optimization mode — ``"single"`` or ``"multi"`` (default ``"single"``).
    solver_name : str, optional
        Backend solver to use (default ``"GLOP"``).
    """

    SUPPORTED_MODES = {"single", "multi"}

    def __init__(self, mode: str = "single", solver_name: str = "GLOP") -> None:
        """Initialize with mode and solver selection."""
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. "
                f"Supported modes: {sorted(self.SUPPORTED_MODES)}"
            )
        self._mode = mode
        self._solver_name = solver_name

    def solve(
        self,
        demand: pl.DataFrame,
        origins_df: pl.DataFrame,
        lanes_df: pl.DataFrame,
        destinations_df: pl.DataFrame | None = None,
        planning_horizon: list[date] | None = None,
        initial_inventory: dict[str, float] | None = None,
    ) -> OptimizationResult | MultiPeriodResult:
        """Dispatch to the appropriate solver based on mode.

        Parameters
        ----------
        demand : pl.DataFrame
            Demand data. For single-period mode: ``[destination_id, demand]``
            or ``[destination_id, date, demand]``. For multi-period mode:
            ``[destination_id, date, demand]``.
        origins_df : pl.DataFrame
            Schema ``[origin_id, daily_capacity]``.
        lanes_df : pl.DataFrame
            Schema ``[origin_id, destination_id, unit_cost]``.
        destinations_df : pl.DataFrame | None, optional
            Schema ``[destination_id, ...]``. Required for multi-period mode.
        planning_horizon : list[date] | None, optional
            Ordered list of dates. Required for multi-period mode.
        initial_inventory : dict[str, float] | None, optional
            Initial inventory per destination. Only used in multi-period mode.

        Returns
        -------
        OptimizationResult | MultiPeriodResult
            Result from the appropriate solver.

        Raises
        ------
        ValueError
            If required parameters for the selected mode are missing.
        """
        if self._mode == "single":
            return self._solve_single(demand, origins_df, lanes_df)
        else:
            return self._solve_multi(
                demand,
                origins_df,
                lanes_df,
                destinations_df,
                planning_horizon,
                initial_inventory,
            )

    def _solve_single(
        self,
        demand: pl.DataFrame,
        origins_df: pl.DataFrame,
        lanes_df: pl.DataFrame,
    ) -> OptimizationResult:
        """Handle single-period mode dispatch.

        If demand has a ``date`` column (multi-period format), aggregate to
        mean demand per destination before delegating to the single-period
        optimizer. Otherwise delegate directly.
        """
        if "date" in demand.columns:
            # Aggregate multi-period demand to mean per destination
            demand_df = demand.group_by("destination_id").agg(
                pl.col("demand").mean()
            )
        else:
            demand_df = demand

        optimizer = Optimizer(solver_name=self._solver_name)
        return optimizer.solve(demand_df, origins_df, lanes_df)

    def _solve_multi(
        self,
        demand: pl.DataFrame,
        origins_df: pl.DataFrame,
        lanes_df: pl.DataFrame,
        destinations_df: pl.DataFrame | None,
        planning_horizon: list[date] | None,
        initial_inventory: dict[str, float] | None,
    ) -> MultiPeriodResult:
        """Handle multi-period mode dispatch.

        Validates that required parameters are provided, then delegates
        to ``MultiPeriodOptimizer``.
        """
        if planning_horizon is None:
            raise ValueError(
                "planning_horizon is required for multi-period mode"
            )
        if destinations_df is None:
            raise ValueError(
                "destinations_df is required for multi-period mode"
            )

        optimizer = MultiPeriodOptimizer(solver_name=self._solver_name)
        return optimizer.solve(
            demand_ts=demand,
            origins_df=origins_df,
            lanes_df=lanes_df,
            destinations_df=destinations_df,
            planning_horizon=planning_horizon,
            initial_inventory=initial_inventory,
        )
