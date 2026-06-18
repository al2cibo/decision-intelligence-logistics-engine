"""Multi-period transportation optimization using OR-Tools linear programming solvers."""

import logging
from datetime import date

import polars as pl
from ortools.linear_solver import pywraplp

from .model_builder import (
    build_lookups,
    create_variables,
    add_constraints,
    set_objective,
)
from .preprocessing import preprocess_demand
from .result import MultiPeriodResult
from .solution_extractor import (
    extract_flows,
    extract_holding_cost,
    extract_inventory,
    extract_transportation_cost,
)
from .validation import MAX_VARIABLES, check_feasibility, validate_inputs

logger = logging.getLogger(__name__)

_STATUS_NAMES = {
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}


class MultiPeriodOptimizer:
    """Multi-period minimum-cost transportation LP with inventory tracking.

    Parameters
    ----------
    solver_name : str, optional
        Backend solver to use (default ``"GLOP"``). Must be one of
        :pyattr:`SUPPORTED_SOLVERS`.
    max_variables : int, optional
        Upper bound on the total LP decision-variable count
        (``n_lanes × n_periods + n_destinations × n_periods``). Problems
        that exceed this limit raise ``ValueError`` before LP construction to
        prevent out-of-memory conditions. Defaults to ``MAX_VARIABLES``
        (1 000 000). Increase for large-scale problems on capable hardware;
        decrease to enforce tighter memory limits in constrained environments.
    """

    SUPPORTED_SOLVERS = {"GLOP", "CBC"}
    MAX_VARIABLES = MAX_VARIABLES

    def __init__(
        self, solver_name: str = "GLOP", max_variables: int = MAX_VARIABLES
    ) -> None:
        if solver_name not in self.SUPPORTED_SOLVERS:
            raise ValueError(
                f"Unsupported solver '{solver_name}'. "
                f"Supported solvers: {sorted(self.SUPPORTED_SOLVERS)}"
            )
        if max_variables < 1:
            raise ValueError(
                f"max_variables must be a positive integer, got {max_variables}"
            )
        self._solver_name = solver_name
        self._max_variables = max_variables

    def solve(
        self,
        demand_ts: pl.DataFrame,
        origins_df: pl.DataFrame,
        lanes_df: pl.DataFrame,
        destinations_df: pl.DataFrame,
        planning_horizon: list[date],
        initial_inventory: dict[str, float] | None = None,
    ) -> MultiPeriodResult:
        """Formulate and solve the multi-period transportation LP.

        Parameters
        ----------
        demand_ts : pl.DataFrame
            Schema ``[destination_id, date, demand]``.
        origins_df : pl.DataFrame
            Schema ``[origin_id, daily_capacity]``.
        lanes_df : pl.DataFrame
            Schema ``[origin_id, destination_id, unit_cost]``.
        destinations_df : pl.DataFrame
            Schema ``[destination_id, ...optional holding_cost]``.
        planning_horizon : list[date]
            Ordered list of dates representing time periods.
        initial_inventory : dict[str, float] | None, optional
            Initial inventory per destination. Defaults to zero for all.

        Returns
        -------
        MultiPeriodResult
            Time-indexed flows, inventory levels, and total cost.

        Raises
        ------
        ValueError
            On invalid inputs, infeasible problems, or solver failures.
        """
        validate_inputs(
            demand_ts,
            origins_df,
            lanes_df,
            destinations_df,
            planning_horizon,
            initial_inventory,
            max_variables=self._max_variables,
        )
        check_feasibility(demand_ts, origins_df, lanes_df, planning_horizon)

        demand_ts = preprocess_demand(demand_ts, planning_horizon)

        solver = pywraplp.Solver.CreateSolver(self._solver_name)
        if solver is None:
            raise ValueError(
                f"Could not create solver with backend '{self._solver_name}'"
            )

        if initial_inventory is None:
            initial_inventory = {}

        lookups = build_lookups(demand_ts, origins_df, lanes_df, destinations_df)
        flow_vars, inv_vars = create_variables(solver, lookups, planning_horizon)
        add_constraints(
            solver, lookups, flow_vars, inv_vars, planning_horizon, initial_inventory
        )
        set_objective(solver, lookups, flow_vars, inv_vars, planning_horizon)

        status = solver.Solve()
        if status != pywraplp.Solver.OPTIMAL:
            status_str = _STATUS_NAMES.get(status, f"UNKNOWN({status})")
            raise ValueError(
                f"Solver did not find an optimal solution. Status: {status_str}"
            )

        total_cost = solver.Objective().Value()
        flows_df = extract_flows(flow_vars)
        inventory_df = extract_inventory(
            inv_vars, lookups.destinations, planning_horizon
        )
        transportation_cost = extract_transportation_cost(flow_vars, lookups.lanes_list)
        holding_cost = extract_holding_cost(inv_vars, lookups.holding_cost_map)

        return MultiPeriodResult(
            flows=flows_df,
            inventory=inventory_df,
            total_cost=total_cost,
            transportation_cost=transportation_cost,
            holding_cost=holding_cost,
        )

    @staticmethod
    def _preprocess_demand(
        demand_ts: pl.DataFrame, planning_horizon: list[date]
    ) -> pl.DataFrame:
        """Backward-compatible alias for :func:`preprocessing.preprocess_demand`."""
        return preprocess_demand(demand_ts, planning_horizon)
