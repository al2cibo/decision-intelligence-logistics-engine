"""Transportation optimization using OR-Tools linear programming solvers."""

import logging
from dataclasses import dataclass

import polars as pl
from ortools.linear_solver import pywraplp

from optimization.validation import (
    check_capacity_feasibility,
    check_unreachable_destinations,
    validate_columns,
    validate_not_empty,
)

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of a transportation optimization solve.

    Attributes
    ----------
    flows : pl.DataFrame
        Schema ``origin_id | destination_id | flow`` — only lanes with
        positive flow values.
    total_cost : float
        The minimised total shipping cost.
    """

    flows: pl.DataFrame
    total_cost: float


class Optimizer:
    """Minimum-cost transportation LP solver backed by OR-Tools.

    Parameters
    ----------
    solver_name : str, optional
        Backend solver to use (default ``"GLOP"``).  Must be one of
        :pyattr:`SUPPORTED_SOLVERS`.
    """

    SUPPORTED_SOLVERS = {"GLOP", "CBC"}

    def __init__(self, solver_name: str = "GLOP") -> None:
        if solver_name not in self.SUPPORTED_SOLVERS:
            raise ValueError(
                f"Unsupported solver '{solver_name}'. "
                f"Supported solvers: {sorted(self.SUPPORTED_SOLVERS)}"
            )
        self._solver_name = solver_name

    def solve(
        self,
        demand_df: pl.DataFrame,
        origins_df: pl.DataFrame,
        lanes_df: pl.DataFrame,
    ) -> OptimizationResult:
        """Formulate and solve the transportation LP.

        Parameters
        ----------
        demand_df : pl.DataFrame
            Schema ``destination_id | demand``.
        origins_df : pl.DataFrame
            Schema ``origin_id | daily_capacity``.
        lanes_df : pl.DataFrame
            Schema ``origin_id | destination_id | cost``.

        Returns
        -------
        OptimizationResult
            Flows (positive only) and total cost.

        Raises
        ------
        ValueError
            On invalid inputs, unreachable destinations, insufficient
            daily_capacity, or infeasible / unbounded solver status.
        """
        # --- validation ---------------------------------------------------
        validate_not_empty(
            **{
                "Demand DataFrame is empty \u2014 no demand data available": demand_df,
                "Origins DataFrame is empty \u2014 no origin data available": origins_df,
                "Lanes DataFrame is empty \u2014 no lane data available": lanes_df,
            }
        )
        validate_columns(
            demand_df, {"destination_id", "demand"}, "Demand DataFrame"
        )
        validate_columns(
            origins_df, {"origin_id", "daily_capacity"}, "Origins DataFrame"
        )
        validate_columns(
            lanes_df, {"origin_id", "destination_id", "unit_cost"}, "Lanes DataFrame"
        )
        check_unreachable_destinations(demand_df, lanes_df)
        total_demand = demand_df["demand"].sum()
        total_capacity = origins_df["daily_capacity"].sum()
        check_capacity_feasibility(total_demand, total_capacity)

        # --- build lookup dicts -------------------------------------------
        demand_map: dict[str, float] = dict(
            zip(
                demand_df["destination_id"].to_list(),
                demand_df["demand"].to_list(),
            )
        )
        daily_capacity_map: dict[str, float] = dict(
            zip(
                origins_df["origin_id"].to_list(),
                origins_df["daily_capacity"].to_list(),
            )
        )

        n_origins = len(daily_capacity_map)
        n_destinations = len(demand_map)
        n_lanes = len(lanes_df)

        logger.info(
            "Formulating LP: %d origins, %d destinations, %d lanes",
            n_origins,
            n_destinations,
            n_lanes,
        )

        # --- create solver ------------------------------------------------
        solver = pywraplp.Solver.CreateSolver(self._solver_name)
        if solver is None:
            raise ValueError(f"Could not create solver '{self._solver_name}'")

        # --- decision variables  x_od >= 0 --------------------------------
        x: dict[tuple[str, str], pywraplp.Variable] = {}
        lane_costs: dict[tuple[str, str], float] = {}

        for row in lanes_df.iter_rows(named=True):
            o = row["origin_id"]
            d = row["destination_id"]
            c = row["unit_cost"]
            x[(o, d)] = solver.NumVar(0.0, solver.infinity(), f"x_{o}_{d}")
            lane_costs[(o, d)] = c

        # --- objective: minimise Σ c_od · x_od ----------------------------
        objective = solver.Objective()
        for (o, d), var in x.items():
            objective.SetCoefficient(var, lane_costs[(o, d)])
        objective.SetMinimization()

        # --- demand constraints: Σ_o x_od >= dem_d  for each d ------------
        for d, dem in demand_map.items():
            ct = solver.Constraint(dem, solver.infinity(), f"demand_{d}")
            for (o2, d2), var in x.items():
                if d2 == d:
                    ct.SetCoefficient(var, 1.0)

        # --- daily_capacity constraints: Σ_d x_od <= cap_o  for each o ---------
        for o, cap in daily_capacity_map.items():
            ct = solver.Constraint(0.0, cap, f"daily_capacity_{o}")
            for (o2, d2), var in x.items():
                if o2 == o:
                    ct.SetCoefficient(var, 1.0)

        # --- solve --------------------------------------------------------
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            total_cost = solver.Objective().Value()
            logger.info("Solver status: OPTIMAL, total cost: %.4f", total_cost)
        elif status == pywraplp.Solver.INFEASIBLE:
            logger.warning("Solver status: INFEASIBLE")
            raise ValueError(
                "Solver returned INFEASIBLE — the problem has no feasible solution"
            )
        elif status == pywraplp.Solver.UNBOUNDED:
            logger.warning("Solver status: UNBOUNDED")
            raise ValueError("Solver returned UNBOUNDED — the problem is unbounded")
        else:
            logger.warning("Solver status: %s", status)
            raise ValueError(
                f"Solver did not find an optimal solution (status={status})"
            )

        # --- extract flows ---------------------------------------
        rows: list[dict[str, object]] = []
        for (o, d), var in x.items():
            flow = var.solution_value()
            if flow > 0:
                rows.append({"origin_id": o, "destination_id": d, "flow": flow})

        flows_df = pl.DataFrame(
            rows,
            schema={
                "origin_id": pl.Utf8,
                "destination_id": pl.Utf8,
                "flow": pl.Float64,
            },
        )

        return OptimizationResult(flows=flows_df, total_cost=total_cost)
