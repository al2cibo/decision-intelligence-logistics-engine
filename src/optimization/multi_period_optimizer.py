"""Multi-period transportation optimization using OR-Tools linear programming solvers."""

import logging
from datetime import date

import polars as pl
from ortools.linear_solver import pywraplp

from .multi_period_result import MultiPeriodResult

logger = logging.getLogger(__name__)


class MultiPeriodOptimizer:
    """Multi-period minimum-cost transportation LP with inventory tracking.

    Parameters
    ----------
    solver_name : str, optional
        Backend solver to use (default ``"GLOP"``).  Must be one of
        :pyattr:`SUPPORTED_SOLVERS`.
    """

    SUPPORTED_SOLVERS = {"GLOP", "CBC"}
    MAX_VARIABLES = 1_000_000

    def __init__(self, solver_name: str = "GLOP") -> None:
        """Initialize with solver backend selection."""
        if solver_name not in self.SUPPORTED_SOLVERS:
            raise ValueError(
                f"Unsupported solver '{solver_name}'. "
                f"Supported solvers: {sorted(self.SUPPORTED_SOLVERS)}"
            )
        self._solver_name = solver_name

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
        # --- validation (order matters) -----------------------------------
        self._validate_not_empty(demand_ts, origins_df, lanes_df, planning_horizon)
        self._validate_demand_schema(demand_ts)
        self._validate_costs(lanes_df, destinations_df)
        self._validate_capacities(origins_df)
        self._validate_origins_in_lanes(origins_df, lanes_df)
        self._validate_initial_inventory(initial_inventory)
        self._validate_variable_count(lanes_df, demand_ts, planning_horizon)

        # --- feasibility pre-checks ---------------------------------------
        self._check_unreachable_destinations(demand_ts, lanes_df)
        self._check_capacity_feasibility(demand_ts, origins_df, planning_horizon)

        # --- demand preprocessing -----------------------------------------
        demand_ts = self._preprocess_demand(demand_ts, planning_horizon)

        # --- LP formulation -----------------------------------------------
        solver = pywraplp.Solver.CreateSolver(self._solver_name)
        if solver is None:
            raise ValueError(
                f"Could not create solver with backend '{self._solver_name}'"
            )

        # Resolve initial inventory defaults
        if initial_inventory is None:
            initial_inventory = {}

        # Determine if holding costs are present
        has_holding_cost = "holding_cost" in destinations_df.columns

        # Build lookup structures
        # Lanes: list of (origin_id, destination_id, unit_cost)
        lanes_list = lanes_df.select(
            "origin_id", "destination_id", "unit_cost"
        ).to_dicts()

        # Origins: dict of origin_id -> daily_capacity
        capacity_map: dict[str, float] = dict(
            zip(
                origins_df["origin_id"].to_list(),
                origins_df["daily_capacity"].to_list(),
            )
        )

        # Destinations with holding costs (if applicable)
        holding_cost_map: dict[str, float] = {}
        if has_holding_cost:
            holding_cost_map = dict(
                zip(
                    destinations_df["destination_id"].to_list(),
                    destinations_df["holding_cost"].to_list(),
                )
            )

        # Demand lookup: (destination_id, date) -> demand value
        demand_map: dict[tuple[str, date], float] = {}
        for row in demand_ts.to_dicts():
            demand_map[(row["destination_id"], row["date"])] = row["demand"]

        # Unique destinations that appear in demand (need inventory variables)
        destinations = sorted(demand_ts["destination_id"].unique().to_list())

        # Unique origins from lanes
        origins = sorted(origins_df["origin_id"].unique().to_list())

        # Lanes grouped by destination: destination_id -> list of (origin_id, unit_cost)
        lanes_by_dest: dict[str, list[tuple[str, float]]] = {}
        # Lanes grouped by origin: origin_id -> list of (destination_id, unit_cost)
        lanes_by_origin: dict[str, list[tuple[str, float]]] = {}
        for lane in lanes_list:
            o_id = lane["origin_id"]
            d_id = lane["destination_id"]
            cost = lane["unit_cost"]
            lanes_by_dest.setdefault(d_id, []).append((o_id, cost))
            lanes_by_origin.setdefault(o_id, []).append((d_id, cost))

        # --- Create decision variables ------------------------------------

        # Flow variables: flow[o, d, t] >= 0
        flow_vars: dict[tuple[str, str, date], pywraplp.Variable] = {}
        for lane in lanes_list:
            o_id = lane["origin_id"]
            d_id = lane["destination_id"]
            for t in planning_horizon:
                var_name = f"flow_{o_id}_{d_id}_{t}"
                flow_vars[(o_id, d_id, t)] = solver.NumVar(
                    0.0, solver.infinity(), var_name
                )

        # Inventory variables: inventory[d, t] >= 0
        inv_vars: dict[tuple[str, date], pywraplp.Variable] = {}
        for d_id in destinations:
            for t in planning_horizon:
                var_name = f"inv_{d_id}_{t}"
                inv_vars[(d_id, t)] = solver.NumVar(
                    0.0, solver.infinity(), var_name
                )

        # --- Add constraints ----------------------------------------------

        # Inventory balance constraints: for each destination d and period t
        for d_id in destinations:
            for t_idx, t in enumerate(planning_horizon):
                # Determine previous inventory
                if t_idx == 0:
                    prev_inv = initial_inventory.get(d_id, 0.0)
                else:
                    prev_inv = None  # will use variable

                # Inflow: sum of flow[o, d, t] for all origins with a lane to d
                inflow_terms = []
                if d_id in lanes_by_dest:
                    for o_id, _ in lanes_by_dest[d_id]:
                        if (o_id, d_id, t) in flow_vars:
                            inflow_terms.append(flow_vars[(o_id, d_id, t)])

                # Demand for this destination at this period
                demand_val = demand_map.get((d_id, t), 0.0)

                # Constraint: inventory[d,t] = prev_inv + inflow - demand
                # Rearranged: inventory[d,t] - inflow + demand - prev_inv = 0
                ct = solver.Constraint(0.0, 0.0, f"inv_bal_{d_id}_{t}")
                ct.SetCoefficient(inv_vars[(d_id, t)], 1.0)
                for flow_var in inflow_terms:
                    ct.SetCoefficient(flow_var, -1.0)

                if t_idx == 0:
                    # inventory[d,0] - inflow = initial_inv - demand
                    # (from: inventory[d,0] = initial_inv + inflow - demand)
                    ct.SetBounds(
                        prev_inv - demand_val, prev_inv - demand_val
                    )
                else:
                    # inventory[d,t] - inventory[d,t-1] - inflow + demand = 0
                    prev_t = planning_horizon[t_idx - 1]
                    ct.SetCoefficient(inv_vars[(d_id, prev_t)], -1.0)
                    ct.SetBounds(-demand_val, -demand_val)

        # Capacity constraints: for each origin o and period t
        for o_id in origins:
            for t in planning_horizon:
                cap = capacity_map[o_id]
                ct = solver.Constraint(
                    0.0, cap, f"cap_{o_id}_{t}"
                )
                if o_id in lanes_by_origin:
                    for d_id, _ in lanes_by_origin[o_id]:
                        if (o_id, d_id, t) in flow_vars:
                            ct.SetCoefficient(flow_vars[(o_id, d_id, t)], 1.0)

        # --- Build objective function -------------------------------------
        objective = solver.Objective()
        objective.SetMinimization()

        # Transportation costs: sum of unit_cost[o,d] * flow[o,d,t]
        for lane in lanes_list:
            o_id = lane["origin_id"]
            d_id = lane["destination_id"]
            cost = lane["unit_cost"]
            for t in planning_horizon:
                objective.SetCoefficient(flow_vars[(o_id, d_id, t)], cost)

        # Holding costs (optional): sum of holding_cost[d] * inventory[d,t]
        if has_holding_cost:
            for d_id in destinations:
                h_cost = holding_cost_map.get(d_id, 0.0)
                if h_cost > 0:
                    for t in planning_horizon:
                        objective.SetCoefficient(inv_vars[(d_id, t)], h_cost)

        # --- Solve --------------------------------------------------------
        status = solver.Solve()

        if status != pywraplp.Solver.OPTIMAL:
            status_map = {
                pywraplp.Solver.FEASIBLE: "FEASIBLE",
                pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
                pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
                pywraplp.Solver.ABNORMAL: "ABNORMAL",
                pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
            }
            status_str = status_map.get(status, f"UNKNOWN({status})")
            raise ValueError(
                f"Solver did not find an optimal solution. "
                f"Status: {status_str}"
            )

        # --- Solution extraction (task 5.2) -------------------------------
        # Extract total cost from objective value
        total_cost = solver.Objective().Value()

        # Extract flows DataFrame: only include rows where flow > 1e-6
        flow_records: list[dict] = []
        for (o_id, d_id, t), var in flow_vars.items():
            val = var.solution_value()
            if val > 1e-6:
                flow_records.append(
                    {
                        "origin_id": o_id,
                        "destination_id": d_id,
                        "period": t,
                        "flow": val,
                    }
                )

        if flow_records:
            flows_df = pl.DataFrame(flow_records).cast(
                {
                    "origin_id": pl.Utf8,
                    "destination_id": pl.Utf8,
                    "period": pl.Date,
                    "flow": pl.Float64,
                }
            )
        else:
            # Edge case: no flows exceed threshold → empty DataFrame with correct schema
            flows_df = pl.DataFrame(
                schema={
                    "origin_id": pl.Utf8,
                    "destination_id": pl.Utf8,
                    "period": pl.Date,
                    "flow": pl.Float64,
                }
            )

        # Extract inventory DataFrame: all destination × period combinations
        inv_records: list[dict] = []
        for d_id in destinations:
            for t in planning_horizon:
                inv_records.append(
                    {
                        "destination_id": d_id,
                        "period": t,
                        "inventory": inv_vars[(d_id, t)].solution_value(),
                    }
                )

        inventory_df = pl.DataFrame(inv_records).cast(
            {
                "destination_id": pl.Utf8,
                "period": pl.Date,
                "inventory": pl.Float64,
            }
        )

        return MultiPeriodResult(
            flows=flows_df,
            inventory=inventory_df,
            total_cost=total_cost,
        )

    # ------------------------------------------------------------------
    # Input validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_not_empty(
        demand_ts: pl.DataFrame,
        origins_df: pl.DataFrame,
        lanes_df: pl.DataFrame,
        planning_horizon: list[date],
    ) -> None:
        """Raise ``ValueError`` if any required input is empty."""
        if demand_ts.is_empty():
            raise ValueError("no demand data available")
        if origins_df.is_empty():
            raise ValueError("Origins DataFrame is empty")
        if lanes_df.is_empty():
            raise ValueError("Lanes DataFrame is empty")
        if len(planning_horizon) == 0:
            raise ValueError("Planning horizon contains zero periods")

    @staticmethod
    def _validate_demand_schema(demand_ts: pl.DataFrame) -> None:
        """Raise ``ValueError`` if demand_ts is missing required columns."""
        required_columns = {"destination_id", "date", "demand"}
        missing = required_columns - set(demand_ts.columns)
        if missing:
            raise ValueError(
                f"Demand time series missing required columns: {sorted(missing)}"
            )

    @staticmethod
    def _validate_costs(
        lanes_df: pl.DataFrame,
        destinations_df: pl.DataFrame,
    ) -> None:
        """Raise ``ValueError`` if any cost values are negative."""
        # Check unit_cost in lanes
        if "unit_cost" in lanes_df.columns:
            negative_costs = lanes_df.filter(pl.col("unit_cost") < 0)
            if not negative_costs.is_empty():
                invalid_rows = negative_costs.select(
                    "origin_id", "destination_id", "unit_cost"
                ).to_dicts()
                raise ValueError(
                    f"Negative unit_cost values found: {invalid_rows}"
                )

        # Check holding_cost in destinations (only if column exists)
        if "holding_cost" in destinations_df.columns:
            negative_holding = destinations_df.filter(pl.col("holding_cost") < 0)
            if not negative_holding.is_empty():
                invalid_rows = negative_holding.select(
                    "destination_id", "holding_cost"
                ).to_dicts()
                raise ValueError(
                    f"Negative holding_cost values found: {invalid_rows}"
                )

    @staticmethod
    def _validate_capacities(origins_df: pl.DataFrame) -> None:
        """Raise ``ValueError`` if any origin has non-positive daily_capacity."""
        if "daily_capacity" in origins_df.columns:
            invalid = origins_df.filter(pl.col("daily_capacity") <= 0)
            if not invalid.is_empty():
                invalid_rows = invalid.select(
                    "origin_id", "daily_capacity"
                ).to_dicts()
                raise ValueError(
                    f"Non-positive daily_capacity values found: {invalid_rows}"
                )

    @staticmethod
    def _validate_origins_in_lanes(
        origins_df: pl.DataFrame,
        lanes_df: pl.DataFrame,
    ) -> None:
        """Raise ``ValueError`` if lanes reference origins not in origins_df."""
        origin_ids = set(origins_df["origin_id"].to_list())
        lane_origin_ids = set(lanes_df["origin_id"].to_list())
        missing_origins = sorted(lane_origin_ids - origin_ids)
        if missing_origins:
            raise ValueError(
                f"Origins referenced in lanes but missing from origins_df: "
                f"{missing_origins}"
            )

    @staticmethod
    def _validate_initial_inventory(
        initial_inventory: dict[str, float] | None,
    ) -> None:
        """Raise ``ValueError`` if any initial inventory value is negative."""
        if initial_inventory is None:
            return
        for dest_id, value in initial_inventory.items():
            if value < 0:
                raise ValueError(
                    f"Negative initial inventory for destination '{dest_id}': "
                    f"{value}"
                )

    @staticmethod
    def _validate_variable_count(
        lanes_df: pl.DataFrame,
        demand_ts: pl.DataFrame,
        planning_horizon: list[date],
    ) -> None:
        """Raise ``ValueError`` if variable count exceeds MAX_VARIABLES."""
        n_periods = len(planning_horizon)
        n_lanes = len(lanes_df)
        n_destinations = demand_ts["destination_id"].n_unique()

        flow_vars = n_lanes * n_periods
        inventory_vars = n_destinations * n_periods
        total_vars = flow_vars + inventory_vars

        if total_vars > MultiPeriodOptimizer.MAX_VARIABLES:
            raise ValueError(
                f"Variable count ({total_vars}) exceeds maximum limit "
                f"({MultiPeriodOptimizer.MAX_VARIABLES}). "
                f"Flow variables: {flow_vars}, "
                f"inventory variables: {inventory_vars}."
            )

    # ------------------------------------------------------------------
    # Feasibility pre-checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_unreachable_destinations(
        demand_ts: pl.DataFrame,
        lanes_df: pl.DataFrame,
    ) -> None:
        """Raise ``ValueError`` if any demanded destination has no lane serving it."""
        demanded_destinations = set(demand_ts["destination_id"].unique().to_list())
        served_destinations = set(lanes_df["destination_id"].unique().to_list())
        unreachable = sorted(demanded_destinations - served_destinations)
        if unreachable:
            raise ValueError(
                f"Unreachable destinations (no lane serves them): {unreachable}"
            )

    @staticmethod
    def _check_capacity_feasibility(
        demand_ts: pl.DataFrame,
        origins_df: pl.DataFrame,
        planning_horizon: list[date],
    ) -> None:
        """Raise ``ValueError`` if total capacity < total demand.

        Total capacity is the sum of daily_capacity for all origins multiplied
        by the number of periods in the planning horizon.
        Total demand is the sum of all demand values (nulls treated as zero).
        """
        n_periods = len(planning_horizon)
        total_capacity = origins_df["daily_capacity"].sum() * n_periods
        total_demand = demand_ts["demand"].fill_null(0).sum()

        if total_capacity < total_demand:
            shortfall = total_demand - total_capacity
            raise ValueError(
                f"Insufficient total capacity: total demand = {total_demand}, "
                f"total capacity = {total_capacity}, shortfall = {shortfall}"
            )

    # ------------------------------------------------------------------
    # Demand preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess_demand(
        demand_ts: pl.DataFrame,
        planning_horizon: list[date],
    ) -> pl.DataFrame:
        """Preprocess demand time series before LP formulation.

        Steps:
        1. Filter out rows with dates not in planning_horizon
        2. Fill null demand values with 0
        3. Deduplicate by summing demand for duplicate (destination_id, date) pairs

        Missing (destination, period) pairs are treated as zero demand during
        LP formulation (no explicit rows added here).

        Parameters
        ----------
        demand_ts : pl.DataFrame
            Raw demand time series ``[destination_id, date, demand]``.
        planning_horizon : list[date]
            Ordered list of dates representing valid time periods.

        Returns
        -------
        pl.DataFrame
            Preprocessed demand with schema ``[destination_id, date, demand]``.
        """
        horizon_set = set(planning_horizon)

        # 1. Filter out rows with dates outside planning_horizon
        demand_ts = demand_ts.filter(pl.col("date").is_in(list(horizon_set)))

        # 2. Fill null demand values with 0
        demand_ts = demand_ts.with_columns(pl.col("demand").fill_null(0))

        # 3. Deduplicate: sum demand for duplicate (destination_id, date) pairs
        demand_ts = demand_ts.group_by("destination_id", "date").agg(
            pl.col("demand").sum()
        )

        return demand_ts
