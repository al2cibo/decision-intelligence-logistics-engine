"""Unit tests for OptimizerInterface."""

from datetime import date

import polars as pl
import pytest

from optimization.optimizer_interface import OptimizerInterface
from optimization.optimizer import OptimizationResult
from optimization.multi_period_result import MultiPeriodResult


# ---------------------------------------------------------------------------
# Fixtures: minimal valid inputs
# ---------------------------------------------------------------------------


@pytest.fixture
def single_period_demand() -> pl.DataFrame:
    """Single-period demand format: [destination_id, demand]."""
    return pl.DataFrame(
        {
            "destination_id": ["D1", "D2"],
            "demand": [10.0, 20.0],
        }
    )


@pytest.fixture
def multi_period_demand() -> pl.DataFrame:
    """Multi-period demand format: [destination_id, date, demand]."""
    return pl.DataFrame(
        {
            "destination_id": ["D1", "D1", "D1", "D2", "D2", "D2"],
            "date": [
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 3),
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 3),
            ],
            "demand": [10.0, 20.0, 30.0, 5.0, 15.0, 25.0],
        }
    )


@pytest.fixture
def origins_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "origin_id": ["O1", "O2"],
            "daily_capacity": [100.0, 100.0],
        }
    )


@pytest.fixture
def lanes_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "origin_id": ["O1", "O2"],
            "destination_id": ["D1", "D2"],
            "unit_cost": [5.0, 3.0],
        }
    )


@pytest.fixture
def destinations_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "destination_id": ["D1", "D2"],
            "holding_cost": [1.0, 2.0],
        }
    )


@pytest.fixture
def planning_horizon() -> list[date]:
    return [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]


# ---------------------------------------------------------------------------
# Tests: Invalid mode raises ValueError
# ---------------------------------------------------------------------------


class TestInvalidMode:
    """Test that invalid mode raises ValueError with 'Supported modes'."""

    def test_invalid_mode_raises_value_error(self):
        with pytest.raises(ValueError, match="Supported modes"):
            OptimizerInterface(mode="invalid")

    def test_invalid_mode_empty_string(self):
        with pytest.raises(ValueError, match="Supported modes"):
            OptimizerInterface(mode="")


# ---------------------------------------------------------------------------
# Tests: Single-period mode with single-period demand delegates correctly
# ---------------------------------------------------------------------------


class TestSinglePeriodDirect:
    """Test single-period mode with single-period demand delegates correctly."""

    def test_single_period_direct_returns_optimization_result(
        self, single_period_demand, origins_df, lanes_df
    ):
        interface = OptimizerInterface(mode="single")
        result = interface.solve(
            demand=single_period_demand,
            origins_df=origins_df,
            lanes_df=lanes_df,
        )
        assert isinstance(result, OptimizationResult)

    def test_single_period_direct_has_correct_flows(
        self, single_period_demand, origins_df, lanes_df
    ):
        interface = OptimizerInterface(mode="single")
        result = interface.solve(
            demand=single_period_demand,
            origins_df=origins_df,
            lanes_df=lanes_df,
        )
        # Flows should satisfy demand
        assert result.flows.height > 0
        assert result.total_cost > 0


# ---------------------------------------------------------------------------
# Tests: Single-period mode with multi-period demand aggregates to mean
# ---------------------------------------------------------------------------


class TestSinglePeriodAggregation:
    """Test single-period mode with multi-period demand aggregates to mean."""

    def test_multi_period_demand_aggregated_to_mean(
        self, multi_period_demand, origins_df, lanes_df
    ):
        interface = OptimizerInterface(mode="single")
        result = interface.solve(
            demand=multi_period_demand,
            origins_df=origins_df,
            lanes_df=lanes_df,
        )
        assert isinstance(result, OptimizationResult)

    def test_aggregation_produces_correct_cost(
        self, multi_period_demand, origins_df, lanes_df
    ):
        """Verify the result matches solving with mean demand directly."""
        interface = OptimizerInterface(mode="single")
        result = interface.solve(
            demand=multi_period_demand,
            origins_df=origins_df,
            lanes_df=lanes_df,
        )

        # Mean demand: D1 = mean(10, 20, 30) = 20, D2 = mean(5, 15, 25) = 15
        mean_demand = pl.DataFrame(
            {
                "destination_id": ["D1", "D2"],
                "demand": [20.0, 15.0],
            }
        )
        direct_result = interface.solve(
            demand=mean_demand,
            origins_df=origins_df,
            lanes_df=lanes_df,
        )

        assert abs(result.total_cost - direct_result.total_cost) < 1e-6


# ---------------------------------------------------------------------------
# Tests: Multi-period mode dispatches to MultiPeriodOptimizer
# ---------------------------------------------------------------------------


class TestMultiPeriodDispatch:
    """Test multi-period mode dispatches to MultiPeriodOptimizer."""

    def test_multi_period_returns_multi_period_result(
        self, multi_period_demand, origins_df, lanes_df, destinations_df, planning_horizon
    ):
        interface = OptimizerInterface(mode="multi")
        result = interface.solve(
            demand=multi_period_demand,
            origins_df=origins_df,
            lanes_df=lanes_df,
            destinations_df=destinations_df,
            planning_horizon=planning_horizon,
        )
        assert isinstance(result, MultiPeriodResult)

    def test_multi_period_result_has_flows_and_inventory(
        self, multi_period_demand, origins_df, lanes_df, destinations_df, planning_horizon
    ):
        interface = OptimizerInterface(mode="multi")
        result = interface.solve(
            demand=multi_period_demand,
            origins_df=origins_df,
            lanes_df=lanes_df,
            destinations_df=destinations_df,
            planning_horizon=planning_horizon,
        )
        assert result.flows.height > 0
        assert result.inventory.height > 0
        assert result.total_cost > 0


# ---------------------------------------------------------------------------
# Tests: Multi-period mode without planning_horizon raises error
# ---------------------------------------------------------------------------


class TestMultiPeriodMissingParams:
    """Test multi-period mode raises errors for missing required params."""

    def test_multi_period_without_planning_horizon_raises(
        self, multi_period_demand, origins_df, lanes_df, destinations_df
    ):
        interface = OptimizerInterface(mode="multi")
        with pytest.raises(ValueError, match="planning_horizon is required"):
            interface.solve(
                demand=multi_period_demand,
                origins_df=origins_df,
                lanes_df=lanes_df,
                destinations_df=destinations_df,
                planning_horizon=None,
            )

    def test_multi_period_without_destinations_df_raises(
        self, multi_period_demand, origins_df, lanes_df, planning_horizon
    ):
        interface = OptimizerInterface(mode="multi")
        with pytest.raises(ValueError, match="destinations_df is required"):
            interface.solve(
                demand=multi_period_demand,
                origins_df=origins_df,
                lanes_df=lanes_df,
                destinations_df=None,
                planning_horizon=planning_horizon,
            )
