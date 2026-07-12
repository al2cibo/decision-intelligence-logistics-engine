"""Unit tests for naive_heuristic: compute_lag1_forecast and run_naive_allocation_heuristic."""

from datetime import date, timedelta

import polars as pl
import pytest

from naive_heuristic import (
    HeuristicResult,
    compute_lag1_forecast,
    run_naive_allocation_heuristic,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_DATES = [date(2024, 1, 1) + timedelta(days=i) for i in range(5)]


@pytest.fixture
def demand_history_two_dests() -> pl.DataFrame:
    """5 dates × 2 destinations, demand = day_index + 1 for D1, day_index + 11 for D2."""
    rows = []
    for i, d in enumerate(_DATES):
        rows.append({"date": d, "destination_id": "D1", "demand": float(i + 1)})
        rows.append({"date": d, "destination_id": "D2", "demand": float(i + 11)})
    return pl.DataFrame(rows)


@pytest.fixture
def origins_df() -> pl.DataFrame:
    return pl.DataFrame({"origin_id": ["O1", "O2"], "daily_capacity": [60.0, 40.0]})


@pytest.fixture
def lanes_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "origin_id": ["O1", "O1", "O2", "O2"],
            "destination_id": ["D1", "D2", "D1", "D2"],
            "unit_cost": [1.0, 2.0, 3.0, 4.0],
        }
    )


@pytest.fixture
def destinations_df() -> pl.DataFrame:
    return pl.DataFrame({"destination_id": ["D1", "D2"]})


# ---------------------------------------------------------------------------
# compute_lag1_forecast — schema and shape
# ---------------------------------------------------------------------------


class TestComputeLag1ForecastSchema:
    def test_returns_three_columns(self, demand_history_two_dests):
        result = compute_lag1_forecast(demand_history_two_dests, test_periods=2)
        assert set(result.columns) == {"destination_id", "date", "demand"}

    def test_row_count(self, demand_history_two_dests):
        # test_periods=3, 2 destinations → 6 rows (assuming no null drops in test window)
        result = compute_lag1_forecast(demand_history_two_dests, test_periods=3)
        assert result.height == 6

    def test_dates_are_last_n_dates(self, demand_history_two_dests):
        result = compute_lag1_forecast(demand_history_two_dests, test_periods=2)
        returned_dates = set(result["date"].to_list())
        expected_dates = {_DATES[3], _DATES[4]}
        assert returned_dates == expected_dates

    def test_single_period(self, demand_history_two_dests):
        result = compute_lag1_forecast(demand_history_two_dests, test_periods=1)
        assert result.height == 2
        assert set(result["date"].to_list()) == {_DATES[4]}


# ---------------------------------------------------------------------------
# compute_lag1_forecast — lag-1 correctness
# ---------------------------------------------------------------------------


class TestComputeLag1ForecastValues:
    def test_lag1_values_for_d1(self, demand_history_two_dests):
        """forecast(D1, t) == actual_demand(D1, t-1)."""
        result = compute_lag1_forecast(demand_history_two_dests, test_periods=3)
        d1 = result.filter(pl.col("destination_id") == "D1").sort("date")
        # D1 demands are [1,2,3,4,5]; last 3 dates are _DATES[2..4]
        # lag-1: forecast at _DATES[2]=demand[_DATES[1]]=2, etc.
        assert d1["demand"].to_list() == pytest.approx([2.0, 3.0, 4.0])

    def test_lag1_values_for_d2(self, demand_history_two_dests):
        """forecast(D2, t) == actual_demand(D2, t-1)."""
        result = compute_lag1_forecast(demand_history_two_dests, test_periods=3)
        d2 = result.filter(pl.col("destination_id") == "D2").sort("date")
        # D2 demands are [11,12,13,14,15]; lag-1 for last 3 dates
        assert d2["demand"].to_list() == pytest.approx([12.0, 13.0, 14.0])

    def test_first_date_in_history_not_in_output(self):
        """If the test window would require a lag beyond the history, that row is dropped."""
        history = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "destination_id": ["D1", "D1"],
                "demand": [10.0, 20.0],
            }
        )
        # test_periods=2 covers both dates; lag for first date is null → dropped
        result = compute_lag1_forecast(history, test_periods=2)
        assert result.height == 1
        assert result["date"][0] == date(2024, 1, 2)
        assert result["demand"][0] == pytest.approx(10.0)

    def test_single_destination(self):
        dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(4)]
        history = pl.DataFrame(
            {
                "date": dates,
                "destination_id": ["D1"] * 4,
                "demand": [5.0, 10.0, 15.0, 20.0],
            }
        )
        result = compute_lag1_forecast(history, test_periods=2)
        result_sorted = result.sort("date")
        assert result_sorted["demand"].to_list() == pytest.approx([10.0, 15.0])


# ---------------------------------------------------------------------------
# run_naive_allocation_heuristic — result structure
# ---------------------------------------------------------------------------


class TestNaiveHeuristicResultStructure:
    def test_returns_heuristic_result_instance(
        self, demand_history_two_dests, origins_df, lanes_df, destinations_df
    ):
        forecast_ts = compute_lag1_forecast(demand_history_two_dests, test_periods=2)
        result = run_naive_allocation_heuristic(
            forecast_ts=forecast_ts,
            demand_history=demand_history_two_dests,
            origins_df=origins_df,
            lanes_df=lanes_df,
            destinations_df=destinations_df,
        )
        assert isinstance(result, HeuristicResult)

    def test_flows_schema(
        self, demand_history_two_dests, origins_df, lanes_df, destinations_df
    ):
        forecast_ts = compute_lag1_forecast(demand_history_two_dests, test_periods=2)
        result = run_naive_allocation_heuristic(
            forecast_ts=forecast_ts,
            demand_history=demand_history_two_dests,
            origins_df=origins_df,
            lanes_df=lanes_df,
            destinations_df=destinations_df,
        )
        assert set(result.flows.columns) == {
            "origin_id",
            "destination_id",
            "period",
            "flow",
        }

    def test_inventory_schema(
        self, demand_history_two_dests, origins_df, lanes_df, destinations_df
    ):
        forecast_ts = compute_lag1_forecast(demand_history_two_dests, test_periods=2)
        result = run_naive_allocation_heuristic(
            forecast_ts=forecast_ts,
            demand_history=demand_history_two_dests,
            origins_df=origins_df,
            lanes_df=lanes_df,
            destinations_df=destinations_df,
        )
        assert set(result.inventory.columns) == {
            "destination_id",
            "period",
            "inventory",
        }

    def test_flows_row_count(
        self, demand_history_two_dests, origins_df, lanes_df, destinations_df
    ):
        # 2 origins × 2 destinations × 2 test periods = 8 rows
        forecast_ts = compute_lag1_forecast(demand_history_two_dests, test_periods=2)
        result = run_naive_allocation_heuristic(
            forecast_ts=forecast_ts,
            demand_history=demand_history_two_dests,
            origins_df=origins_df,
            lanes_df=lanes_df,
            destinations_df=destinations_df,
        )
        assert result.flows.height == 8

    def test_inventory_row_count(
        self, demand_history_two_dests, origins_df, lanes_df, destinations_df
    ):
        # 2 destinations × 2 test periods = 4 rows
        forecast_ts = compute_lag1_forecast(demand_history_two_dests, test_periods=2)
        result = run_naive_allocation_heuristic(
            forecast_ts=forecast_ts,
            demand_history=demand_history_two_dests,
            origins_df=origins_df,
            lanes_df=lanes_df,
            destinations_df=destinations_df,
        )
        assert result.inventory.height == 4

    def test_holding_cost_is_zero(
        self, demand_history_two_dests, origins_df, lanes_df, destinations_df
    ):
        forecast_ts = compute_lag1_forecast(demand_history_two_dests, test_periods=2)
        result = run_naive_allocation_heuristic(
            forecast_ts=forecast_ts,
            demand_history=demand_history_two_dests,
            origins_df=origins_df,
            lanes_df=lanes_df,
            destinations_df=destinations_df,
        )
        assert result.holding_cost == 0.0

    def test_total_cost_equals_transportation_cost(
        self, demand_history_two_dests, origins_df, lanes_df, destinations_df
    ):
        forecast_ts = compute_lag1_forecast(demand_history_two_dests, test_periods=2)
        result = run_naive_allocation_heuristic(
            forecast_ts=forecast_ts,
            demand_history=demand_history_two_dests,
            origins_df=origins_df,
            lanes_df=lanes_df,
            destinations_df=destinations_df,
        )
        assert result.total_cost == pytest.approx(result.transportation_cost)

    def test_infeasibility_rate_is_zero_for_proportional_heuristic(
        self, demand_history_two_dests, origins_df, lanes_df, destinations_df
    ):
        forecast_ts = compute_lag1_forecast(demand_history_two_dests, test_periods=2)
        result = run_naive_allocation_heuristic(
            forecast_ts=forecast_ts,
            demand_history=demand_history_two_dests,
            origins_df=origins_df,
            lanes_df=lanes_df,
            destinations_df=destinations_df,
        )
        assert result.infeasibility_rate == 0.0

    def test_all_flows_non_negative(
        self, demand_history_two_dests, origins_df, lanes_df, destinations_df
    ):
        forecast_ts = compute_lag1_forecast(demand_history_two_dests, test_periods=2)
        result = run_naive_allocation_heuristic(
            forecast_ts=forecast_ts,
            demand_history=demand_history_two_dests,
            origins_df=origins_df,
            lanes_df=lanes_df,
            destinations_df=destinations_df,
        )
        assert (result.flows["flow"] >= 0.0).all()

    def test_all_inventory_non_negative(
        self, demand_history_two_dests, origins_df, lanes_df, destinations_df
    ):
        forecast_ts = compute_lag1_forecast(demand_history_two_dests, test_periods=2)
        result = run_naive_allocation_heuristic(
            forecast_ts=forecast_ts,
            demand_history=demand_history_two_dests,
            origins_df=origins_df,
            lanes_df=lanes_df,
            destinations_df=destinations_df,
        )
        assert (result.inventory["inventory"] >= 0.0).all()


# ---------------------------------------------------------------------------
# run_naive_allocation_heuristic — capacity constraints
# ---------------------------------------------------------------------------


class TestNaiveHeuristicCapacityConstraints:
    def test_origin_outflow_does_not_exceed_capacity(
        self, demand_history_two_dests, origins_df, lanes_df, destinations_df
    ):
        forecast_ts = compute_lag1_forecast(demand_history_two_dests, test_periods=3)
        result = run_naive_allocation_heuristic(
            forecast_ts=forecast_ts,
            demand_history=demand_history_two_dests,
            origins_df=origins_df,
            lanes_df=lanes_df,
            destinations_df=destinations_df,
        )
        capacity_map = dict(
            zip(
                origins_df["origin_id"].to_list(),
                origins_df["daily_capacity"].to_list(),
            )
        )
        per_origin_period = result.flows.group_by(["origin_id", "period"]).agg(
            pl.col("flow").sum().alias("total_flow")
        )
        for row in per_origin_period.to_dicts():
            cap = capacity_map[row["origin_id"]]
            assert row["total_flow"] <= cap + 1e-6, (
                f"{row['origin_id']} exceeded capacity on {row['period']}: "
                f"{row['total_flow']:.4f} > {cap}"
            )

    def test_total_outflow_per_period_equals_sum_of_capacities(
        self, demand_history_two_dests, origins_df, lanes_df, destinations_df
    ):
        """Proportional heuristic always ships full capacity every period."""
        forecast_ts = compute_lag1_forecast(demand_history_two_dests, test_periods=2)
        result = run_naive_allocation_heuristic(
            forecast_ts=forecast_ts,
            demand_history=demand_history_two_dests,
            origins_df=origins_df,
            lanes_df=lanes_df,
            destinations_df=destinations_df,
        )
        total_capacity = origins_df["daily_capacity"].sum()
        per_period = result.flows.group_by("period").agg(
            pl.col("flow").sum().alias("total_flow")
        )
        for row in per_period.to_dicts():
            assert row["total_flow"] == pytest.approx(total_capacity, abs=1e-6), (
                f"Period {row['period']}: total flow {row['total_flow']} "
                f"!= total capacity {total_capacity}"
            )


# ---------------------------------------------------------------------------
# run_naive_allocation_heuristic — proportional distribution
# ---------------------------------------------------------------------------


class TestNaiveHeuristicProportionalDistribution:
    def test_proportional_share_single_period(self):
        """With equal demand at D1 and D2, each destination gets exactly half the capacity."""
        history = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)] * 2,
                "destination_id": ["D1", "D1", "D2", "D2"],
                "demand": [10.0, 10.0, 10.0, 10.0],
            }
        )
        forecast_ts = pl.DataFrame(
            {
                "destination_id": ["D1", "D2"],
                "date": [date(2024, 1, 2), date(2024, 1, 2)],
                "demand": [10.0, 10.0],
            }
        )
        origins_df = pl.DataFrame({"origin_id": ["O1"], "daily_capacity": [100.0]})
        lanes_df = pl.DataFrame(
            {
                "origin_id": ["O1", "O1"],
                "destination_id": ["D1", "D2"],
                "unit_cost": [1.0, 1.0],
            }
        )
        destinations_df = pl.DataFrame({"destination_id": ["D1", "D2"]})

        result = run_naive_allocation_heuristic(
            forecast_ts=forecast_ts,
            demand_history=history,
            origins_df=origins_df,
            lanes_df=lanes_df,
            destinations_df=destinations_df,
        )
        d1_flow = result.flows.filter(pl.col("destination_id") == "D1")["flow"].sum()
        d2_flow = result.flows.filter(pl.col("destination_id") == "D2")["flow"].sum()
        assert d1_flow == pytest.approx(50.0, abs=1e-6)
        assert d2_flow == pytest.approx(50.0, abs=1e-6)

    def test_proportional_share_unequal_demand(self):
        """With D1 demand=3× D2's, D1 gets 3/4 of capacity, D2 gets 1/4."""
        history = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)] * 2,
                "destination_id": ["D1", "D1", "D2", "D2"],
                "demand": [30.0, 30.0, 10.0, 10.0],
            }
        )
        forecast_ts = pl.DataFrame(
            {
                "destination_id": ["D1", "D2"],
                "date": [date(2024, 1, 2), date(2024, 1, 2)],
                "demand": [30.0, 10.0],
            }
        )
        origins_df = pl.DataFrame({"origin_id": ["O1"], "daily_capacity": [80.0]})
        lanes_df = pl.DataFrame(
            {
                "origin_id": ["O1", "O1"],
                "destination_id": ["D1", "D2"],
                "unit_cost": [1.0, 1.0],
            }
        )
        destinations_df = pl.DataFrame({"destination_id": ["D1", "D2"]})

        result = run_naive_allocation_heuristic(
            forecast_ts=forecast_ts,
            demand_history=history,
            origins_df=origins_df,
            lanes_df=lanes_df,
            destinations_df=destinations_df,
        )
        d1_flow = result.flows.filter(pl.col("destination_id") == "D1")["flow"].sum()
        d2_flow = result.flows.filter(pl.col("destination_id") == "D2")["flow"].sum()
        assert d1_flow == pytest.approx(60.0, abs=1e-6)
        assert d2_flow == pytest.approx(20.0, abs=1e-6)

    def test_zero_total_forecast_uses_equal_shares(self):
        """When all forecasts are 0, capacity is split equally across destinations."""
        history = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)] * 2,
                "destination_id": ["D1", "D1", "D2", "D2"],
                "demand": [5.0, 5.0, 5.0, 5.0],
            }
        )
        forecast_ts = pl.DataFrame(
            {
                "destination_id": ["D1", "D2"],
                "date": [date(2024, 1, 2), date(2024, 1, 2)],
                "demand": [0.0, 0.0],
            }
        )
        origins_df = pl.DataFrame({"origin_id": ["O1"], "daily_capacity": [100.0]})
        lanes_df = pl.DataFrame(
            {
                "origin_id": ["O1", "O1"],
                "destination_id": ["D1", "D2"],
                "unit_cost": [1.0, 1.0],
            }
        )
        destinations_df = pl.DataFrame({"destination_id": ["D1", "D2"]})

        result = run_naive_allocation_heuristic(
            forecast_ts=forecast_ts,
            demand_history=history,
            origins_df=origins_df,
            lanes_df=lanes_df,
            destinations_df=destinations_df,
        )
        d1_flow = result.flows.filter(pl.col("destination_id") == "D1")["flow"].sum()
        d2_flow = result.flows.filter(pl.col("destination_id") == "D2")["flow"].sum()
        assert d1_flow == pytest.approx(50.0, abs=1e-6)
        assert d2_flow == pytest.approx(50.0, abs=1e-6)


# ---------------------------------------------------------------------------
# run_naive_allocation_heuristic — transportation cost
# ---------------------------------------------------------------------------


class TestNaiveHeuristicTransportationCost:
    def test_known_cost_single_origin_single_destination(self):
        """transport_cost = total_flow × unit_cost."""
        history = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "destination_id": ["D1", "D1"],
                "demand": [20.0, 20.0],
            }
        )
        forecast_ts = pl.DataFrame(
            {
                "destination_id": ["D1"],
                "date": [date(2024, 1, 2)],
                "demand": [20.0],
            }
        )
        origins_df = pl.DataFrame({"origin_id": ["O1"], "daily_capacity": [50.0]})
        lanes_df = pl.DataFrame(
            {"origin_id": ["O1"], "destination_id": ["D1"], "unit_cost": [3.0]}
        )
        destinations_df = pl.DataFrame({"destination_id": ["D1"]})

        result = run_naive_allocation_heuristic(
            forecast_ts=forecast_ts,
            demand_history=history,
            origins_df=origins_df,
            lanes_df=lanes_df,
            destinations_df=destinations_df,
        )
        # 1 period, capacity=50, all goes to D1, cost=3 → 150
        assert result.transportation_cost == pytest.approx(150.0, abs=1e-6)

    def test_missing_lane_unit_cost_treated_as_zero(self):
        """Lanes not present in lanes_df are treated as zero cost."""
        history = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)] * 2,
                "destination_id": ["D1", "D1", "D2", "D2"],
                "demand": [10.0, 10.0, 10.0, 10.0],
            }
        )
        forecast_ts = pl.DataFrame(
            {
                "destination_id": ["D1", "D2"],
                "date": [date(2024, 1, 2), date(2024, 1, 2)],
                "demand": [10.0, 10.0],
            }
        )
        origins_df = pl.DataFrame({"origin_id": ["O1"], "daily_capacity": [100.0]})
        # Only O1→D1 has a cost; O1→D2 is missing
        lanes_df = pl.DataFrame(
            {"origin_id": ["O1"], "destination_id": ["D1"], "unit_cost": [5.0]}
        )
        destinations_df = pl.DataFrame({"destination_id": ["D1", "D2"]})

        result = run_naive_allocation_heuristic(
            forecast_ts=forecast_ts,
            demand_history=history,
            origins_df=origins_df,
            lanes_df=lanes_df,
            destinations_df=destinations_df,
        )
        # 50 units to D1 at cost 5 = 250; 50 to D2 at cost 0 = 0
        assert result.transportation_cost == pytest.approx(250.0, abs=1e-6)


# ---------------------------------------------------------------------------
# run_naive_allocation_heuristic — unmet demand
# ---------------------------------------------------------------------------


class TestNaiveHeuristicUnmetDemand:
    def test_zero_unmet_when_capacity_exceeds_demand(self):
        """When capacity far exceeds demand, unmet_demand_pct is 0."""
        history = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "destination_id": ["D1", "D1"],
                "demand": [5.0, 5.0],
            }
        )
        forecast_ts = pl.DataFrame(
            {
                "destination_id": ["D1"],
                "date": [date(2024, 1, 2)],
                "demand": [5.0],
            }
        )
        origins_df = pl.DataFrame({"origin_id": ["O1"], "daily_capacity": [1000.0]})
        lanes_df = pl.DataFrame(
            {"origin_id": ["O1"], "destination_id": ["D1"], "unit_cost": [1.0]}
        )
        destinations_df = pl.DataFrame({"destination_id": ["D1"]})

        result = run_naive_allocation_heuristic(
            forecast_ts=forecast_ts,
            demand_history=history,
            origins_df=origins_df,
            lanes_df=lanes_df,
            destinations_df=destinations_df,
        )
        assert result.unmet_demand_pct == pytest.approx(0.0, abs=1e-6)

    def test_known_unmet_demand_pct(self):
        """When capacity = half of demand, unmet_demand_pct should be approximately 50%."""
        history = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "destination_id": ["D1", "D1"],
                "demand": [100.0, 100.0],
            }
        )
        forecast_ts = pl.DataFrame(
            {
                "destination_id": ["D1"],
                "date": [date(2024, 1, 2)],
                "demand": [100.0],
            }
        )
        # capacity=50, demand=100 over 1 period
        origins_df = pl.DataFrame({"origin_id": ["O1"], "daily_capacity": [50.0]})
        lanes_df = pl.DataFrame(
            {"origin_id": ["O1"], "destination_id": ["D1"], "unit_cost": [1.0]}
        )
        destinations_df = pl.DataFrame({"destination_id": ["D1"]})

        result = run_naive_allocation_heuristic(
            forecast_ts=forecast_ts,
            demand_history=history,
            origins_df=origins_df,
            lanes_df=lanes_df,
            destinations_df=destinations_df,
        )
        assert result.unmet_demand_pct == pytest.approx(50.0, abs=1e-6)

    def test_inventory_carries_over_to_next_period(self):
        """Excess inflow in period 1 reduces shortage in period 2."""
        # period 1: capacity=100, demand=20 → 80 units leftover inventory
        # period 2: capacity=100, demand=150 → available=180, fully met
        history = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
                "destination_id": ["D1"] * 3,
                "demand": [20.0, 20.0, 150.0],
            }
        )
        forecast_ts = pl.DataFrame(
            {
                "destination_id": ["D1", "D1"],
                "date": [date(2024, 1, 2), date(2024, 1, 3)],
                "demand": [20.0, 150.0],
            }
        )
        origins_df = pl.DataFrame({"origin_id": ["O1"], "daily_capacity": [100.0]})
        lanes_df = pl.DataFrame(
            {"origin_id": ["O1"], "destination_id": ["D1"], "unit_cost": [1.0]}
        )
        destinations_df = pl.DataFrame({"destination_id": ["D1"]})

        result = run_naive_allocation_heuristic(
            forecast_ts=forecast_ts,
            demand_history=history,
            origins_df=origins_df,
            lanes_df=lanes_df,
            destinations_df=destinations_df,
        )
        # period 1: inflow=100, demand=20, surplus→inventory=80
        # period 2: inflow=100+80=180, demand=150, no shortage
        assert result.unmet_demand_pct == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# run_naive_allocation_heuristic — negative forecast clamping
# ---------------------------------------------------------------------------


class TestNaiveHeuristicNegativeForecastClamping:
    def test_negative_forecast_clamped_to_zero_for_share_calculation(self):
        """Negative forecast values are clamped to 0 before computing demand shares.

        D1 has forecast=-5 (clamped to 0), D2 has forecast=10.
        Since D1 contributes 0 to total, all capacity goes to D2.
        """
        history = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)] * 2,
                "destination_id": ["D1", "D1", "D2", "D2"],
                "demand": [5.0, 5.0, 10.0, 10.0],
            }
        )
        forecast_ts = pl.DataFrame(
            {
                "destination_id": ["D1", "D2"],
                "date": [date(2024, 1, 2), date(2024, 1, 2)],
                "demand": [-5.0, 10.0],
            }
        )
        origins_df = pl.DataFrame({"origin_id": ["O1"], "daily_capacity": [100.0]})
        lanes_df = pl.DataFrame(
            {
                "origin_id": ["O1", "O1"],
                "destination_id": ["D1", "D2"],
                "unit_cost": [1.0, 1.0],
            }
        )
        destinations_df = pl.DataFrame({"destination_id": ["D1", "D2"]})

        result = run_naive_allocation_heuristic(
            forecast_ts=forecast_ts,
            demand_history=history,
            origins_df=origins_df,
            lanes_df=lanes_df,
            destinations_df=destinations_df,
        )
        d1_flow = result.flows.filter(pl.col("destination_id") == "D1")["flow"].sum()
        d2_flow = result.flows.filter(pl.col("destination_id") == "D2")["flow"].sum()
        assert d1_flow == pytest.approx(0.0, abs=1e-6)
        assert d2_flow == pytest.approx(100.0, abs=1e-6)

    def test_all_negative_forecasts_fall_back_to_equal_shares(self):
        """When all forecasts clamp to 0, equal-share fallback is used (same as zero-forecast)."""
        history = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)] * 2,
                "destination_id": ["D1", "D1", "D2", "D2"],
                "demand": [5.0, 5.0, 5.0, 5.0],
            }
        )
        forecast_ts = pl.DataFrame(
            {
                "destination_id": ["D1", "D2"],
                "date": [date(2024, 1, 2), date(2024, 1, 2)],
                "demand": [-10.0, -3.0],
            }
        )
        origins_df = pl.DataFrame({"origin_id": ["O1"], "daily_capacity": [80.0]})
        lanes_df = pl.DataFrame(
            {
                "origin_id": ["O1", "O1"],
                "destination_id": ["D1", "D2"],
                "unit_cost": [1.0, 1.0],
            }
        )
        destinations_df = pl.DataFrame({"destination_id": ["D1", "D2"]})

        result = run_naive_allocation_heuristic(
            forecast_ts=forecast_ts,
            demand_history=history,
            origins_df=origins_df,
            lanes_df=lanes_df,
            destinations_df=destinations_df,
        )
        d1_flow = result.flows.filter(pl.col("destination_id") == "D1")["flow"].sum()
        d2_flow = result.flows.filter(pl.col("destination_id") == "D2")["flow"].sum()
        assert d1_flow == pytest.approx(40.0, abs=1e-6)
        assert d2_flow == pytest.approx(40.0, abs=1e-6)


# ---------------------------------------------------------------------------
# run_naive_allocation_heuristic — multi-origin cost
# ---------------------------------------------------------------------------


class TestNaiveHeuristicMultiOriginCost:
    def test_known_cost_two_origins_two_destinations(self):
        """Verify transport cost with 2 origins, 2 destinations, and distinct unit costs.

        Setup (1 period):
          O1: capacity=60, O2: capacity=40 → total=100
          D1 forecast=30, D2 forecast=70 → shares: D1=30%, D2=70%
          Flows:
            O1→D1: 60×0.3=18,  O1→D2: 60×0.7=42
            O2→D1: 40×0.3=12,  O2→D2: 40×0.7=28
          Unit costs: O1→D1=1, O1→D2=2, O2→D1=3, O2→D2=4
          Transport cost: 18×1 + 42×2 + 12×3 + 28×4 = 18+84+36+112 = 250
        """
        history = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)] * 2,
                "destination_id": ["D1", "D1", "D2", "D2"],
                "demand": [30.0, 30.0, 70.0, 70.0],
            }
        )
        forecast_ts = pl.DataFrame(
            {
                "destination_id": ["D1", "D2"],
                "date": [date(2024, 1, 2), date(2024, 1, 2)],
                "demand": [30.0, 70.0],
            }
        )
        origins_df = pl.DataFrame(
            {"origin_id": ["O1", "O2"], "daily_capacity": [60.0, 40.0]}
        )
        lanes_df = pl.DataFrame(
            {
                "origin_id": ["O1", "O1", "O2", "O2"],
                "destination_id": ["D1", "D2", "D1", "D2"],
                "unit_cost": [1.0, 2.0, 3.0, 4.0],
            }
        )
        destinations_df = pl.DataFrame({"destination_id": ["D1", "D2"]})

        result = run_naive_allocation_heuristic(
            forecast_ts=forecast_ts,
            demand_history=history,
            origins_df=origins_df,
            lanes_df=lanes_df,
            destinations_df=destinations_df,
        )
        assert result.transportation_cost == pytest.approx(250.0, abs=1e-6)


# ---------------------------------------------------------------------------
# run_naive_allocation_heuristic — destinations derived from forecast_ts
# ---------------------------------------------------------------------------


class TestNaiveHeuristicDestinationSource:
    def test_destinations_from_forecast_ts_not_destinations_df(self):
        """Destinations in the plan come from forecast_ts, not destinations_df.

        D3 appears in destinations_df but not in forecast_ts — it should receive
        no flow and not appear in the output at all.
        """
        history = pl.DataFrame(
            {
                "date": [date(2024, 1, 1), date(2024, 1, 2)],
                "destination_id": ["D1", "D1"],
                "demand": [10.0, 10.0],
            }
        )
        forecast_ts = pl.DataFrame(
            {
                "destination_id": ["D1"],
                "date": [date(2024, 1, 2)],
                "demand": [10.0],
            }
        )
        origins_df = pl.DataFrame({"origin_id": ["O1"], "daily_capacity": [50.0]})
        lanes_df = pl.DataFrame(
            {"origin_id": ["O1"], "destination_id": ["D1"], "unit_cost": [1.0]}
        )
        destinations_df = pl.DataFrame({"destination_id": ["D1", "D3"]})

        result = run_naive_allocation_heuristic(
            forecast_ts=forecast_ts,
            demand_history=history,
            origins_df=origins_df,
            lanes_df=lanes_df,
            destinations_df=destinations_df,
        )
        dest_ids_in_flows = set(result.flows["destination_id"].unique().to_list())
        assert "D3" not in dest_ids_in_flows
        assert "D1" in dest_ids_in_flows
