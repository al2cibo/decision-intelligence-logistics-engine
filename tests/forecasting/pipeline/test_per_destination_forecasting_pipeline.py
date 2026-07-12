"""Unit tests for PerDestinationForecastingPipeline."""

import math
from datetime import date, timedelta

import polars as pl
import pytest

from forecasting.results.forecast_result import ForecastResult, TimePeriod
from forecasting.registry.model_registry import ModelRegistry
from forecasting.evaluation.per_destination_model_selector import SelectionResult
from forecasting.pipeline.per_destination_forecasting_pipeline import (
    AggregatedForecastingResult,
    DestinationOutcome,
    PerDestinationForecastingPipeline,
)

# --- Helpers ---


def _make_registry_with_naive_and_seasonal() -> ModelRegistry:
    """Create a registry with naive and seasonal forecasters."""
    from forecasting.models.naive_forecaster import NaiveForecaster
    from forecasting.models.seasonal_forecaster import SeasonalForecaster

    registry = ModelRegistry()
    registry.register("naive_forecaster", lambda **kw: NaiveForecaster(**kw))
    registry.register("seasonal_forecaster", lambda **kw: SeasonalForecaster(**kw))
    return registry


def _make_destination_df(
    destination_id: str, n_rows: int = 30, start_date: date | None = None
) -> pl.DataFrame:
    """Create a simple demand DataFrame for a single destination."""
    if start_date is None:
        start_date = date(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_rows)]
    demands = [float(10 + i % 7) for i in range(n_rows)]
    return pl.DataFrame(
        {
            "date": dates,
            "destination_id": [destination_id] * n_rows,
            "demand": demands,
        }
    ).cast({"date": pl.Date, "demand": pl.Float64})


def _make_multi_destination_df(
    dest_ids: list[str], n_rows_per_dest: int = 30
) -> pl.DataFrame:
    """Create a DataFrame with multiple destinations."""
    frames = [_make_destination_df(dest_id, n_rows_per_dest) for dest_id in dest_ids]
    return pl.concat(frames)


# --- DestinationOutcome and AggregatedForecastingResult tests ---


class TestDestinationOutcome:
    def test_successful_outcome(self):
        outcome = DestinationOutcome(
            destination_id="dest_1",
            success=True,
            results=[],
            selected=None,
        )
        assert outcome.destination_id == "dest_1"
        assert outcome.success is True
        assert outcome.error is None

    def test_failed_outcome(self):
        outcome = DestinationOutcome(
            destination_id="dest_2",
            success=False,
            error="Something went wrong",
        )
        assert outcome.success is False
        assert outcome.error == "Something went wrong"
        assert outcome.results is None

    def test_frozen(self):
        outcome = DestinationOutcome(destination_id="x", success=True)
        with pytest.raises(Exception):
            outcome.success = False  # type: ignore


class TestAggregatedForecastingResult:
    def test_successful_and_failed_properties(self):
        outcomes = [
            DestinationOutcome(destination_id="a", success=True),
            DestinationOutcome(destination_id="b", success=False, error="err"),
            DestinationOutcome(destination_id="c", success=True),
        ]
        result = AggregatedForecastingResult(outcomes=outcomes)
        assert len(result.successful) == 2
        assert len(result.failed) == 1
        assert result.failed[0].destination_id == "b"


# --- Pipeline __init__ validation tests ---


class TestPipelineInit:
    def test_valid_init(self):
        registry = _make_registry_with_naive_and_seasonal()
        pipeline = PerDestinationForecastingPipeline(
            registry=registry,
            model_names=["naive_forecaster"],
            train_ratio=0.8,
            max_workers=1,
        )
        assert pipeline.train_ratio == 0.8
        assert pipeline.max_workers == 1

    def test_max_workers_too_low(self):
        registry = _make_registry_with_naive_and_seasonal()
        with pytest.raises(ValueError, match="max_workers must be 1-128"):
            PerDestinationForecastingPipeline(
                registry=registry,
                model_names=["naive_forecaster"],
                max_workers=0,
            )

    def test_max_workers_too_high(self):
        registry = _make_registry_with_naive_and_seasonal()
        with pytest.raises(ValueError, match="max_workers must be 1-128"):
            PerDestinationForecastingPipeline(
                registry=registry,
                model_names=["naive_forecaster"],
                max_workers=129,
            )

    def test_train_ratio_zero(self):
        registry = _make_registry_with_naive_and_seasonal()
        with pytest.raises(ValueError, match="train_ratio must be in"):
            PerDestinationForecastingPipeline(
                registry=registry,
                model_names=["naive_forecaster"],
                train_ratio=0.0,
            )

    def test_train_ratio_one(self):
        registry = _make_registry_with_naive_and_seasonal()
        with pytest.raises(ValueError, match="train_ratio must be in"):
            PerDestinationForecastingPipeline(
                registry=registry,
                model_names=["naive_forecaster"],
                train_ratio=1.0,
            )

    def test_train_ratio_negative(self):
        registry = _make_registry_with_naive_and_seasonal()
        with pytest.raises(ValueError, match="train_ratio must be in"):
            PerDestinationForecastingPipeline(
                registry=registry,
                model_names=["naive_forecaster"],
                train_ratio=-0.1,
            )


# --- _split_train_test tests ---


class TestSplitTrainTest:
    def test_basic_split(self):
        registry = _make_registry_with_naive_and_seasonal()
        pipeline = PerDestinationForecastingPipeline(
            registry=registry,
            model_names=["naive_forecaster"],
            train_ratio=0.8,
        )
        df = _make_destination_df("dest_1", n_rows=10)
        train, test = pipeline._split_train_test(df)
        assert train.height == 8  # floor(10 * 0.8) = 8
        assert test.height == 2

    def test_split_preserves_chronological_order(self):
        registry = _make_registry_with_naive_and_seasonal()
        pipeline = PerDestinationForecastingPipeline(
            registry=registry,
            model_names=["naive_forecaster"],
            train_ratio=0.7,
        )
        df = _make_destination_df("dest_1", n_rows=10)
        train, test = pipeline._split_train_test(df)
        # Train should have earliest dates, test should have latest
        assert train.get_column("date").max() <= test.get_column("date").min()

    def test_split_with_small_df(self):
        registry = _make_registry_with_naive_and_seasonal()
        pipeline = PerDestinationForecastingPipeline(
            registry=registry,
            model_names=["naive_forecaster"],
            train_ratio=0.5,
        )
        df = _make_destination_df("dest_1", n_rows=2)
        train, test = pipeline._split_train_test(df)
        assert train.height == 1  # floor(2 * 0.5) = 1
        assert test.height == 1


# --- Basic 2-destination run ---


class TestPipelineRun:
    def test_basic_two_destination_run(self):
        """Test pipeline processes 2 destinations and produces results for each."""
        registry = _make_registry_with_naive_and_seasonal()
        pipeline = PerDestinationForecastingPipeline(
            registry=registry,
            model_names=["naive_forecaster", "seasonal_forecaster"],
            train_ratio=0.7,
            max_workers=1,
            model_params={"seasonal_forecaster": {"lag_value": 3}},
        )
        df = _make_multi_destination_df(["dest_A", "dest_B"], n_rows_per_dest=30)
        result = pipeline.run(df)

        assert len(result.outcomes) == 2
        assert len(result.successful) == 2
        assert len(result.failed) == 0

        # Each destination should have results for both models
        for outcome in result.outcomes:
            assert outcome.success is True
            assert outcome.results is not None
            assert len(outcome.results) == 2
            assert outcome.selected is not None
            # All results should be for the correct destination
            for fr in outcome.results:
                assert fr.destination_id == outcome.destination_id

    def test_insufficient_history_skipping(self):
        """Destinations with fewer rows than minimum_history_length are skipped."""
        registry = _make_registry_with_naive_and_seasonal()
        pipeline = PerDestinationForecastingPipeline(
            registry=registry,
            model_names=["naive_forecaster"],
            train_ratio=0.8,
            minimum_history_length=10,
            max_workers=1,
        )
        # dest_A has 20 rows (enough), dest_B has 5 rows (not enough)
        df_a = _make_destination_df("dest_A", n_rows=20)
        df_b = _make_destination_df("dest_B", n_rows=5)
        df = pl.concat([df_a, df_b])

        result = pipeline.run(df)

        # Only dest_A should be processed
        assert len(result.outcomes) == 1
        assert result.outcomes[0].destination_id == "dest_A"
        assert result.outcomes[0].success is True

    def test_fault_tolerance_bad_model(self):
        """If a model raises an exception, the pipeline continues with other models."""
        from forecasting.models.base_forecaster import BaseForecaster

        class FailingForecaster(BaseForecaster):
            def __init__(self, **kwargs):
                self.forecast_col = "failing_forecast"

            @property
            def name(self):
                return "failing_model"

            def fit(self, df):
                raise RuntimeError("Intentional failure")

            def predict(self, df):
                raise RuntimeError("Intentional failure")

        registry = ModelRegistry()
        registry.register(
            "naive_forecaster",
            lambda **kw: __import__(
                "forecasting.models.naive_forecaster", fromlist=["NaiveForecaster"]
            ).NaiveForecaster(**kw),
        )
        registry.register("failing_model", lambda **kw: FailingForecaster(**kw))

        pipeline = PerDestinationForecastingPipeline(
            registry=registry,
            model_names=["failing_model", "naive_forecaster"],
            train_ratio=0.8,
            max_workers=1,
        )
        df = _make_destination_df("dest_1", n_rows=20)
        result = pipeline.run(df)

        # Should still succeed because naive_forecaster works
        assert len(result.successful) == 1
        outcome = result.outcomes[0]
        assert outcome.success is True
        assert outcome.results is not None
        # Only naive_forecaster should have produced a result
        assert len(outcome.results) == 1
        assert outcome.results[0].model_name == "naive_forecaster"

    def test_fault_tolerance_all_models_fail(self):
        """If all models fail for a destination, it's recorded as failed."""
        from forecasting.models.base_forecaster import BaseForecaster

        class FailingForecaster(BaseForecaster):
            def __init__(self, **kwargs):
                self.forecast_col = "failing_forecast"

            @property
            def name(self):
                return "failing_model"

            def fit(self, df):
                raise RuntimeError("Intentional failure")

            def predict(self, df):
                raise RuntimeError("Intentional failure")

        registry = ModelRegistry()
        registry.register("failing_model", lambda **kw: FailingForecaster(**kw))

        pipeline = PerDestinationForecastingPipeline(
            registry=registry,
            model_names=["failing_model"],
            train_ratio=0.8,
            max_workers=1,
        )
        df = _make_destination_df("dest_1", n_rows=20)
        result = pipeline.run(df)

        assert len(result.failed) == 1
        assert result.outcomes[0].success is False
        assert "All models failed" in result.outcomes[0].error

    def test_parallel_vs_sequential_equivalence(self):
        """Parallel execution (max_workers=2) produces same results as sequential."""
        registry = _make_registry_with_naive_and_seasonal()
        df = _make_multi_destination_df(
            ["dest_A", "dest_B", "dest_C"], n_rows_per_dest=30
        )

        common_kwargs = dict(
            registry=registry,
            model_names=["naive_forecaster"],
            train_ratio=0.8,
            random_seed=42,
        )

        pipeline_seq = PerDestinationForecastingPipeline(max_workers=1, **common_kwargs)
        pipeline_par = PerDestinationForecastingPipeline(max_workers=2, **common_kwargs)

        result_seq = pipeline_seq.run(df)
        result_par = pipeline_par.run(df)

        # Same number of outcomes
        assert len(result_seq.outcomes) == len(result_par.outcomes)

        # Compare outcomes by destination_id
        seq_by_dest = {o.destination_id: o for o in result_seq.outcomes}
        par_by_dest = {o.destination_id: o for o in result_par.outcomes}

        for dest_id in seq_by_dest:
            seq_outcome = seq_by_dest[dest_id]
            par_outcome = par_by_dest[dest_id]
            assert seq_outcome.success == par_outcome.success
            assert seq_outcome.selected == par_outcome.selected

            # Compare forecast values
            if seq_outcome.results and par_outcome.results:
                for sr, pr in zip(seq_outcome.results, par_outcome.results):
                    assert sr.model_name == pr.model_name
                    assert sr.metrics == pr.metrics
                    assert sr.forecast_values.equals(pr.forecast_values)

    def test_row_order_independence(self):
        """Pipeline produces same results regardless of input row order."""
        registry = _make_registry_with_naive_and_seasonal()
        df = _make_multi_destination_df(["dest_A", "dest_B"], n_rows_per_dest=30)

        pipeline = PerDestinationForecastingPipeline(
            registry=registry,
            model_names=["naive_forecaster"],
            train_ratio=0.8,
            max_workers=1,
            random_seed=42,
        )

        # Run on original order
        result_original = pipeline.run(df)

        # Shuffle the DataFrame
        df_shuffled = df.sample(fraction=1.0, seed=123)
        result_shuffled = pipeline.run(df_shuffled)

        # Results should be identical
        orig_by_dest = {o.destination_id: o for o in result_original.outcomes}
        shuf_by_dest = {o.destination_id: o for o in result_shuffled.outcomes}

        for dest_id in orig_by_dest:
            orig = orig_by_dest[dest_id]
            shuf = shuf_by_dest[dest_id]
            assert orig.success == shuf.success
            assert orig.selected == shuf.selected
            if orig.results and shuf.results:
                for or_r, sh_r in zip(orig.results, shuf.results):
                    assert or_r.metrics == sh_r.metrics
                    assert or_r.forecast_values.equals(sh_r.forecast_values)

    def test_model_params_passed_through(self):
        """Model parameters from model_params dict are passed to model creation."""
        registry = _make_registry_with_naive_and_seasonal()
        pipeline = PerDestinationForecastingPipeline(
            registry=registry,
            model_names=["seasonal_forecaster"],
            train_ratio=0.8,
            max_workers=1,
            model_params={"seasonal_forecaster": {"lag_value": 3}},
        )
        df = _make_destination_df("dest_1", n_rows=20)
        result = pipeline.run(df)

        assert len(result.successful) == 1
        outcome = result.outcomes[0]
        assert outcome.results is not None
        # The seasonal forecaster with lag_value=3 should have a name reflecting that
        assert "3" in outcome.results[0].model_name

    def test_forecast_result_fields(self):
        """ForecastResult objects have correct field values."""
        registry = _make_registry_with_naive_and_seasonal()
        pipeline = PerDestinationForecastingPipeline(
            registry=registry,
            model_names=["naive_forecaster"],
            train_ratio=0.8,
            max_workers=1,
        )
        df = _make_destination_df("dest_1", n_rows=20)
        result = pipeline.run(df)

        outcome = result.outcomes[0]
        assert outcome.results is not None
        fr = outcome.results[0]

        assert fr.destination_id == "dest_1"
        assert fr.model_name == "naive_forecaster"
        assert fr.execution_time_seconds >= 0
        assert fr.train_period.start_date < fr.train_period.end_date
        assert fr.validation_period.start_date < fr.validation_period.end_date
        assert set(fr.forecast_values.columns) == {"date", "forecast"}
        assert fr.forecast_values.schema["date"] == pl.Date
        assert fr.forecast_values.schema["forecast"] == pl.Float64
        assert "wape" in fr.metrics

    def test_parallel_stochastic_models_equivalence(self):
        """Parallel execution with ETS and SARIMAX produces same results as sequential.

        Closes the gap identified in the v1.0 review: the existing
        test_parallel_vs_sequential_equivalence only used NaiveForecaster.
        This test runs both statsmodels-backed models (ETSForecaster,
        SARIMAXForecaster) under max_workers=2 and verifies selected model
        identity and metric values match the sequential run.
        """
        from forecasting.models.ets_forecaster import ETSForecaster
        from forecasting.models.sarimax_forecaster import SARIMAXForecaster

        registry = ModelRegistry()
        registry.register(
            "naive_forecaster",
            lambda **kw: __import__(
                "forecasting.models.naive_forecaster", fromlist=["NaiveForecaster"]
            ).NaiveForecaster(**kw),
        )
        registry.register("ets_forecaster", lambda **kw: ETSForecaster(**kw))
        registry.register("sarimax_forecaster", lambda **kw: SARIMAXForecaster(**kw))

        # Use 40 rows so train=32, test=8 — enough for ETS/SARIMAX to fit
        df = _make_multi_destination_df(
            ["dest_A", "dest_B", "dest_C"], n_rows_per_dest=40
        )

        common_kwargs = dict(
            registry=registry,
            model_names=["naive_forecaster", "ets_forecaster", "sarimax_forecaster"],
            train_ratio=0.8,
            random_seed=42,
        )

        pipeline_seq = PerDestinationForecastingPipeline(max_workers=1, **common_kwargs)
        pipeline_par = PerDestinationForecastingPipeline(max_workers=2, **common_kwargs)

        result_seq = pipeline_seq.run(df)
        result_par = pipeline_par.run(df)

        assert len(result_seq.outcomes) == len(result_par.outcomes)

        seq_by_dest = {o.destination_id: o for o in result_seq.outcomes}
        par_by_dest = {o.destination_id: o for o in result_par.outcomes}

        for dest_id in seq_by_dest:
            seq_outcome = seq_by_dest[dest_id]
            par_outcome = par_by_dest[dest_id]

            assert seq_outcome.success == par_outcome.success, (
                f"Success mismatch for {dest_id}: "
                f"seq={seq_outcome.success}, par={par_outcome.success}"
            )
            assert seq_outcome.selected == par_outcome.selected, (
                f"Selected model mismatch for {dest_id}: "
                f"seq={seq_outcome.selected}, par={par_outcome.selected}"
            )

            if seq_outcome.results and par_outcome.results:
                seq_results_by_name = {r.model_name: r for r in seq_outcome.results}
                par_results_by_name = {r.model_name: r for r in par_outcome.results}
                for model_name in seq_results_by_name:
                    if model_name in par_results_by_name:
                        sr = seq_results_by_name[model_name]
                        pr = par_results_by_name[model_name]
                        assert sr.metrics == pr.metrics, (
                            f"Metric mismatch for {dest_id}/{model_name}: "
                            f"seq={sr.metrics}, par={pr.metrics}"
                        )
                        assert sr.forecast_values.equals(
                            pr.forecast_values
                        ), f"Forecast mismatch for {dest_id}/{model_name}"

    def test_all_nan_metric_destination_recorded_as_failed(self):
        """When every model returns NaN for the selection metric, destination is failed.

        This covers the edge case where all-zero demand causes WAPE=NaN for every
        model, forcing PerDestinationModelSelector.select() to raise ValueError,
        which the pipeline catches and records as a failed DestinationOutcome.
        """
        from forecasting.models.base_forecaster import BaseForecaster

        class NaNMetricForecaster(BaseForecaster):
            """Always returns a constant forecast equal to mean demand, forcing
            WAPE=NaN when all actuals are zero."""

            def __init__(self, **kwargs):
                self.forecast_col = "nan_metric_forecast"

            @property
            def name(self) -> str:
                return "nan_metric_forecaster"

            def fit(self, df: pl.DataFrame) -> None:
                pass

            def predict(self, df: pl.DataFrame) -> pl.DataFrame:
                # Return a non-null forecast so evaluate() proceeds, but against
                # all-zero actuals WAPE = sum|0-c| / sum|0| = inf/0 = NaN
                return df.with_columns(pl.lit(1.0).alias(self.forecast_col))

        registry = ModelRegistry()
        registry.register(
            "nan_metric_forecaster", lambda **kw: NaNMetricForecaster(**kw)
        )

        pipeline = PerDestinationForecastingPipeline(
            registry=registry,
            model_names=["nan_metric_forecaster"],
            train_ratio=0.8,
            selection_metric="wape",
            max_workers=1,
        )

        # Build a destination where all demand values are exactly 0
        # so WAPE denominator = sum(|actual|) = 0 → NaN
        df = pl.DataFrame(
            {
                "date": [date(2023, 1, 1) + timedelta(days=i) for i in range(20)],
                "destination_id": ["zero_demand"] * 20,
                "demand": [0.0] * 20,
            }
        ).cast({"date": pl.Date, "demand": pl.Float64})

        result = pipeline.run(df)

        # The destination should be recorded as failed because all metrics are NaN
        assert len(result.failed) == 1
        assert result.failed[0].destination_id == "zero_demand"
        assert result.failed[0].success is False
