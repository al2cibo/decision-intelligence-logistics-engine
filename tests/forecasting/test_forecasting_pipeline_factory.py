"""Tests for the forecasting pipeline factory function."""

import pytest
import polars as pl
from datetime import date, timedelta

from forecasting.pipeline.forecasting_pipeline_factory import create_forecasting_pipeline
from forecasting.pipeline.per_destination_forecasting_pipeline import PerDestinationForecastingPipeline
from utils.config import PerDestinationConfig


class TestCreateForecastingPipeline:
    """Tests for create_forecasting_pipeline."""

    def test_creates_pipeline_with_valid_config(self):
        """Factory returns a PerDestinationForecastingPipeline with valid config."""
        config = PerDestinationConfig(
            model_names=["naive_forecaster", "seasonal_forecaster"],
            train_ratio=0.8,
            selection_metric="wape",
            max_workers=1,
            minimum_history_length=2,
            random_seed=42,
            model_params={"seasonal_forecaster": {"lag_value": 7}},
        )

        pipeline = create_forecasting_pipeline(config)

        assert isinstance(pipeline, PerDestinationForecastingPipeline)
        assert pipeline.model_names == ["naive_forecaster", "seasonal_forecaster"]
        assert pipeline.train_ratio == 0.8
        assert pipeline.selection_metric == "wape"
        assert pipeline.max_workers == 1
        assert pipeline.minimum_history_length == 2
        assert pipeline.random_seed == 42
        assert pipeline.model_params == {"seasonal_forecaster": {"lag_value": 7}}

    def test_validates_model_names_against_registry(self):
        """Factory raises ValueError if a model name is not in the registry."""
        config = PerDestinationConfig(
            model_names=["naive_forecaster", "nonexistent_model"],
            train_ratio=0.8,
            selection_metric="wape",
        )

        with pytest.raises(ValueError, match="not registered"):
            create_forecasting_pipeline(config)

    def test_all_default_models_accepted(self):
        """Factory accepts all 5 default registered models."""
        config = PerDestinationConfig(
            model_names=[
                "naive_forecaster",
                "seasonal_forecaster",
                "rolling_window_forecaster",
                "ets_forecaster",
                "sarimax_forecaster",
            ],
            train_ratio=0.7,
            selection_metric="mae",
            max_workers=2,
        )

        pipeline = create_forecasting_pipeline(config)

        assert isinstance(pipeline, PerDestinationForecastingPipeline)
        assert len(pipeline.model_names) == 5

    def test_model_params_passed_through(self):
        """Factory passes model_params from config to the pipeline."""
        config = PerDestinationConfig(
            model_names=["rolling_window_forecaster", "seasonal_forecaster"],
            train_ratio=0.8,
            selection_metric="wape",
            model_params={
                "rolling_window_forecaster": {"rolling_window": 14},
                "seasonal_forecaster": {"lag_value": 3},
            },
        )

        pipeline = create_forecasting_pipeline(config)

        assert pipeline.model_params == {
            "rolling_window_forecaster": {"rolling_window": 14},
            "seasonal_forecaster": {"lag_value": 3},
        }

    def test_pipeline_runs_end_to_end_with_factory(self):
        """Pipeline created by factory can run on real data."""
        config = PerDestinationConfig(
            model_names=["naive_forecaster", "seasonal_forecaster"],
            train_ratio=0.8,
            selection_metric="wape",
            max_workers=1,
            minimum_history_length=5,
            random_seed=42,
            model_params={"seasonal_forecaster": {"lag_value": 7}},
        )

        pipeline = create_forecasting_pipeline(config)

        # Create simple test data with 2 destinations
        dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(30)]
        rows = []
        for d in dates:
            rows.append({"date": d, "destination_id": "A", "demand": 100.0})
            rows.append({"date": d, "destination_id": "B", "demand": 50.0})

        df = pl.DataFrame(rows).cast({"date": pl.Date, "demand": pl.Float64})

        result = pipeline.run(df)

        assert len(result.successful) == 2
        assert len(result.failed) == 0

        dest_ids = {o.destination_id for o in result.successful}
        assert dest_ids == {"A", "B"}

        # Each destination should have a selected model
        for outcome in result.successful:
            assert outcome.selected is not None
            assert outcome.selected.model_name in [
                "naive_forecaster",
                "seasonal_forecaster",
            ]
