"""Tests verifying forecasting pipeline protocol conformance.

Confirms that PerDestinationForecastingPipeline satisfies ForecastingPipelineProtocol
structurally via isinstance checks at runtime.
"""

from typing import Any

import polars as pl

from forecasting.pipeline.forecasting_pipeline_protocol import (
    ForecastingPipelineProtocol,
)
from forecasting.pipeline.per_destination_forecasting_pipeline import (
    PerDestinationForecastingPipeline,
)
from forecasting.registry.model_registry import ModelRegistry


class TestForecastingPipelineProtocolConformance:
    """Verify PerDestinationForecastingPipeline satisfies ForecastingPipelineProtocol."""

    def test_per_destination_forecasting_pipeline_satisfies_protocol(self) -> None:
        """PerDestinationForecastingPipeline instances pass isinstance check against ForecastingPipelineProtocol."""
        registry = ModelRegistry()
        pipeline = PerDestinationForecastingPipeline(
            registry=registry,
            model_names=[],
        )
        assert isinstance(pipeline, ForecastingPipelineProtocol)

    def test_new_implementation_satisfies_protocol(self) -> None:
        """A new pipeline implementation can satisfy the protocol without modifying existing modules.

        Validates Requirement 10.4: new pipelines can be added by implementing
        the common Pipeline interface without modifying existing pipeline module files.
        """

        class CustomPipeline:
            """A custom pipeline that satisfies ForecastingPipelineProtocol."""

            def run(self, df: pl.DataFrame, **kwargs: Any) -> Any:
                return df

        pipeline = CustomPipeline()
        assert isinstance(pipeline, ForecastingPipelineProtocol)

    def test_non_conforming_class_fails_protocol(self) -> None:
        """A class without a run method does not satisfy ForecastingPipelineProtocol."""

        class NotAPipeline:
            pass

        obj = NotAPipeline()
        assert not isinstance(obj, ForecastingPipelineProtocol)

    def test_per_destination_forecasting_pipeline_run_accepts_kwargs(self) -> None:
        """PerDestinationForecastingPipeline.run accepts **kwargs without TypeError.

        Validates Requirement 10.5.
        """
        registry = ModelRegistry()
        pipeline = PerDestinationForecastingPipeline(
            registry=registry,
            model_names=[],
        )
        df = pl.DataFrame(
            {
                "date": [],
                "destination_id": [],
                "demand": [],
            }
        )
        # Should not raise TypeError
        result = pipeline.run(df, some_extra_kwarg="value")
        assert result is not None
