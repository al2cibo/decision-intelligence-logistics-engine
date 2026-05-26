"""Tests verifying pipeline protocol conformance.

Confirms that ForecastingPipeline and PerDestinationPipeline satisfy
PipelineProtocol structurally via isinstance checks at runtime.
"""

from typing import Any

import polars as pl

from forecasting.pipeline.pipeline_protocol import PipelineProtocol
from forecasting.pipeline.pipeline import ForecastingPipeline
from forecasting.pipeline.per_destination_pipeline import PerDestinationPipeline
from forecasting.registry.model_registry import ModelRegistry


class TestPipelineProtocolConformance:
    """Verify both pipeline implementations satisfy PipelineProtocol."""

    def test_forecasting_pipeline_satisfies_protocol(self) -> None:
        """ForecastingPipeline instances pass isinstance check against PipelineProtocol."""
        pipeline = ForecastingPipeline(models=[])
        assert isinstance(pipeline, PipelineProtocol)

    def test_per_destination_pipeline_satisfies_protocol(self) -> None:
        """PerDestinationPipeline instances pass isinstance check against PipelineProtocol."""
        registry = ModelRegistry()
        pipeline = PerDestinationPipeline(
            registry=registry,
            model_names=[],
        )
        assert isinstance(pipeline, PipelineProtocol)

    def test_new_implementation_satisfies_protocol(self) -> None:
        """A new pipeline implementation can satisfy the protocol without modifying existing modules.

        Validates Requirement 10.4: new pipelines can be added by implementing
        the common Pipeline interface without modifying existing pipeline module files.
        """

        class CustomPipeline:
            """A custom pipeline that satisfies PipelineProtocol."""

            def run(self, df: pl.DataFrame, **kwargs: Any) -> Any:
                return df

        pipeline = CustomPipeline()
        assert isinstance(pipeline, PipelineProtocol)

    def test_non_conforming_class_fails_protocol(self) -> None:
        """A class without a run method does not satisfy PipelineProtocol."""

        class NotAPipeline:
            pass

        obj = NotAPipeline()
        assert not isinstance(obj, PipelineProtocol)

    def test_forecasting_pipeline_run_accepts_kwargs(self) -> None:
        """ForecastingPipeline.run accepts **kwargs without TypeError.

        Validates Requirement 10.5: calling run on any conforming Pipeline
        implementation accepts a Polars DataFrame and returns without raising
        TypeError due to signature mismatch.
        """
        pipeline = ForecastingPipeline(models=[])
        df = pl.DataFrame({"date": [], "demand": []})
        # Should not raise TypeError
        result = pipeline.run(df, some_extra_kwarg="value")
        assert isinstance(result, pl.DataFrame)

    def test_per_destination_pipeline_run_accepts_kwargs(self) -> None:
        """PerDestinationPipeline.run accepts **kwargs without TypeError.

        Validates Requirement 10.5.
        """
        registry = ModelRegistry()
        pipeline = PerDestinationPipeline(
            registry=registry,
            model_names=[],
        )
        df = pl.DataFrame({
            "date": [],
            "destination_id": [],
            "demand": [],
        })
        # Should not raise TypeError
        result = pipeline.run(df, some_extra_kwarg="value")
        assert result is not None
