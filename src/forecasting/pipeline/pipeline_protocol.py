"""Structural protocol for forecasting pipelines."""

from typing import Any, Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class PipelineProtocol(Protocol):
    """Protocol satisfied by any object with a ``run(df, **kwargs)`` method.

    Conformance is checked via structural subtyping (duck typing): no base
    class inheritance is required. :class:`PerDestinationPipeline` conforms
    to this protocol.

    Examples
    --------
    >>> class MyPipeline:
    ...     def run(self, df: pl.DataFrame, **kwargs: Any) -> Any:
    ...         return df
    >>> isinstance(MyPipeline(), PipelineProtocol)
    True
    """

    def run(self, df: pl.DataFrame, **kwargs: Any) -> Any:
        """Execute the pipeline and return results."""
        ...
