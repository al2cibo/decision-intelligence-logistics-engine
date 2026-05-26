"""Common pipeline interface for forecasting pipelines.

Defines a structural subtyping protocol that all pipeline implementations
must satisfy. Both ``ForecastingPipeline`` and ``PerDestinationPipeline``
conform to this protocol without explicit inheritance.
"""

from typing import Any, Protocol, runtime_checkable

import polars as pl


@runtime_checkable
class PipelineProtocol(Protocol):
    """Protocol that all pipeline implementations must satisfy.

    Any class with a ``run(df: pl.DataFrame, **kwargs: Any) -> Any`` method
    satisfies this protocol via structural subtyping (duck typing). No base
    class inheritance is required.

    Examples
    --------
    >>> class MyPipeline:
    ...     def run(self, df: pl.DataFrame, **kwargs: Any) -> Any:
    ...         return df
    >>> isinstance(MyPipeline(), PipelineProtocol)
    True
    """

    def run(self, df: pl.DataFrame, **kwargs: Any) -> Any:
        """Execute the pipeline on input data.

        Parameters
        ----------
        df : pl.DataFrame
            Input DataFrame containing the data to process.
        **kwargs : Any
            Additional keyword arguments specific to the pipeline
            implementation.

        Returns
        -------
        Any
            Pipeline result. The concrete type depends on the implementation
            (e.g., ``pl.DataFrame`` for ``ForecastingPipeline``,
            ``AggregatedPipelineResult`` for ``PerDestinationPipeline``).
        """
        ...
