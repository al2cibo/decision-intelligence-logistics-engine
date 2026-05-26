"""Per-destination forecasting pipeline.

Orchestrates per-destination model training, evaluation, and selection.
Supports parallel execution via joblib and provides fault tolerance
at the destination level.
"""

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

import polars as pl
from joblib import Parallel, delayed

from forecasting.results.forecast_result import ForecastResult, TimePeriod
from forecasting.registry.model_registry import ModelRegistry
from forecasting.evaluation.per_destination_model_selector import (
    PerDestinationModelSelector,
    SelectionResult,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DestinationOutcome:
    """Outcome for a single destination: either success with results or failure."""

    destination_id: str
    success: bool
    results: list[ForecastResult] | None = None
    selected: SelectionResult | None = None
    error: str | None = None


@dataclass(frozen=True)
class AggregatedPipelineResult:
    """Aggregated results across all destinations."""

    outcomes: list[DestinationOutcome]

    @property
    def successful(self) -> list[DestinationOutcome]:
        """Return outcomes where processing succeeded."""
        return [o for o in self.outcomes if o.success]

    @property
    def failed(self) -> list[DestinationOutcome]:
        """Return outcomes where processing failed."""
        return [o for o in self.outcomes if not o.success]


class PerDestinationPipeline:
    """Orchestrates per-destination model training, evaluation, and selection."""

    def __init__(
        self,
        registry: ModelRegistry,
        model_names: list[str],
        train_ratio: float = 0.8,
        selection_metric: str = "wape",
        max_workers: int = 1,
        minimum_history_length: int = 2,
        random_seed: int = 42,
        model_params: dict[str, dict[str, Any]] | None = None,
    ):
        """Initialize the per-destination pipeline.

        Parameters
        ----------
        registry : ModelRegistry
            Registry for creating model instances.
        model_names : list[str]
            Ordered list of model names to train per destination.
        train_ratio : float
            Fraction of data for training (0 < ratio < 1).
        selection_metric : str
            Metric to minimize for model selection.
        max_workers : int
            Number of parallel workers (1-128).
        minimum_history_length : int
            Minimum rows required per destination.
        random_seed : int
            Base seed for reproducibility.
        model_params : dict | None
            Per-model parameter overrides: {model_name: {param: value}}.

        Raises
        ------
        ValueError
            If max_workers or train_ratio are out of valid range.
        """
        if max_workers < 1 or max_workers > 128:
            raise ValueError(f"max_workers must be 1-128, got {max_workers}")
        if train_ratio <= 0 or train_ratio >= 1:
            raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")

        self.registry = registry
        self.model_names = model_names
        self.train_ratio = train_ratio
        self.selection_metric = selection_metric
        self.max_workers = max_workers
        self.minimum_history_length = minimum_history_length
        self.random_seed = random_seed
        self.model_params = model_params or {}
        self._selector = PerDestinationModelSelector(metric=selection_metric)

    def run(self, df: pl.DataFrame) -> AggregatedPipelineResult:
        """Execute the per-destination pipeline.

        Parameters
        ----------
        df : pl.DataFrame
            Input DataFrame with columns: date, destination_id, demand.

        Returns
        -------
        AggregatedPipelineResult
            Contains outcome for each destination.
        """
        # 1. Sort by destination_id and date for determinism
        df_sorted = df.sort(["destination_id", "date"])

        # 2. Partition by destination_id
        destination_ids = (
            df_sorted.select("destination_id")
            .unique()
            .sort("destination_id")
            .get_column("destination_id")
            .to_list()
        )

        # 3. Filter destinations with insufficient history and prepare work items
        work_items: list[tuple[str, pl.DataFrame]] = []
        for dest_id in destination_ids:
            dest_df = df_sorted.filter(pl.col("destination_id") == dest_id)
            if dest_df.height < self.minimum_history_length:
                logger.warning(
                    f"Destination '{dest_id}' has insufficient history "
                    f"({dest_df.height} rows, minimum {self.minimum_history_length}). "
                    f"Skipping."
                )
                continue
            work_items.append((dest_id, dest_df))

        # 4. Process destinations (parallel or sequential)
        if self.max_workers == 1:
            outcomes = [
                self._process_destination(dest_id, dest_df)
                for dest_id, dest_df in work_items
            ]
        else:
            outcomes = Parallel(n_jobs=self.max_workers)(
                delayed(self._process_destination)(dest_id, dest_df)
                for dest_id, dest_df in work_items
            )

        # 5. Aggregate results
        return AggregatedPipelineResult(outcomes=outcomes)

    def _process_destination(
        self, destination_id: str, dest_df: pl.DataFrame
    ) -> DestinationOutcome:
        """Process a single destination: fit all models, evaluate, select best.

        This method is the unit of parallelism. It creates fresh model instances
        from the registry (no shared state) and processes them sequentially.
        """
        try:
            # Set deterministic random seed per destination
            dest_seed = abs(hash(destination_id)) ^ self.random_seed
            import numpy as np

            np.random.seed(dest_seed % (2**32))

            # Sort by date and split
            dest_df_sorted = dest_df.sort("date")
            train_df, test_df = self._split_train_test(dest_df_sorted)

            # Compute time periods
            train_dates = train_df.get_column("date")
            test_dates = test_df.get_column("date")
            train_period = TimePeriod(
                start_date=train_dates.min(), end_date=train_dates.max()
            )
            validation_period = TimePeriod(
                start_date=test_dates.min(), end_date=test_dates.max()
            )

            # Process each model
            results: list[ForecastResult] = []
            model_metrics: list[tuple[str, dict[str, float]]] = []

            for model_name in self.model_names:
                try:
                    # Create fresh model instance from registry
                    kwargs = self.model_params.get(model_name, {})
                    model = self.registry.create(model_name, **kwargs)

                    # Fit on train, predict on test, evaluate
                    start_time = time.time()
                    model.fit(train_df)
                    predicted_df = model.predict(test_df)
                    execution_time = time.time() - start_time

                    # Extract forecast column and evaluate
                    forecast_col = model.forecast_col
                    metrics = model.evaluate(
                        predicted_df, target_col="demand", forecast_col=forecast_col
                    )

                    # Build forecast_values DataFrame (date + forecast columns only)
                    forecast_values = predicted_df.select([
                        pl.col("date"),
                        pl.col(forecast_col).alias("forecast"),
                    ]).cast({"forecast": pl.Float64})

                    # Build model parameters dict
                    model_parameters = dict(kwargs)

                    result = ForecastResult(
                        destination_id=destination_id,
                        model_name=model.name,
                        train_period=train_period,
                        validation_period=validation_period,
                        forecast_values=forecast_values,
                        metrics=metrics,
                        execution_time_seconds=execution_time,
                        model_parameters=model_parameters,
                    )
                    results.append(result)
                    model_metrics.append((model.name, metrics))

                except Exception as e:
                    logger.error(
                        f"Model '{model_name}' failed for destination "
                        f"'{destination_id}': {e}"
                    )
                    continue

            # If no models succeeded, record as failed
            if not results:
                return DestinationOutcome(
                    destination_id=destination_id,
                    success=False,
                    error=f"All models failed for destination '{destination_id}'",
                )

            # Select best model
            selection = self._selector.select(destination_id, model_metrics)

            return DestinationOutcome(
                destination_id=destination_id,
                success=True,
                results=results,
                selected=selection,
            )

        except Exception as e:
            logger.error(
                f"Failed to process destination '{destination_id}': {e}"
            )
            return DestinationOutcome(
                destination_id=destination_id,
                success=False,
                error=str(e),
            )

    def _split_train_test(
        self, df: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Split a sorted destination DataFrame into train/test by train_ratio.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame sorted by date ascending.

        Returns
        -------
        tuple[pl.DataFrame, pl.DataFrame]
            (train_df, test_df) where train has floor(n * train_ratio) rows.
        """
        split_idx = int(math.floor(len(df) * self.train_ratio))
        return df[:split_idx], df[split_idx:]
