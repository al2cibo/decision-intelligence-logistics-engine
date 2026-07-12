"""Per-destination forecasting pipeline.

Orchestrates independent model training, evaluation, and selection for each
destination. Supports parallel execution via joblib and tolerates per-model
and per-destination failures without aborting the full run.
"""

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from joblib import Parallel, delayed

from forecasting.evaluation.per_destination_model_selector import (
    PerDestinationModelSelector,
    SelectionResult,
)
from forecasting.registry.model_registry import ModelRegistry
from forecasting.results.forecast_result import ForecastResult, TimePeriod

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DestinationOutcome:
    """Result for a single destination: either success with results, or failure."""

    destination_id: str
    success: bool
    results: list[ForecastResult] | None = None
    selected: SelectionResult | None = None
    error: str | None = None


@dataclass(frozen=True)
class AggregatedForecastingResult:
    """Collection of per-destination outcomes from a full pipeline run."""

    outcomes: list[DestinationOutcome]

    @property
    def successful(self) -> list[DestinationOutcome]:
        """Outcomes where all models ran and a winner was selected."""
        return [o for o in self.outcomes if o.success]

    @property
    def failed(self) -> list[DestinationOutcome]:
        """Outcomes where processing failed (all models errored or insufficient data)."""
        return [o for o in self.outcomes if not o.success]

    def export_forecasts(self) -> pl.DataFrame:
        """Return selected forecast values as a ``[destination_id, date, demand]`` DataFrame.

        Covers the validation window of the winning model for each successful destination.
        Failed destinations are silently excluded. Raises ``ValueError`` if no successful
        outcomes exist.
        """
        frames = []
        for outcome in self.successful:
            selected_fr = next(
                (
                    fr
                    for fr in outcome.results
                    if fr.model_name == outcome.selected.model_name
                ),
                None,
            )
            if selected_fr is None:
                logger.warning(
                    "No ForecastResult found for selected model '%s' at destination '%s'. Skipping.",
                    outcome.selected.model_name,
                    outcome.destination_id,
                )
                continue
            frames.append(
                selected_fr.forecast_values.with_columns(
                    pl.lit(outcome.destination_id).alias("destination_id")
                )
                .rename({"forecast": "demand"})
                .select(["destination_id", "date", "demand"])
            )
        if not frames:
            raise ValueError(
                "No forecast data available — all destinations failed or had no results."
            )
        return pl.concat(frames).sort(["destination_id", "date"])


class PerDestinationForecastingPipeline:
    """Trains, evaluates, and selects a forecasting model independently per destination.

    Parameters
    ----------
    registry : ModelRegistry
        Registry mapping model names to factory callables.
    model_names : list[str]
        Ordered list of model names to evaluate for each destination.
    train_ratio : float
        Fraction of rows used for training (0 < ratio < 1). Defaults to ``0.8``.
    selection_metric : str
        Metric to minimise for model selection. Defaults to ``"wape"``.
    max_workers : int
        Number of parallel workers (1–128). Use 1 for sequential execution.
        Defaults to ``1``.
    minimum_history_length : int
        Destinations with fewer rows than this are skipped with a warning.
        Defaults to ``2``.
    random_seed : int
        Some of the predictive models may contain stochastic parameters
        (e.g. ETS/SARIMAX), hence we fix this seed to maintain results reproducibility.
        Each destination gets a derived seed
        ``abs(hash(destination_id)) ^ random_seed`` to ensure determinism
        independent of execution order. Defaults to ``42``.
    model_params : dict | None
        Per-model parameter overrides: ``{model_name: {param: value}}``.
        Forwarded to the registry factory at creation time.
    """

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
        validation_periods: int | None = None,
        test_periods: int | None = None,
    ) -> None:
        if not (1 <= max_workers <= 128):
            raise ValueError(f"max_workers must be 1-128, got {max_workers}")
        if validation_periods is None and test_periods is None:
            if not (0 < train_ratio < 1):
                raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
        if (validation_periods is None) != (test_periods is None):
            raise ValueError(
                "validation_periods and test_periods must both be set or both be None."
            )

        self.registry = registry
        self.model_names = model_names
        self.train_ratio = train_ratio
        self.selection_metric = selection_metric
        self.max_workers = max_workers
        self.minimum_history_length = minimum_history_length
        self.random_seed = random_seed
        self.model_params = model_params or {}
        self.validation_periods = validation_periods
        self.test_periods = test_periods
        self._selector = PerDestinationModelSelector(metric=selection_metric)

    def run(self, df: pl.DataFrame, **kwargs: Any) -> AggregatedForecastingResult:
        """Execute the pipeline across all destinations in the input DataFrame.

        Parameters
        ----------
        df : pl.DataFrame
            Demand history with columns: ``date``, ``destination_id``, ``demand``.

        Returns
        -------
        AggregatedForecastingResult
            One :class:`DestinationOutcome` per destination that had enough history.
        """
        df_sorted = df.sort(["destination_id", "date"])

        destination_ids = (
            df_sorted.select("destination_id")
            .unique()
            .sort("destination_id")
            .get_column("destination_id")
            .to_list()
        )

        work_items: list[tuple[str, pl.DataFrame]] = []
        for dest_id in destination_ids:
            dest_df = df_sorted.filter(pl.col("destination_id") == dest_id)
            if dest_df.height < self.minimum_history_length:
                logger.warning(
                    "Destination '%s' skipped: %d rows < minimum %d.",
                    dest_id,
                    dest_df.height,
                    self.minimum_history_length,
                )
                continue
            work_items.append((dest_id, dest_df))

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

        return AggregatedForecastingResult(outcomes=outcomes)

    def _process_destination(
        self, destination_id: str, dest_df: pl.DataFrame
    ) -> DestinationOutcome:
        """Fit all models, evaluate, select the best, and export holdout forecasts.

        Two-way mode (``test_periods`` not set):
            Fit on train, evaluate on test, export test predictions.

        Three-way mode (``test_periods`` and ``validation_periods`` both set):
            1. Fit all models on train, evaluate on validation → select winner.
            2. Re-fit winner on train+validation, predict holdout.
            3. Export holdout predictions; report holdout WAPE.

        This is the unit of parallelism — each call creates fresh model instances
        with no shared state between destinations.
        """
        try:
            dest_seed = abs(hash(destination_id)) ^ self.random_seed
            np.random.seed(dest_seed % (2**32))

            dest_df_sorted = dest_df.sort("date")

            three_way = self.test_periods is not None
            if three_way:
                train_df, val_df, test_df = self._split_three_way(dest_df_sorted)
                selection_df = val_df
            else:
                train_df, test_df = self._split_two_way(dest_df_sorted)
                val_df = None
                selection_df = test_df

            train_period = TimePeriod(
                start_date=train_df["date"].min(),
                end_date=train_df["date"].max(),
            )
            selection_period = TimePeriod(
                start_date=selection_df["date"].min(),
                end_date=selection_df["date"].max(),
            )

            results: list[ForecastResult] = []
            model_metrics: list[tuple[str, dict[str, float]]] = []
            # Maps model display name → registry key (config key in self.model_names)
            display_to_registry_key: dict[str, str] = {}

            # Phase 1: fit on train, evaluate on selection_df (val or test)
            for registry_key in self.model_names:
                try:
                    model_kwargs = self.model_params.get(registry_key, {})
                    model = self.registry.create(registry_key, **model_kwargs)
                    display_to_registry_key[model.name] = registry_key

                    start_time = time.time()
                    model.fit(train_df)
                    predicted_df = model.predict(selection_df)
                    execution_time = time.time() - start_time

                    metrics = model.evaluate(
                        predicted_df,
                        target_col="demand",
                        forecast_col=model.forecast_col,
                    )
                    forecast_values = predicted_df.select(
                        pl.col("date"),
                        pl.col(model.forecast_col).alias("forecast").cast(pl.Float64),
                    )

                    results.append(
                        ForecastResult(
                            destination_id=destination_id,
                            model_name=model.name,
                            train_period=train_period,
                            validation_period=selection_period,
                            forecast_values=forecast_values,
                            metrics=metrics,
                            execution_time_seconds=execution_time,
                            model_parameters=dict(model_kwargs),
                        )
                    )
                    model_metrics.append((model.name, metrics))
                    logger.info(
                        "  [%s] %s: %s WAPE=%.4f",
                        destination_id,
                        model.name,
                        "val" if three_way else "test",
                        metrics.get("wape", float("nan")),
                    )

                except Exception as exc:
                    logger.error(
                        "Model '%s' failed for destination '%s': %s",
                        registry_key,
                        destination_id,
                        exc,
                    )

            if not results:
                return DestinationOutcome(
                    destination_id=destination_id,
                    success=False,
                    error=f"All models failed for destination '{destination_id}'",
                )

            selection = self._selector.select(destination_id, model_metrics)

            # Phase 2 (three-way only): re-fit winner on train+val, predict holdout
            if three_way:
                try:
                    winner_registry_key = display_to_registry_key.get(
                        selection.model_name, selection.model_name
                    )
                    winner_kwargs = self.model_params.get(winner_registry_key, {})
                    winner = self.registry.create(winner_registry_key, **winner_kwargs)
                    train_val_df = pl.concat([train_df, val_df])
                    winner.fit(train_val_df)
                    predicted_holdout = winner.predict(test_df)
                    holdout_metrics = winner.evaluate(
                        predicted_holdout,
                        target_col="demand",
                        forecast_col=winner.forecast_col,
                    )
                    holdout_forecast_values = predicted_holdout.select(
                        pl.col("date"),
                        pl.col(winner.forecast_col).alias("forecast").cast(pl.Float64),
                    )
                    holdout_train_period = TimePeriod(
                        start_date=train_df["date"].min(),
                        end_date=val_df["date"].max(),
                    )
                    holdout_period = TimePeriod(
                        start_date=test_df["date"].min(),
                        end_date=test_df["date"].max(),
                    )
                    holdout_result = ForecastResult(
                        destination_id=destination_id,
                        model_name=winner.name,
                        train_period=holdout_train_period,
                        validation_period=holdout_period,
                        forecast_values=holdout_forecast_values,
                        metrics=holdout_metrics,
                        execution_time_seconds=0.0,
                        model_parameters=dict(winner_kwargs),
                    )
                    # Replace winner's entry with the holdout-evaluated version
                    results = [
                        holdout_result if r.model_name == selection.model_name else r
                        for r in results
                    ]
                    logger.info(
                        "  [%s] %s: holdout WAPE=%.4f",
                        destination_id,
                        selection.model_name,
                        holdout_metrics.get("wape", float("nan")),
                    )
                except Exception as exc:
                    logger.error(
                        "Holdout re-fit failed for winner '%s' at destination '%s': %s",
                        selection.model_name,
                        destination_id,
                        exc,
                    )

            return DestinationOutcome(
                destination_id=destination_id,
                success=True,
                results=results,
                selected=selection,
            )

        except Exception as exc:
            logger.error("Failed to process destination '%s': %s", destination_id, exc)
            return DestinationOutcome(
                destination_id=destination_id,
                success=False,
                error=str(exc),
            )

    def _split_two_way(self, df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Split by train_ratio: first floor(n * ratio) rows train, rest test."""
        split_idx = int(math.floor(len(df) * self.train_ratio))
        return df[:split_idx], df[split_idx:]

    def _split_train_test(self, df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Alias for _split_two_way kept for backward compatibility."""
        return self._split_two_way(df)

    def _split_three_way(
        self, df: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Split into train / validation / holdout using period counts from the end.

        Layout::

            [  train  ][  validation  ][  holdout  ]
            ← n-hp-vp →     ← vp →       ← hp →
        """
        n = len(df)
        hp = self.test_periods
        vp = self.validation_periods
        if n < hp + vp + 1:
            raise ValueError(
                f"Destination has {n} rows but three-way split requires at least "
                f"{hp + vp + 1} (holdout={hp} + validation={vp} + 1 train row)."
            )
        test_df = df[-hp:]
        val_df = df[-(hp + vp) : -hp]
        train_df = df[: -(hp + vp)]
        return train_df, val_df, test_df
