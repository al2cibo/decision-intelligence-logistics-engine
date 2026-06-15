"""
End-to-end pipeline: per-destination forecasting → multi-period optimization.

Workflow:
  1. Ingest and process raw data
  2. Run PerDestinationPipeline — independently train, evaluate, and select
     the best forecasting model per destination
  3. Extract the selected model's test-period forecasts as a demand time series
  4. Feed that demand time series into the MultiPeriodOptimizer

Usage:
    python example_end_to_end_pipeline.py --config configs/test_config.yaml
"""

import argparse
import logging

import polars as pl

from data.ingestion import Reader
from data.processing.data_processor import DataProcessor

from forecasting.pipeline.pipeline_factory import create_per_destination_pipeline_from_config
from forecasting.pipeline.per_destination_pipeline import AggregatedPipelineResult

from optimization import MultiPeriodOptimizer, MultiPeriodResult

from utils.config import load_config, PerDestinationConfig
from utils.system_paths import get_project_root

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run end-to-end pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/test_config.yaml",
        help="Path to config file",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------
# Per-destination forecasting
# ---------------------------------------------------------------------
def run_per_destination_forecasting(
    demand_df: pl.DataFrame,
    config: PerDestinationConfig,
) -> AggregatedPipelineResult:
    """Train, evaluate, and select the best model independently per destination."""
    pipeline = create_per_destination_pipeline_from_config(config)

    logger.info("Running per-destination pipeline on %d rows...", demand_df.height)
    result = pipeline.run(demand_df)

    logger.info(
        "Per-destination forecasting complete: %d successful, %d failed",
        len(result.successful),
        len(result.failed),
    )
    for outcome in result.successful:
        selected = outcome.selected
        logger.info(
            "  Destination %-6s -> Best model: %-30s (WAPE: %.4f, MAE: %.4f)",
            outcome.destination_id,
            selected.model_name,
            selected.metrics.get("wape", float("nan")),
            selected.metrics.get("mae", float("nan")),
        )
        for fr in outcome.results:
            marker = " <-- SELECTED" if fr.model_name == selected.model_name else ""
            logger.info(
                "    %-30s WAPE=%.4f  MAE=%.4f  RMSE=%.4f%s",
                fr.model_name,
                fr.metrics.get("wape", float("nan")),
                fr.metrics.get("mae", float("nan")),
                fr.metrics.get("rmse", float("nan")),
                marker,
            )
    for outcome in result.failed:
        logger.warning(
            "  Destination %-6s -> FAILED: %s",
            outcome.destination_id,
            outcome.error,
        )

    return result


# ---------------------------------------------------------------------
# Demand extraction
# ---------------------------------------------------------------------
def extract_demand_time_series(result: AggregatedPipelineResult) -> pl.DataFrame:
    """Build a [destination_id, date, demand] DataFrame from each destination's
    selected model forecast values (covering the test period).

    Destinations that failed or whose selected ForecastResult is missing
    are excluded with a warning.
    """
    frames = []

    for outcome in result.successful:
        selected_name = outcome.selected.model_name
        selected_fr = next(
            (fr for fr in outcome.results if fr.model_name == selected_name), None
        )
        if selected_fr is None:
            logger.warning(
                "No ForecastResult found for selected model '%s' at destination '%s'. Skipping.",
                selected_name,
                outcome.destination_id,
            )
            continue

        # forecast_values has schema [date: Date, forecast: Float64]
        dest_df = (
            selected_fr.forecast_values
            .with_columns(pl.lit(outcome.destination_id).alias("destination_id"))
            .rename({"forecast": "demand"})
            .select(["destination_id", "date", "demand"])
        )
        frames.append(dest_df)

    if not frames:
        raise ValueError(
            "No forecast data available — all destinations failed or had no results."
        )

    demand_ts = pl.concat(frames).sort(["destination_id", "date"])
    logger.info(
        "Extracted demand time series: %d rows across %d destinations",
        demand_ts.height,
        demand_ts["destination_id"].n_unique(),
    )
    return demand_ts


# ---------------------------------------------------------------------
# Multi-period optimization
# ---------------------------------------------------------------------
def run_multi_period_optimization(
    demand_ts: pl.DataFrame,
    origins_df: pl.DataFrame,
    lanes_df: pl.DataFrame,
    destinations_df: pl.DataFrame,
    solver_name: str = "GLOP",
) -> MultiPeriodResult:
    """Solve the multi-period transportation LP.

    The planning horizon is derived from the unique dates present in
    demand_ts (the test-period forecasts).

    Note: destinations_df does not need a holding_cost column — if absent,
    the optimizer minimises shipping costs only.
    """
    planning_horizon = sorted(demand_ts["date"].unique().to_list())
    logger.info(
        "Planning horizon: %d periods (%s → %s)",
        len(planning_horizon),
        planning_horizon[0],
        planning_horizon[-1],
    )

    optimizer = MultiPeriodOptimizer(solver_name=solver_name)
    result = optimizer.solve(
        demand_ts=demand_ts,
        origins_df=origins_df,
        lanes_df=lanes_df,
        destinations_df=destinations_df,
        planning_horizon=planning_horizon,
    )

    logger.info("Multi-period optimization complete. Total cost: %.4f", result.total_cost)
    logger.info("Flows (non-zero):\n%s", result.flows)
    logger.info("Inventory levels:\n%s", result.inventory)

    return result


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    project_root = get_project_root()
    config = load_config(project_root, project_root / args.config)

    # --- Data ingestion & processing ---
    reader = Reader(config.data.input_path)
    raw_data = reader.read()
    clean_data = DataProcessor.process(raw_data)

    if clean_data.demand_history.is_empty():
        raise ValueError("Empty demand history after processing")

    # --- Per-destination forecasting config ---
    # ETS and SARIMAX are available but slower; uncomment to include them.
    per_dest_config = PerDestinationConfig(
        model_names=[
            "naive_forecaster",
            "seasonal_forecaster",
            "rolling_window_forecaster",
            # "ets_forecaster",
            # "sarimax_forecaster",
        ],
        train_ratio=config.forecasting.train_ratio if config.forecasting else 0.8,
        selection_metric=config.forecasting.metric if config.forecasting else "wape",
        max_workers=1,
        minimum_history_length=10,
        random_seed=42,
        model_params={
            "seasonal_forecaster": {"lag_value": 7},
            "rolling_window_forecaster": {"rolling_window": 7},
        },
    )

    # --- Per-destination forecasting ---
    forecast_result = run_per_destination_forecasting(
        clean_data.demand_history, per_dest_config
    )

    # --- Extract test-period forecasts as demand time series ---
    demand_ts = extract_demand_time_series(forecast_result)

    # --- Multi-period optimization ---
    opt_result = run_multi_period_optimization(
        demand_ts=demand_ts,
        origins_df=clean_data.origins,
        lanes_df=clean_data.lanes,
        destinations_df=clean_data.destinations,
    )

    return forecast_result, opt_result


if __name__ == "__main__":
    main()
