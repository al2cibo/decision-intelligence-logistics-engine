"""
End-to-end pipeline: per-destination forecasting → multi-period optimization.

Workflow:
  1. Ingest and process raw data
  2. Run PerDestinationForecastingPipeline — independently train, evaluate, and select
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

from forecasting import create_forecasting_pipeline, AggregatedForecastingResult
from forecasting.config import PerDestinationConfig

from optimization import MultiPeriodOptimizer, MultiPeriodResult

from utils.config import load_config
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
) -> AggregatedForecastingResult:
    """Train, evaluate, and select the best model independently per destination."""
    pipeline = create_forecasting_pipeline(config)

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

    logger.info(
        "Multi-period optimization complete. Total cost: %.4f", result.total_cost
    )
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

    # --- Per-destination forecasting config reading ---
    if config.per_destination_forecasting is None:
        raise ValueError(
            "per_destination_forecasting section is missing from the config file."
        )
    per_dest_config = config.per_destination_forecasting

    # --- Per-destination forecasting ---
    forecast_result = run_per_destination_forecasting(
        clean_data.demand_history, per_dest_config
    )

    # --- Extract test-period forecasts as demand time series ---
    demand_ts = forecast_result.export_forecasts()
    logger.info(
        "Extracted demand time series: %d rows across %d destinations",
        demand_ts.height,
        demand_ts["destination_id"].n_unique(),
    )

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
