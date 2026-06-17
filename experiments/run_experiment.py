"""Run a single experiment from a YAML config and persist all artifacts.

Usage (from project root):
    PYTHONPATH=src python experiments/run_experiment.py experiments/configs/baseline_naive.yaml

Artifacts written to the config's output_path:
    metrics.json       — forecast metrics per destination + cost breakdown
    forecasts.parquet  — [destination_id, date, demand] from the winning model
    flows.parquet      — [origin_id, destination_id, period, flow]
    inventory.parquet  — [destination_id, period, inventory]
    config.yaml        — exact copy of the config that produced this run
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import polars as pl

from data.ingestion import Reader
from data.processing.data_processor import DataProcessor
from forecasting import create_forecasting_pipeline, AggregatedForecastingResult
from optimization import MultiPeriodOptimizer, MultiPeriodResult
from utils.config import ExperimentConfig, load_experiment_config
from utils.system_paths import get_project_root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_experiment(config_path: Path) -> None:
    project_root = get_project_root()
    config = load_experiment_config(project_root, config_path)

    logger.info("Starting experiment: %s", config.experiment_name)

    config.output_path.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    reader = Reader(config.dataset_path)
    raw_data = reader.read()
    clean_data = DataProcessor.process(raw_data)

    if clean_data.demand_history.is_empty():
        raise ValueError("Empty demand history after processing.")

    # --- Forecasting ---
    pipeline = create_forecasting_pipeline(config.forecasting)
    logger.info("Running forecasting pipeline...")
    forecast_result = pipeline.run(clean_data.demand_history)

    logger.info(
        "Forecasting complete: %d successful, %d failed",
        len(forecast_result.successful),
        len(forecast_result.failed),
    )
    for outcome in forecast_result.successful:
        logger.info(
            "  %s -> %s (WAPE=%.4f)",
            outcome.destination_id,
            outcome.selected.model_name,
            outcome.selected.metrics.get("wape", float("nan")),
        )

    demand_ts = forecast_result.export_forecasts()

    # --- Optimization ---
    planning_horizon = sorted(demand_ts["date"].unique().to_list())
    logger.info("Planning horizon: %d periods", len(planning_horizon))

    optimizer = MultiPeriodOptimizer(solver_name="GLOP")
    opt_result = optimizer.solve(
        demand_ts=demand_ts,
        origins_df=clean_data.origins,
        lanes_df=clean_data.lanes,
        destinations_df=clean_data.destinations,
        planning_horizon=planning_horizon,
    )

    logger.info("Optimization complete. Total cost: %.2f", opt_result.total_cost)

    # --- Save artifacts ---
    _save_metrics(config, forecast_result, opt_result)
    demand_ts.write_parquet(config.output_path / "forecasts.parquet")
    opt_result.flows.write_parquet(config.output_path / "flows.parquet")
    opt_result.inventory.write_parquet(config.output_path / "inventory.parquet")
    shutil.copy(config_path, config.output_path / "config.yaml")

    logger.info("Artifacts saved to: %s", config.output_path)


def _save_metrics(
    config: ExperimentConfig,
    forecast_result: AggregatedForecastingResult,
    opt_result: MultiPeriodResult,
) -> None:
    per_destination = {}
    for outcome in forecast_result.successful:
        selected_fr = next(
            fr for fr in outcome.results
            if fr.model_name == outcome.selected.model_name
        )
        per_destination[outcome.destination_id] = {
            "selected_model": outcome.selected.model_name,
            "wape": selected_fr.metrics.get("wape"),
            "mae": selected_fr.metrics.get("mae"),
            "rmse": selected_fr.metrics.get("rmse"),
        }

    n = len(per_destination)
    mean_wape = sum(v["wape"] for v in per_destination.values()) / n if n else None
    mean_mae = sum(v["mae"] for v in per_destination.values()) / n if n else None

    metrics = {
        "experiment_name": config.experiment_name,
        "n_successful_destinations": len(forecast_result.successful),
        "n_failed_destinations": len(forecast_result.failed),
        "per_destination": per_destination,
        "aggregated_forecast": {
            "mean_wape": mean_wape,
            "mean_mae": mean_mae,
        },
        "costs": {
            "total_cost": opt_result.total_cost,
            "transportation_cost": opt_result.transportation_cost,
            "holding_cost": opt_result.holding_cost,
        },
    }

    with open(config.output_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single experiment.")
    parser.add_argument("config", type=Path, help="Path to experiment YAML config.")
    args = parser.parse_args()
    run_experiment(args.config)
