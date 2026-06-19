"""Run a single experiment from a YAML config and persist all artifacts.

Usage (from project root):
    PYTHONPATH=src python experiments/run_experiment.py experiments/configs/<name>.yaml

Artifacts written to the config's output_path:
    metrics.json          — forecast metrics per destination + cost + service level
    forecasts.parquet     — [destination_id, date, demand] forecast signal used
    flows.parquet         — [origin_id, destination_id, period, flow]
    inventory.parquet     — [destination_id, period, inventory]
    realized_metrics.json — retroactive cost/service computed against actual demand
    config.yaml           — exact copy of the config that produced this run

Supports four scenario types via forecast_strategy / optimization_strategy:
    B00  naive  + naive   full SME baseline
    B01  naive  + dile    isolates optimization impact
    B10  dile   + naive   isolates forecasting impact
    B11  dile   + dile    full DILE system
"""

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Optional

import polars as pl

from data.ingestion import Reader
from data.processing.data_processor import DataProcessor
from forecasting import create_forecasting_pipeline, AggregatedForecastingResult
from optimization import MultiPeriodOptimizer, MultiPeriodResult
from experiment_config import ExperimentConfig, load_experiment_config
from naive_heuristic import (
    HeuristicResult,
    compute_lag1_forecast,
    run_naive_heuristic,
)
from actuals_evaluator import evaluate as evaluate_realized, save_realized_metrics
from utils.system_paths import get_project_root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_experiment(config_path: Path) -> None:
    project_root = get_project_root()
    config = load_experiment_config(project_root, config_path)

    logger.info(
        "Starting experiment: %s  [forecast=%s, opt=%s]",
        config.experiment_name,
        config.forecast_strategy,
        config.optimization_strategy,
    )

    config.output_path.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    reader = Reader(config.dataset_path)
    raw_data = reader.read()
    clean_data = DataProcessor.process(raw_data)

    if clean_data.demand_history.is_empty():
        raise ValueError("Empty demand history after processing.")

    # --- Step 1/3: Forecast demand signal ---
    forecast_result: Optional[AggregatedForecastingResult] = None

    if config.forecast_strategy == "naive":
        logger.info("Forecast strategy: naive (lag-1)")
        demand_ts = compute_lag1_forecast(
            clean_data.demand_history, config.test_periods
        )
        logger.info(
            "Lag-1 forecast: %d rows over %d destinations",
            demand_ts.height,
            demand_ts["destination_id"].n_unique(),
        )

    else:  # dile
        logger.info("Forecast strategy: DILE (per-destination model selection)")
        pipeline = create_forecasting_pipeline(config.forecasting)
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

    # --- Step 2/4: Optimization ---
    unmet_demand_pct = 0.0
    infeasibility_rate = 0.0

    if config.optimization_strategy == "naive":
        logger.info("Optimization strategy: naive (proportional capacity heuristic)")
        test_dates = sorted(demand_ts["date"].unique().to_list())
        actual_demand_ts = clean_data.demand_history.filter(
            pl.col("date").is_in(test_dates)
        )
        heuristic = run_naive_heuristic(
            forecast_ts=demand_ts,
            actual_demand_ts=actual_demand_ts,
            origins_df=clean_data.origins,
            lanes_df=clean_data.lanes,
            destinations_df=clean_data.destinations,
        )
        result: HeuristicResult | MultiPeriodResult = heuristic
        unmet_demand_pct = heuristic.unmet_demand_pct
        infeasibility_rate = heuristic.infeasibility_rate
        logger.info(
            "Heuristic complete. Transport cost: %.2f, unmet demand: %.2f%%",
            heuristic.transportation_cost,
            unmet_demand_pct,
        )

    else:  # dile
        logger.info("Optimization strategy: DILE (multi-period LP)")
        planning_horizon = sorted(demand_ts["date"].unique().to_list())
        logger.info("Planning horizon: %d periods", len(planning_horizon))

        optimizer = MultiPeriodOptimizer(solver_name="GLOP")
        result = optimizer.solve(
            demand_ts=demand_ts,
            origins_df=clean_data.origins,
            lanes_df=clean_data.lanes,
            destinations_df=clean_data.destinations,
            planning_horizon=planning_horizon,
        )
        logger.info("LP complete. Total cost: %.2f", result.total_cost)

    # --- Save artifacts ---
    _save_metrics(config, forecast_result, result, unmet_demand_pct, infeasibility_rate)
    demand_ts.write_parquet(config.output_path / "forecasts.parquet")
    result.flows.write_parquet(config.output_path / "flows.parquet")
    result.inventory.write_parquet(config.output_path / "inventory.parquet")
    shutil.copy(config_path, config.output_path / "config.yaml")
    logger.info("Artifacts saved to: %s", config.output_path)

    # --- Realized evaluation (retroactive against actual demand) ---
    realized = evaluate_realized(config.output_path)
    save_realized_metrics(realized, config.output_path)
    logger.info(
        "Realized evaluation: fill_rate=%.4f, realized_cost=%.2f",
        realized.fill_rate,
        realized.realized_total_cost,
    )


def _save_metrics(
    config: ExperimentConfig,
    forecast_result: Optional[AggregatedForecastingResult],
    result: "HeuristicResult | MultiPeriodResult",
    unmet_demand_pct: float,
    infeasibility_rate: float,
) -> None:
    per_destination: dict = {}

    if forecast_result is not None:
        for outcome in forecast_result.successful:
            selected_fr = next(
                fr
                for fr in outcome.results
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
        n_successful = len(forecast_result.successful)
        n_failed = len(forecast_result.failed)
    else:
        mean_wape = None
        mean_mae = None
        n_successful = None
        n_failed = None

    metrics = {
        "experiment_name": config.experiment_name,
        "forecast_strategy": config.forecast_strategy,
        "optimization_strategy": config.optimization_strategy,
        "n_successful_destinations": n_successful,
        "n_failed_destinations": n_failed,
        "per_destination": per_destination,
        "aggregated_forecast": {
            "mean_wape": mean_wape,
            "mean_mae": mean_mae,
        },
        "costs": {
            "total_cost": result.total_cost,
            "transportation_cost": result.transportation_cost,
            "holding_cost": result.holding_cost,
        },
        "service_level": {
            "unmet_demand_pct": unmet_demand_pct,
            "infeasibility_rate": infeasibility_rate,
        },
    }

    with open(config.output_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single experiment.")
    parser.add_argument("config", type=Path, help="Path to experiment YAML config.")
    args = parser.parse_args()
    run_experiment(args.config)
