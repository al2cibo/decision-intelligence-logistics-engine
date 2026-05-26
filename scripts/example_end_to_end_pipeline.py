"""
Example end-to-end pipeline for the Decision Intelligence Logistics Engine.

This script demonstrates the full workflow:
- data ingestion
- data processing
- forecasting
- model evaluation
- model selection
- demand aggregation
- optimization
- visualization
- per-destination forecasting (independent model selection per destination)

Usage:
    python example_end_to_end_pipeline.py --config configs/test_config.yaml
"""

import argparse
import logging
from datetime import date, timedelta

import numpy as np
import polars as pl

from data.ingestion import Reader
from data.processing.data_processor import DataProcessor

from forecasting.models.sarimax_forecaster import ARIMAForecaster
from forecasting.models.ets_forecaster import ETSForecaster
from forecasting.models.naive_forecaster import NaiveForecaster
from forecasting.models.rolling_window_forecaster import RollingWindowForecaster
from forecasting.models.seasonal_forecaster import SeasonalForecaster
from forecasting.pipeline.pipeline import ForecastingPipeline
from forecasting.evaluation.evaluator import Evaluator
from forecasting.results.forecast_extractor import ForecastExtractor
from forecasting.evaluation.model_selector import ModelSelector
from forecasting.pipeline.pipeline_factory import create_per_destination_pipeline_from_config

from postprocessing.metrics_summary import MetricsSummary
from postprocessing.visualization import VisualizationEngine

from optimization import Optimizer

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


def run_forecasting(clean_data, models, train_ratio=None):
    pipeline = ForecastingPipeline(models=models)
    results = pipeline.run(clean_data.demand_history, train_ratio=train_ratio)

    logger.info("Forecasting completed. Shape: %s", results.shape)
    return results


def evaluate_models(results, models, output_path):
    evaluator = Evaluator(results, "demand")
    metrics_summary = MetricsSummary(output_folder_path=output_path)

    for model in models:
        metrics_results = evaluator.compute_metrics(
            forecast_col_name=model.forecast_col
        )
        metrics_summary.collect(model_name=model.name, results=metrics_results)

    summary = metrics_summary.produce_summary()
    metrics_summary.save_summary(summary)

    logger.info("Model evaluation completed.")
    logger.info("Metrics summary (all %d models):\n%s", summary.shape[0], summary)
    return summary


def select_and_aggregate(results, summary, models, config=None):
    metric = "wape"
    if config is not None and config.forecasting is not None:
        metric = config.forecasting.metric
    logger.info("Using SINGLE-METRIC model selection on '%s'", metric)
    best_model_name = ModelSelector.select_best(summary, metric=metric)

    best_model = next(m for m in models if m.name == best_model_name)

    logger.info("Best model selected: %s", best_model_name)

    forecast_df = ForecastExtractor.extract(results, best_model.forecast_col)
    demand_df = ForecastExtractor.aggregate_average_demand(forecast_df)

    return best_model, demand_df


# ---------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------
def run_optimization(demand_df, origins_df, lanes_df):
    optimizer = Optimizer(solver_name="GLOP")

    result = optimizer.solve(
        demand_df=demand_df,
        origins_df=origins_df,
        lanes_df=lanes_df,
    )

    logger.info("Optimization completed. Total cost: %.4f", result.total_cost)
    return result


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------
def run_visualization(results, best_model, project_root):
    visualization = VisualizationEngine(df=results)

    visualization.produce_timeseries_plots(
        target_destination="D01",
        actuals_col_name="demand",
        predicted_col_name=best_model.forecast_col,
        save_fig_location=project_root / "data" / "output" / "plots",
    )

    logger.info("Visualization generated.")


# ---------------------------------------------------------------------
# Per-Destination Forecasting Pipeline
# ---------------------------------------------------------------------
def generate_synthetic_destination_data(
    destinations: list[str],
    n_days: int = 90,
    start_date: date | None = None,
) -> pl.DataFrame:
    """Generate synthetic demand data with distinct patterns per destination.

    Each destination gets a different demand profile:
    - D01: steady demand with weekly seasonality
    - D02: trending upward with noise
    - D03: high volatility with occasional spikes
    - D04: low, stable demand
    """
    if start_date is None:
        start_date = date(2024, 1, 1)

    rng = np.random.default_rng(seed=42)
    rows = []

    for dest in destinations:
        for day_offset in range(n_days):
            current_date = start_date + timedelta(days=day_offset)
            day_of_week = current_date.weekday()

            if dest == "D01":
                # Weekly seasonality: higher on weekdays
                base = 100 + (20 if day_of_week < 5 else -10)
                demand = base + rng.normal(0, 5)
            elif dest == "D02":
                # Upward trend
                base = 50 + day_offset * 0.5
                demand = base + rng.normal(0, 8)
            elif dest == "D03":
                # High volatility with spikes
                base = 80
                spike = 50 if rng.random() < 0.1 else 0
                demand = base + spike + rng.normal(0, 15)
            else:
                # Low stable demand
                demand = 30 + rng.normal(0, 3)

            rows.append({
                "date": current_date,
                "destination_id": dest,
                "demand": max(0.0, float(demand)),
            })

    return pl.DataFrame(rows).cast({"date": pl.Date, "demand": pl.Float64})


def run_per_destination_pipeline():
    """Demonstrate the per-destination forecasting pipeline on synthetic data."""
    logger.info("=" * 70)
    logger.info("PER-DESTINATION FORECASTING PIPELINE DEMO")
    logger.info("=" * 70)

    # Generate synthetic data with 4 destinations
    destinations = ["D01", "D02", "D03", "D04"]
    df = generate_synthetic_destination_data(destinations, n_days=90)

    logger.info(
        "Generated synthetic data: %d rows, %d destinations, %d days",
        df.height,
        len(destinations),
        90,
    )
    logger.info("Destinations: %s", destinations)
    logger.info("Data sample:\n%s", df.head(8))

    # Configure the per-destination pipeline
    config = PerDestinationConfig(
        model_names=[
            "naive_forecaster",
            "seasonal_forecaster",
            "rolling_window_forecaster",
        ],
        train_ratio=0.8,
        selection_metric="wape",
        max_workers=1,
        minimum_history_length=10,
        random_seed=42,
        model_params={
            "seasonal_forecaster": {"lag_value": 7},
            "rolling_window_forecaster": {"rolling_window": 7},
        },
    )

    # Create pipeline from config (validates model names against registry)
    pipeline = create_per_destination_pipeline_from_config(config)

    # Run the pipeline
    logger.info("Running per-destination pipeline...")
    result = pipeline.run(df)

    # Display results
    logger.info("-" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("-" * 70)
    logger.info(
        "Total destinations processed: %d successful, %d failed",
        len(result.successful),
        len(result.failed),
    )

    for outcome in result.successful:
        selected = outcome.selected
        logger.info(
            "  Destination %-4s -> Best model: %-28s (WAPE: %.4f, MAE: %.4f)",
            outcome.destination_id,
            selected.model_name,
            selected.metrics.get("wape", float("nan")),
            selected.metrics.get("mae", float("nan")),
        )

        # Show all model metrics for this destination
        if outcome.results:
            for fr in outcome.results:
                marker = " <-- SELECTED" if fr.model_name == selected.model_name else ""
                logger.info(
                    "    %-28s WAPE=%.4f  MAE=%.4f  RMSE=%.4f%s",
                    fr.model_name,
                    fr.metrics.get("wape", float("nan")),
                    fr.metrics.get("mae", float("nan")),
                    fr.metrics.get("rmse", float("nan")),
                    marker,
                )

    for outcome in result.failed:
        logger.info(
            "  Destination %-4s -> FAILED: %s",
            outcome.destination_id,
            outcome.error,
        )

    logger.info("-" * 70)
    return result


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

    # --- Forecasting ---
    models = [
        NaiveForecaster(),
        SeasonalForecaster(lag_value=7),
        RollingWindowForecaster(rolling_window=7),
        ETSForecaster(),
        ARIMAForecaster(),
    ]

    train_ratio = 0.8
    if config.forecasting is not None:
        train_ratio = config.forecasting.train_ratio

    n_rows = clean_data.demand_history.shape[0]
    split_idx = int(n_rows * train_ratio)
    logger.info(
        "Train/test split boundary: index %d of %d rows (train_ratio=%.2f)",
        split_idx,
        n_rows,
        train_ratio,
    )

    results = run_forecasting(clean_data, models, train_ratio=train_ratio)

    # --- Evaluation ---
    summary = evaluate_models(
        results,
        models,
        project_root / "data" / "output",
    )

    # --- Model selection + demand ---
    best_model, demand_df = select_and_aggregate(
        results, summary, models, config=config
    )

    # --- Optimization ---
    opt_result = run_optimization(
        demand_df=demand_df,
        origins_df=clean_data.origins,
        lanes_df=clean_data.lanes,
    )

    logger.info("Flows:\n%s", opt_result.flows)

    # --- Visualization ---
    run_visualization(results, best_model, project_root)

    # --- Per-Destination Forecasting ---
    per_dest_result = run_per_destination_pipeline()

    return opt_result, per_dest_result


if __name__ == "__main__":
    main()
