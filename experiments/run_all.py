"""Run all four experiments in sequence and print a summary table.

Usage (from any directory):
    PYTHONPATH=<project_root>/src python experiments/run_all.py
"""

import json
import logging
from pathlib import Path

from run_experiment import run_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent

CONFIGS = [
    _HERE / "configs" / "baseline_naive.yaml",
    _HERE / "configs" / "baseline_global_ets.yaml",
    _HERE / "configs" / "model_selection.yaml",
    _HERE / "configs" / "scale_test.yaml",
]


def main() -> None:
    failed = []

    for config_path in CONFIGS:
        logger.info("=" * 60)
        logger.info("Running: %s", config_path.name)
        logger.info("=" * 60)
        try:
            run_experiment(config_path)
        except Exception as exc:
            logger.error("Experiment %s failed: %s", config_path.name, exc)
            failed.append(config_path.name)

    _print_summary()

    if failed:
        logger.error("The following experiments failed: %s", failed)
        raise SystemExit(1)


def _print_summary() -> None:
    results_root = _HERE / "results"
    rows = []

    for config_path in CONFIGS:
        experiment_name = config_path.stem
        metrics_path = results_root / experiment_name / "metrics.json"
        if not metrics_path.exists():
            rows.append((experiment_name, "MISSING", "-", "-", "-", "-"))
            continue

        with open(metrics_path) as f:
            m = json.load(f)

        agg = m.get("aggregated_forecast", {})
        costs = m.get("costs", {})
        rows.append(
            (
                experiment_name,
                "OK",
                f"{agg.get('mean_wape', float('nan')):.4f}",
                f"{costs.get('total_cost', float('nan')):.2f}",
                f"{costs.get('transportation_cost', float('nan')):.2f}",
                f"{costs.get('holding_cost', float('nan')):.2f}",
            )
        )

    header = f"{'Experiment':<30} {'Status':<8} {'WAPE':>8} {'Total':>12} {'Transport':>12} {'Holding':>10}"
    separator = "-" * len(header)
    print("\n" + separator)
    print(header)
    print(separator)
    for row in rows:
        print(
            f"{row[0]:<30} {row[1]:<8} {row[2]:>8} {row[3]:>12} {row[4]:>12} {row[5]:>10}"
        )
    print(separator + "\n")


if __name__ == "__main__":
    main()
