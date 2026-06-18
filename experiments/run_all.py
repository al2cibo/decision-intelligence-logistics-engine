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
        realized_path = results_root / experiment_name / "realized_metrics.json"

        if not metrics_path.exists():
            rows.append((experiment_name, "MISSING", "-", "-", "-", "-", "-"))
            continue

        with open(metrics_path) as f:
            m = json.load(f)

        agg = m.get("aggregated_forecast", {})
        costs = m.get("costs", {})
        planned_cost = costs.get("total_cost", float("nan"))

        if realized_path.exists():
            with open(realized_path) as f:
                r = json.load(f)
            realized_cost = r.get("realized_total_cost", float("nan"))
            fill_rate = r.get("fill_rate", float("nan"))
        else:
            realized_cost = float("nan")
            fill_rate = float("nan")

        rows.append(
            (
                experiment_name,
                "OK",
                f"{agg.get('mean_wape', float('nan')):.4f}",
                f"{planned_cost:.2f}",
                f"{realized_cost:.2f}",
                f"{fill_rate:.4f}",
            )
        )

    header = (
        f"{'Experiment':<30} {'Status':<8} {'WAPE':>8}"
        f" {'Planned Cost':>14} {'Realized Cost':>14} {'Fill Rate':>10}"
    )
    separator = "-" * len(header)
    print("\n" + separator)
    print(header)
    print(separator)
    for row in rows:
        print(
            f"{row[0]:<30} {row[1]:<8} {row[2]:>8}"
            f" {row[3]:>14} {row[4]:>14} {row[5]:>10}"
        )
    print(separator + "\n")

    _print_destination_breakdown(results_root)


def _print_destination_breakdown(results_root: Path) -> None:
    dest_header = (
        f"  {'Destination':<14} {'Transport':>12} {'Holding':>10}"
        f" {'Realized':>12} {'Demand':>10} {'Shortage':>10}"
        f" {'Fill Rate':>10} {'$/unit dem':>11} {'$/unit ful':>11}"
    )
    dest_separator = "  " + "-" * (len(dest_header) - 2)

    for config_path in CONFIGS:
        experiment_name = config_path.stem
        realized_path = results_root / experiment_name / "realized_metrics.json"
        if not realized_path.exists():
            continue

        with open(realized_path) as f:
            r = json.load(f)

        per_dest = r.get("per_destination", {})
        if not per_dest:
            continue

        print(f"  {experiment_name}")
        print(dest_separator)
        print(dest_header)
        print(dest_separator)
        for d_id in sorted(per_dest):
            d = per_dest[d_id]
            print(
                f"  {d_id:<14}"
                f" {d['transport_cost']:>12.2f}"
                f" {d['realized_holding_cost']:>10.2f}"
                f" {d['realized_total_cost']:>12.2f}"
                f" {d['total_actual_demand']:>10.2f}"
                f" {d['total_shortage']:>10.2f}"
                f" {d['fill_rate']:>10.4f}"
                f" {d['cost_per_unit_demanded']:>11.2f}"
                f" {d['cost_per_unit_fulfilled']:>11.2f}"
            )
        print(dest_separator + "\n")


if __name__ == "__main__":
    main()
