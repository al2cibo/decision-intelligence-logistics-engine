"""Run all 2×2 paper experiments in sequence and print a summary table.

Usage (from any directory):
    PYTHONPATH=<project_root>/src python experiments/run_all.py

Runs B00 → B01 → B10 → B11 in order and prints a consolidated summary table
covering forecast quality (WAPE), planned cost, realized cost, fill rate,
and unmet demand percentage.
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
    _HERE / "configs" / "B00_naive_forecast_naive_opt.yaml",
    _HERE / "configs" / "B01_naive_forecast_dile_opt.yaml",
    _HERE / "configs" / "B10_dile_forecast_naive_opt.yaml",
    _HERE / "configs" / "B11_dile_forecast_dile_opt.yaml",
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

    header = (
        f"{'Scenario':<36} {'F':>5} {'O':>5} {'WAPE':>8}"
        f" {'Planned$':>10} {'Realized$':>10} {'Holding$':>10} {'Unmet%':>8}"
    )
    separator = "-" * len(header)
    rows = []

    for config_path in CONFIGS:
        experiment_name = config_path.stem
        planning_path = results_root / experiment_name / "planning_metrics.json"
        realized_path = results_root / experiment_name / "realized_metrics.json"

        if not planning_path.exists():
            rows.append(
                (experiment_name, "?", "?", "MISSING", "-", "-", "-", "-")
            )
            continue

        with open(planning_path) as f:
            m = json.load(f)

        f_strat = m.get("forecast_strategy", "dile")[0].upper()
        o_strat = m.get("optimization_strategy", "dile")[0].upper()
        agg = m.get("aggregated_forecast", {})
        costs = m.get("costs", {})

        mean_wape = agg.get("mean_wape")
        wape_str = f"{mean_wape:.4f}" if mean_wape is not None else "  N/A"
        planned_cost = costs.get("total_cost", float("nan"))

        if realized_path.exists():
            with open(realized_path) as f:
                r = json.load(f)
            realized_cost = r.get("realized_total_cost", float("nan"))
            holding_cost = r.get("realized_holding_cost", float("nan"))
            fill_rate = r.get("fill_rate", float("nan"))
            unmet_pct = (1.0 - fill_rate) * 100
        else:
            realized_cost = float("nan")
            holding_cost = float("nan")
            unmet_pct = float("nan")

        rows.append(
            (
                experiment_name,
                f_strat,
                o_strat,
                wape_str,
                f"{planned_cost:.2f}",
                f"{realized_cost:.2f}",
                f"{holding_cost:.2f}",
                f"{unmet_pct:.2f}",
            )
        )

    print("\n" + separator)
    print(header)
    print(separator)
    for row in rows:
        print(
            f"{row[0]:<36} {row[1]:>5} {row[2]:>5} {row[3]:>8}"
            f" {row[4]:>10} {row[5]:>10} {row[6]:>10} {row[7]:>8}"
        )
    print(separator)

    _print_cost_decomposition(results_root)
    _print_destination_breakdown(results_root)


def _print_cost_decomposition(results_root: Path) -> None:
    """Print the 2×2 value decomposition from realized costs."""
    costs: dict[str, float] = {}
    for config_path in CONFIGS:
        name = config_path.stem
        path = results_root / name / "realized_metrics.json"
        if path.exists():
            with open(path) as f:
                r = json.load(f)
            costs[name] = r.get("realized_total_cost", float("nan"))

    b00 = costs.get("B00_naive_forecast_naive_opt", float("nan"))
    b01 = costs.get("B01_naive_forecast_dile_opt", float("nan"))
    b10 = costs.get("B10_dile_forecast_naive_opt", float("nan"))
    b11 = costs.get("B11_dile_forecast_dile_opt", float("nan"))

    print("\n  Value decomposition (realized costs):")
    print(f"    Optimization impact  (B00 - B01): {b00 - b01:+.2f}")
    print(f"    Forecasting impact   (B00 - B10): {b00 - b10:+.2f}")
    print(f"    Total DILE impact    (B00 - B11): {b00 - b11:+.2f}")
    print(f"    Interaction effect             : {(b00 - b11) - (b00 - b01) - (b00 - b10):+.2f}\n")


def _print_destination_breakdown(results_root: Path) -> None:
    dest_header = (
        f"  {'Destination':<14} {'Transport':>12} {'Holding':>10}"
        f" {'Realized$':>12} {'Demand':>10} {'Shortage':>10}"
        f" {'Fill%':>8} {'$/unit':>8}"
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
                f" {d['fill_rate'] * 100:>8.2f}"
                f" {d['cost_per_unit_demanded']:>8.2f}"
            )
        print(dest_separator + "\n")


if __name__ == "__main__":
    main()
