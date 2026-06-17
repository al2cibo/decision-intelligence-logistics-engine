"""Experiment configuration: schema and YAML loading for reproducible experiment runs."""

from dataclasses import dataclass
from pathlib import Path
import yaml

from forecasting.config import PerDestinationConfig, _validate_per_destination_config


@dataclass
class ExperimentConfig:
    """Configuration for a single reproducible experiment run."""

    experiment_name: str
    dataset_path: Path
    output_path: Path
    forecasting: PerDestinationConfig


def load_experiment_config(project_root: Path, config_path: Path) -> ExperimentConfig:
    """Load and validate an experiment configuration from a YAML file.

    Expected top-level keys: ``experiment_name``, ``dataset_path``,
    ``output_path``, and ``per_destination_forecasting``.
    Paths are resolved relative to ``project_root``.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If any required field is missing or invalid.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(
            f"Experiment config must be a YAML mapping, got {type(raw).__name__}."
        )

    experiment_name = raw.get("experiment_name")
    if not experiment_name or not isinstance(experiment_name, str):
        raise ValueError("experiment_name is required and must be a non-empty string.")

    dataset_path_raw = raw.get("dataset_path")
    if not dataset_path_raw:
        raise ValueError("dataset_path is required.")
    dataset_path = project_root / dataset_path_raw

    output_path_raw = raw.get("output_path")
    if not output_path_raw:
        raise ValueError("output_path is required.")
    output_path = project_root / output_path_raw

    per_dest_raw = raw.get("per_destination_forecasting")
    if per_dest_raw is None:
        raise ValueError("per_destination_forecasting section is required.")
    forecasting = _validate_per_destination_config(per_dest_raw)

    return ExperimentConfig(
        experiment_name=experiment_name,
        dataset_path=dataset_path,
        output_path=output_path,
        forecasting=forecasting,
    )
