"""Experiment configuration: schema and YAML loading for reproducible experiment runs."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml

from forecasting.config import PerDestinationConfig, _validate_per_destination_config

_VALID_FORECAST_STRATEGIES = {"naive", "dile"}
_VALID_OPTIMIZATION_STRATEGIES = {"naive", "dile"}


@dataclass
class ExperimentConfig:
    """Configuration for a single reproducible experiment run.

    Parameters
    ----------
    experiment_name : str
        Human-readable name used to identify the run and name the output directory.
    dataset_path : Path
        Absolute path to the dataset directory (4 parquet files).
    output_path : Path
        Absolute path where all artifacts for this run are written.
    forecast_strategy : str
        ``"naive"`` — lag-1 heuristic, computed outside the DILE forecasting pipeline.
        ``"dile"`` — per-destination model selection via ``PerDestinationForecastingPipeline``.
        Defaults to ``"dile"`` for backward compatibility with legacy configs.
    optimization_strategy : str
        ``"naive"`` — proportional capacity heuristic, no LP, no inventory tracking.
        ``"dile"`` — ``MultiPeriodOptimizer`` (OR-Tools GLOP LP).
        Defaults to ``"dile"`` for backward compatibility.
    test_periods : int
        Number of trailing dates in the demand history used as the planning/test horizon.
        Defaults to ``30``.
    forecasting : PerDestinationConfig | None
        Required when ``forecast_strategy == "dile"``. Ignored otherwise.
    """

    experiment_name: str
    dataset_path: Path
    output_path: Path
    forecast_strategy: str = "dile"
    optimization_strategy: str = "dile"
    test_periods: int = 30
    forecasting: Optional[PerDestinationConfig] = field(default=None)


def load_experiment_config(project_root: Path, config_path: Path) -> ExperimentConfig:
    """Load and validate an experiment configuration from a YAML file.

    Required top-level keys: ``experiment_name``, ``dataset_path``, ``output_path``.
    Optional keys: ``forecast_strategy`` (default ``"dile"``),
    ``optimization_strategy`` (default ``"dile"``), ``test_periods`` (default ``30``).
    ``per_destination_forecasting`` is required when ``forecast_strategy`` is ``"dile"``.

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

    forecast_strategy = raw.get("forecast_strategy", "dile")
    if forecast_strategy not in _VALID_FORECAST_STRATEGIES:
        raise ValueError(
            f"forecast_strategy must be one of {_VALID_FORECAST_STRATEGIES}, "
            f"got '{forecast_strategy}'."
        )

    optimization_strategy = raw.get("optimization_strategy", "dile")
    if optimization_strategy not in _VALID_OPTIMIZATION_STRATEGIES:
        raise ValueError(
            f"optimization_strategy must be one of {_VALID_OPTIMIZATION_STRATEGIES}, "
            f"got '{optimization_strategy}'."
        )

    test_periods_raw = raw.get("test_periods", 30)
    if not isinstance(test_periods_raw, int) or test_periods_raw < 1:
        raise ValueError(f"test_periods must be a positive integer, got {test_periods_raw!r}.")
    test_periods = test_periods_raw

    forecasting: Optional[PerDestinationConfig] = None
    if forecast_strategy == "dile":
        per_dest_raw = raw.get("per_destination_forecasting")
        if per_dest_raw is None:
            raise ValueError(
                "per_destination_forecasting section is required when "
                "forecast_strategy is 'dile'."
            )
        forecasting = _validate_per_destination_config(per_dest_raw)

    return ExperimentConfig(
        experiment_name=experiment_name,
        dataset_path=dataset_path,
        output_path=output_path,
        forecast_strategy=forecast_strategy,
        optimization_strategy=optimization_strategy,
        test_periods=test_periods,
        forecasting=forecasting,
    )
