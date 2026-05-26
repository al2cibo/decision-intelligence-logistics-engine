"""
Main class adopted to read YAML configuration file.

The structure fo the config would be the following:

[1] Main field:
    [2] Secondary field
[1] Second main field

etc.

Hence, the idea would be to create a single dataclass for each main field, with an attribute
for each secondary field. Then merge all together using a main Config dataclass.

"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml

### ----- constants

KNOWN_METRICS = {"mae", "mse", "rmse", "mape", "wape"}

### ----- config related dataclasses


@dataclass
class DataConfig:
    input_path: Path


@dataclass
class ForecastingConfig:
    metric: str = "wape"
    train_ratio: float = 0.8


@dataclass
class PerDestinationConfig:
    """Configuration for the per-destination forecasting pipeline."""

    model_names: list[str]
    train_ratio: float = 0.8
    selection_metric: str = "wape"
    max_workers: int = 1
    minimum_history_length: int = 2
    random_seed: int = 42
    model_params: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class Config:
    data: DataConfig
    forecasting: ForecastingConfig | None = None
    per_destination_forecasting: PerDestinationConfig | None = None


### ----- validation helpers


def _validate_per_destination_config(raw: dict[str, Any]) -> PerDestinationConfig:
    """Parse and validate the per_destination_forecasting section from raw YAML dict.

    Raises
    ------
    ValueError
        If any parameter fails validation, with a clear message indicating which
        parameter failed and why.
    """
    # --- model_names ---
    model_names = raw.get("model_names")
    if model_names is None:
        raise ValueError(
            "per_destination_forecasting.model_names is required but missing."
        )
    if not isinstance(model_names, list):
        raise ValueError(
            "per_destination_forecasting.model_names must be a list of strings."
        )
    if len(model_names) < 1 or len(model_names) > 20:
        raise ValueError(
            f"per_destination_forecasting.model_names must contain 1-20 model names, "
            f"got {len(model_names)}."
        )
    for i, name in enumerate(model_names):
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"per_destination_forecasting.model_names[{i}] must be a non-empty string."
            )

    # --- train_ratio ---
    train_ratio = raw.get("train_ratio", 0.8)
    if not isinstance(train_ratio, (int, float)):
        raise ValueError(
            f"per_destination_forecasting.train_ratio must be a number, "
            f"got {type(train_ratio).__name__}."
        )
    train_ratio = float(train_ratio)
    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError(
            f"per_destination_forecasting.train_ratio must be in (0, 1), "
            f"got {train_ratio}."
        )

    # --- selection_metric ---
    selection_metric = raw.get("selection_metric", "wape")
    if selection_metric not in KNOWN_METRICS:
        raise ValueError(
            f"per_destination_forecasting.selection_metric '{selection_metric}' "
            f"is not recognised. Must be one of {sorted(KNOWN_METRICS)}."
        )

    # --- max_workers ---
    max_workers = raw.get("max_workers", 1)
    if not isinstance(max_workers, int):
        raise ValueError(
            f"per_destination_forecasting.max_workers must be an integer, "
            f"got {type(max_workers).__name__}."
        )
    if max_workers < 1 or max_workers > 128:
        raise ValueError(
            f"per_destination_forecasting.max_workers must be between 1 and 128, "
            f"got {max_workers}."
        )

    # --- minimum_history_length ---
    minimum_history_length = raw.get("minimum_history_length", 2)
    if not isinstance(minimum_history_length, int):
        raise ValueError(
            f"per_destination_forecasting.minimum_history_length must be an integer, "
            f"got {type(minimum_history_length).__name__}."
        )
    if minimum_history_length <= 0:
        raise ValueError(
            f"per_destination_forecasting.minimum_history_length must be > 0, "
            f"got {minimum_history_length}."
        )

    # --- random_seed ---
    random_seed = raw.get("random_seed", 42)
    if not isinstance(random_seed, int):
        raise ValueError(
            f"per_destination_forecasting.random_seed must be an integer, "
            f"got {type(random_seed).__name__}."
        )

    # --- model_params ---
    model_params = raw.get("model_params", {})
    if not isinstance(model_params, dict):
        raise ValueError(
            "per_destination_forecasting.model_params must be a mapping of "
            "model_name to parameter dict."
        )
    for key in model_params:
        if key not in model_names:
            raise ValueError(
                f"per_destination_forecasting.model_params contains key '{key}' "
                f"which is not in model_names {model_names}."
            )
        if not isinstance(model_params[key], dict):
            raise ValueError(
                f"per_destination_forecasting.model_params['{key}'] must be a dict, "
                f"got {type(model_params[key]).__name__}."
            )

    return PerDestinationConfig(
        model_names=model_names,
        train_ratio=train_ratio,
        selection_metric=selection_metric,
        max_workers=max_workers,
        minimum_history_length=minimum_history_length,
        random_seed=random_seed,
        model_params=model_params,
    )


### ----- main config reading method


def load_config(project_root: Path, config_path: Path) -> Config:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    try:
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(
            f"Malformed YAML in configuration file {config_path}: {e}"
        ) from e

    if not isinstance(raw, dict):
        raise ValueError(
            f"Configuration file must contain a YAML mapping, got {type(raw).__name__}."
        )

    forecasting_config: ForecastingConfig | None = None
    forecasting_raw = raw.get("forecasting")

    if forecasting_raw is not None:
        metric = forecasting_raw.get("metric", "wape")
        if metric not in KNOWN_METRICS:
            raise ValueError(
                f"Unrecognised metric '{metric}'. Must be one of {sorted(KNOWN_METRICS)}."
            )

        train_ratio = forecasting_raw.get("train_ratio", 0.8)
        forecasting_config = ForecastingConfig(
            metric=metric,
            train_ratio=train_ratio,
        )

    # --- per_destination_forecasting section ---
    per_destination_config: PerDestinationConfig | None = None
    per_dest_raw = raw.get("per_destination_forecasting")

    if per_dest_raw is not None:
        per_destination_config = _validate_per_destination_config(per_dest_raw)

    return Config(
        data=DataConfig(project_root / (raw["data"]["input_path"])),
        forecasting=forecasting_config,
        per_destination_forecasting=per_destination_config,
    )
