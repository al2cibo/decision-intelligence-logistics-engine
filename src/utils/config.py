"""Application-level configuration: data paths, high-level forecasting settings, and YAML loading."""

from dataclasses import dataclass
from pathlib import Path
import yaml

_KNOWN_METRICS = {"mae", "mse", "rmse", "mape", "wape"}


@dataclass
class DataConfig:
    input_path: Path


@dataclass
class ForecastingConfig:
    metric: str = "wape"
    train_ratio: float = 0.8


@dataclass
class Config:
    data: DataConfig
    forecasting: ForecastingConfig | None = None


def load_config(project_root: Path, config_path: Path) -> Config:
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

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
        if metric not in _KNOWN_METRICS:
            raise ValueError(
                f"Unrecognised metric '{metric}'. Must be one of {sorted(_KNOWN_METRICS)}."
            )
        train_ratio = forecasting_raw.get("train_ratio", 0.8)
        forecasting_config = ForecastingConfig(metric=metric, train_ratio=train_ratio)

    return Config(
        data=DataConfig(project_root / raw["data"]["input_path"]),
        forecasting=forecasting_config,
    )
