"""Application-level configuration: data paths, high-level forecasting settings, and YAML loading."""

from dataclasses import dataclass
from pathlib import Path
import yaml

from forecasting.config import PerDestinationConfig, _validate_per_destination_config


@dataclass
class DataConfig:
    input_path: Path


@dataclass
class Config:
    data: DataConfig
    per_destination_forecasting: PerDestinationConfig | None = None


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

    per_destination_config: PerDestinationConfig | None = None
    per_destination_raw = raw.get("per_destination_forecasting")
    if per_destination_raw is not None:
        per_destination_config = _validate_per_destination_config(per_destination_raw)

    return Config(
        data=DataConfig(project_root / raw["data"]["input_path"]),
        per_destination_forecasting=per_destination_config,
    )
