"""Reads the four required parquet files from a directory into an InputData."""

from pathlib import Path

import polars as pl

from data.input_data import InputData


class Reader:
    """Reads the four logistics parquet files from a single directory.

    Parameters
    ----------
    input_path : Path
        Directory containing ``demand_history.parquet``, ``destinations.parquet``,
        ``lanes.parquet``, and ``origins.parquet``.

    Raises
    ------
    FileNotFoundError
        If ``input_path`` does not exist, or if any of the four files is missing.
    """

    _REQUIRED_FILES = (
        "demand_history.parquet",
        "destinations.parquet",
        "lanes.parquet",
        "origins.parquet",
    )

    def __init__(self, input_path: Path) -> None:
        if not input_path.exists():
            raise FileNotFoundError(f"Input path not found: {input_path}")
        self.input_path = input_path

    def read(self) -> InputData:
        """Load all four tables and return them as an InputData."""
        return InputData(
            demand_history=self._read_parquet("demand_history.parquet"),
            destinations=self._read_parquet("destinations.parquet"),
            lanes=self._read_parquet("lanes.parquet"),
            origins=self._read_parquet("origins.parquet"),
        )

    def _read_parquet(self, filename: str) -> pl.DataFrame:
        file_path = self.input_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")
        return pl.read_parquet(file_path)
