"""End-to-end parquet creation and metadata helpers."""

from pathlib import Path

import pandas as pd

from .io import read_weather_files, read_variable_overview
from .resampling import to_quarter_hourly


def build_weather_table(input_dir: str | Path, *, frequency: str = "15min") -> pd.DataFrame:
    """Build a long station/time weather table at the requested frequency."""
    hourly = read_weather_files(input_dir)
    if frequency != "15min":
        raise ValueError("Only 15min output is currently supported")
    return to_quarter_hourly(hourly)


def write_weather_parquet(input_dir: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Create the complete quarter-hour table and save it as a parquet file."""
    table = build_weather_table(input_dir)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(target, index=False)
    return table


def load_overview(path: str | Path) -> pd.DataFrame:
    """Load source variable metadata for use alongside the weather table."""
    return read_variable_overview(path)
