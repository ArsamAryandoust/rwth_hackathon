"""Input discovery and loading for the semicolon-delimited weather source files."""

from pathlib import Path

import pandas as pd


def read_weather_files(input_dir: str | Path) -> pd.DataFrame:
    """Read all station CSVs, keeping every source column and station identifier."""
    paths = sorted(Path(input_dir).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")
    frames = [pd.read_csv(path, sep=";") for path in paths]
    result = pd.concat(frames, ignore_index=True)
    required = {"Weather_ID", "Timestamp"}
    missing = required.difference(result.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    result["Timestamp"] = pd.to_datetime(result["Timestamp"], utc=True, errors="raise")
    if result.duplicated(["Weather_ID", "Timestamp"]).any():
        raise ValueError("Duplicate Weather_ID/Timestamp rows found")
    return result.sort_values(["Weather_ID", "Timestamp"]).reset_index(drop=True)


def read_variable_overview(path: str | Path) -> pd.DataFrame:
    """Load the supplied variable dictionary without dropping any metadata."""
    return pd.read_csv(path, sep=";")
