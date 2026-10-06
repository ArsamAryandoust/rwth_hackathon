"""Quarter-hour expansion using the temporal meaning of each weather variable."""

from collections.abc import Iterable

import pandas as pd
from pandas.api.types import is_numeric_dtype


HOURLY_TOTALS = {"Precipitation_total_hourly", "Sunshine_duration_hourly"}
INSTANTANEOUS_READINGS = {"DewPoint_hourly"}
HOURLY_AVERAGE_EXCEPTIONS = {"WindSpeed_hourly"}


def classify_variables(
    columns: Iterable[str],
) -> tuple[list[str], list[str], list[str]]:
    """Return hourly averages, instantaneous readings, and hourly totals."""
    columns = list(columns)
    totals = [column for column in columns if column in HOURLY_TOTALS]
    instantaneous = [
        column for column in columns if column in INSTANTANEOUS_READINGS
    ]
    averages = [
        column
        for column in columns
        if column.endswith("_avg_hourly")
        or column in HOURLY_AVERAGE_EXCEPTIONS
    ]
    classified = set(averages) | set(instantaneous) | set(totals)
    unsupported = [column for column in columns if column not in classified]
    if unsupported:
        raise ValueError(
            "Weather variables have no defined quarter-hour semantics: "
            f"{unsupported}"
        )
    return averages, instantaneous, totals


def quarter_total_column_name(column: str) -> str:
    """Name the estimated 15-minute amount derived from an hourly total."""
    if column.endswith("_hourly"):
        return f"{column[:-len('_hourly')]}_15min"
    return f"{column}_15min"


def quarter_source_column_name(
    column: str,
    *,
    is_hourly_total: bool = False,
) -> str:
    """Name the quarter-hour field retaining the source hourly observation."""
    if is_hourly_total:
        stem = column[:-len("_hourly")] if column.endswith("_hourly") else column
        return f"{stem}_hourly_observation_15min"
    if column.endswith("_hourly"):
        return f"{column[:-len('_hourly')]}_15min"
    return f"{column}_15min"


def to_quarter_hourly(hourly: pd.DataFrame) -> pd.DataFrame:
    """Expand hourly weather to UTC quarter-hours while retaining hourly values.

    Hourly average and instantaneous readings are linearly interpolated between
    valid adjacent hourly observations; values at source timestamps remain exact.
    Hourly precipitation and sunshine totals are evenly allocated across four
    quarters so hourly sums are conserved. Their original values remain in
    separate sparse observation columns. All sub-hour values are estimates, not
    measurements, because the source data is hourly.
    """
    required = {"Weather_ID", "Timestamp"}
    missing = required.difference(hourly.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if hourly.empty:
        raise ValueError("Hourly weather data is empty")
    if not isinstance(hourly["Timestamp"].dtype, pd.DatetimeTZDtype):
        raise TypeError("Timestamp must be timezone-aware and expressed as UTC")
    if hourly["Timestamp"].isna().any():
        raise ValueError("Timestamp contains missing values")
    if hourly["Weather_ID"].isna().any():
        raise ValueError("Weather_ID contains missing values")

    hourly = hourly.copy()
    hourly["Timestamp"] = hourly["Timestamp"].dt.tz_convert("UTC")
    weather_columns = [
        column for column in hourly.columns if column not in required
    ]
    averages, instantaneous, totals = classify_variables(weather_columns)
    for column in weather_columns:
        if not is_numeric_dtype(hourly[column].dtype):
            raise TypeError(f"Weather variable {column!r} must be numeric")
    if (
        (hourly["Timestamp"].dt.minute != 0).any()
        or (hourly["Timestamp"].dt.second != 0).any()
        or (hourly["Timestamp"].dt.microsecond != 0).any()
    ):
        raise ValueError("Hourly timestamps must be aligned to whole UTC hours")

    pieces: list[pd.DataFrame] = []
    for weather_id, group in hourly.groupby("Weather_ID", sort=False):
        source = group.drop(columns="Weather_ID").set_index("Timestamp").sort_index()
        if source.index.has_duplicates:
            raise ValueError(f"Duplicate timestamps for station {weather_id}")

        quarter_index = pd.date_range(
            source.index.min(),
            source.index.max() + pd.Timedelta(minutes=45),
            freq="15min",
        )
        expanded = source.reindex(quarter_index)
        floor_hours = quarter_index.floor("h")

        interpolated_columns = [*averages, *instantaneous]
        if interpolated_columns:
            candidates = expanded[interpolated_columns].interpolate(
                method="time",
                limit_area="inside",
            )
            source_timestamps = pd.Series(source.index, index=source.index)
            adjacent_hour = (
                source_timestamps.shift(-1) - source_timestamps
                == pd.Timedelta(hours=1)
            )
            is_quarter = quarter_index.minute != 0
            for column in interpolated_columns:
                valid_interval = (
                    source[column].notna()
                    & source[column].shift(-1).notna()
                    & adjacent_hour
                )
                allowed = (
                    valid_interval.reindex(floor_hours)
                    .to_numpy(dtype=bool, na_value=False)
                    & is_quarter
                )
                values = expanded[column].copy()
                values.loc[allowed] = candidates.loc[allowed, column]
                expanded[column] = values

        for column in totals:
            hourly_amount = source[column].reindex(floor_hours)
            expanded[quarter_total_column_name(column)] = (
                hourly_amount.to_numpy(dtype="float64", na_value=float("nan")) / 4
            )

        expanded.insert(0, "Weather_ID", weather_id)
        expanded.index.name = "Timestamp"
        pieces.append(expanded.reset_index())

    result = pd.concat(pieces, ignore_index=True)
    result = result.sort_values(["Weather_ID", "Timestamp"]).reset_index(drop=True)
    source_columns = [column for column in hourly.columns if column not in required]
    derived_total_columns = [quarter_total_column_name(column) for column in totals]
    result = result[["Weather_ID", "Timestamp", *source_columns, *derived_total_columns]]
    rename = {
        column: quarter_source_column_name(
            column,
            is_hourly_total=column in totals,
        )
        for column in source_columns
    }
    return result.rename(columns=rename)
