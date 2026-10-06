"""Plotly-only visualizations for every weather variable."""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .resampling import quarter_source_column_name, quarter_total_column_name


def variable_columns(weather: pd.DataFrame) -> list[str]:
    return [
        column
        for column in weather.columns
        if column not in {"Weather_ID", "Timestamp"}
        and not column.endswith("_hourly_observation_15min")
    ]


def _set_sequential_time_view(fig: go.Figure, timestamps: pd.Series, rows: int) -> None:
    """Open on the latest week and add controls for longer chronological ranges."""
    end = pd.Timestamp(timestamps.max())
    start = end - pd.Timedelta(days=7)
    fig.update_xaxes(range=[start, end])
    fig.update_xaxes(
        rangeselector={"buttons": [
            {"count": 1, "label": "1w", "step": "day", "stepmode": "backward"},
            {"count": 1, "label": "1m", "step": "month", "stepmode": "backward"},
            {"count": 6, "label": "6m", "step": "month", "stepmode": "backward"},
            {"count": 1, "label": "1y", "step": "year", "stepmode": "backward"},
            {"label": "All", "step": "all"},
        ]},
        rangeslider={"visible": True, "thickness": 0.06},
        row=rows,
        col=1,
    )


def plot_all_variables(weather: pd.DataFrame, weather_id: str | None = None) -> go.Figure:
    """Return a Plotly dashboard with an individual time-series panel per variable."""
    data = weather if weather_id is None else weather.loc[weather.Weather_ID == weather_id]
    # Bound chart payload size while retaining the full date span and all variables.
    sampled = []
    for _, station_data in data.groupby("Weather_ID", sort=False):
        stride = max(1, (len(station_data) + 2999) // 3000)
        sampled.append(station_data.iloc[::stride])
    data = pd.concat(sampled, ignore_index=True) if sampled else data
    columns = variable_columns(data)
    if data.empty or not columns:
        raise ValueError("Weather data must contain rows and at least one weather variable")
    fig = make_subplots(rows=len(columns), cols=1, shared_xaxes=True, vertical_spacing=0.015,
                        subplot_titles=columns)
    for row, column in enumerate(columns, start=1):
        for weather_station, station_data in data.groupby("Weather_ID", sort=False):
            fig.add_trace(go.Scatter(x=station_data.Timestamp, y=station_data[column], mode="lines",
                                     name=str(weather_station), legendgroup=str(weather_station),
                                     showlegend=(row == 1), connectgaps=False), row=row, col=1)
        fig.update_yaxes(title_text=column, row=row, col=1)
    fig.update_layout(title=f"Weather variables — {weather_id or 'all stations'}", height=max(500, 230 * len(columns)),
                      hovermode="x unified")
    fig.update_xaxes(title_text="Timestamp (UTC)", row=len(columns), col=1)
    _set_sequential_time_view(fig, data.Timestamp, len(columns))
    return fig


def plot_variable_coverage(weather: pd.DataFrame) -> go.Figure:
    """Plot missing-data percentage for every weather variable and station."""
    columns = variable_columns(weather)
    coverage = weather.groupby("Weather_ID", observed=True)[columns].apply(lambda group: group.isna().mean() * 100)
    tidy = coverage.reset_index().melt(id_vars="Weather_ID", var_name="Variable", value_name="Missing (%)")
    return px.bar(tidy, x="Variable", y="Missing (%)", color="Weather_ID", barmode="group",
                  title="Missing data by weather variable and station")


def plot_granularity_comparison(
    hourly: pd.DataFrame,
    quarter_hourly: pd.DataFrame,
    weather_id: str,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> go.Figure:
    """Compare original hourly observations with their 15-minute representation.

    Each variable gets its own subplot. Every quarter-hour observation in the
    selected interval is shown as a marker and line. By default the interval is
    the latest seven days; pass start/end to inspect another sequence. Hourly
    accumulation totals are compared with summed quarter totals on the left axis,
    while individual quarter totals are shown on the right axis.
    """
    source_all = hourly.loc[hourly.Weather_ID == weather_id].sort_values("Timestamp")
    fine_all = quarter_hourly.loc[quarter_hourly.Weather_ID == weather_id].sort_values("Timestamp")
    if source_all.empty or fine_all.empty:
        raise ValueError(f"No hourly and quarter-hour data found for station {weather_id!r}")
    window_end = pd.Timestamp(end) if end is not None else fine_all.Timestamp.max()
    if window_end.tzinfo is None:
        window_end = window_end.tz_localize("UTC")
    else:
        window_end = window_end.tz_convert("UTC")
    window_start = pd.Timestamp(start) if start is not None else window_end.floor("h") - pd.Timedelta(days=7)
    if window_start.tzinfo is None:
        window_start = window_start.tz_localize("UTC")
    else:
        window_start = window_start.tz_convert("UTC")
    source = source_all.loc[source_all.Timestamp.between(window_start, window_end)]
    fine = fine_all.loc[fine_all.Timestamp.between(window_start, window_end)]
    if source.empty or fine.empty:
        raise ValueError(f"No data for station {weather_id!r} between {window_start} and {window_end}")
    columns = [c for c in hourly.columns if c not in {"Weather_ID", "Timestamp"}]
    if not columns:
        raise ValueError("Hourly data has no weather variables")
    full_fine = fine
    fig = make_subplots(
        rows=len(columns), cols=1, shared_xaxes=True, vertical_spacing=0.015,
        subplot_titles=columns, specs=[[{"secondary_y": True}] for _ in columns],
    )
    first_total_row = next((i for i, c in enumerate(columns, start=1)
                            if c.startswith(("Precipitation_total_", "Sunshine_duration_"))), None)
    for row, column in enumerate(columns, start=1):
        is_total = column.startswith(("Precipitation_total_", "Sunshine_duration_"))
        quarter_column = quarter_total_column_name(column) if is_total else quarter_source_column_name(column)
        if is_total:
            quarter_sum = (full_fine.set_index("Timestamp")[quarter_column]
                           .resample("1h").sum(min_count=1).loc[window_start.floor("h"):window_end.floor("h")])
            fig.add_trace(go.Scattergl(x=fine.Timestamp, y=fine[quarter_column], mode="lines+markers",
                                       line={"width": 1, "color": "#e07a18"},
                                       marker={"size": 5, "color": "#e07a18"},
                                       name="15-minute total (right axis)", legendgroup="15-minute-total",
                                       showlegend=row == first_total_row, connectgaps=False),
                          row=row, col=1, secondary_y=True)
            fig.add_trace(go.Scattergl(x=quarter_sum.index, y=quarter_sum.to_numpy(), mode="lines",
                                       line={"width": 1.5, "color": "#16806a", "dash": "dash"},
                                       name="15-minute amounts summed hourly", legendgroup="hourly-rebuilt",
                                       showlegend=row == first_total_row, connectgaps=False),
                          row=row, col=1, secondary_y=False)
            fig.update_yaxes(title_text=column, row=row, col=1, secondary_y=False)
            fig.update_yaxes(title_text="15-minute amount", row=row, col=1, secondary_y=True)
        else:
            fig.add_trace(go.Scattergl(x=fine.Timestamp, y=fine[quarter_column], mode="lines+markers",
                                       line={"width": 1, "color": "#e07a18"},
                                       marker={"size": 5, "color": "#e07a18"},
                                       name="15-minute observations", legendgroup="15-minute",
                                       showlegend=row == 1, connectgaps=False), row=row, col=1,
                          secondary_y=False)
            fig.update_yaxes(title_text=column, row=row, col=1, secondary_y=False)
        fig.add_trace(go.Scattergl(x=source.Timestamp, y=source[column], mode="markers",
                                   marker={"size": 9, "symbol": "circle-open", "color": "#155a8a",
                                           "line": {"width": 2, "color": "#155a8a"}},
                                   name="Original hourly observations", legendgroup="hourly",
                                   showlegend=row == 1), row=row, col=1, secondary_y=False)
    fig.update_layout(title=f"Hourly vs 15-minute weather — {weather_id}<br><sup>{window_start} to {window_end}; all 15-minute points shown. Accumulation quarter values use the right axis.</sup>",
                      height=max(600, 230 * len(columns)), hovermode="x unified")
    fig.update_xaxes(title_text="Timestamp (UTC)", row=len(columns), col=1)
    fig.update_xaxes(range=[window_start, window_end])
    return fig


def plot_all_variables_matplotlib(
    weather: pd.DataFrame,
    weather_id: str | None = None,
):
    """Create inline-friendly Matplotlib time-series panels for every variable."""
    data = weather if weather_id is None else weather.loc[weather.Weather_ID == weather_id]
    sampled = []
    for _, station_data in data.groupby("Weather_ID", sort=False):
        stride = max(1, (len(station_data) + 2999) // 3000)
        sampled.append(station_data.iloc[::stride])
    data = pd.concat(sampled, ignore_index=True) if sampled else data
    columns = variable_columns(data)
    if data.empty or not columns:
        raise ValueError("Weather data must contain rows and at least one weather variable")
    fig, axes = plt.subplots(len(columns), 1, figsize=(14, max(3 * len(columns), 8)), sharex=True)
    for axis, column in zip(axes, columns):
        for station_id, station_data in data.groupby("Weather_ID", sort=False):
            axis.plot(station_data.Timestamp, station_data[column], linewidth=0.8, label=str(station_id))
        axis.set_ylabel(column, fontsize=8)
        axis.grid(alpha=0.25)
    axes[0].legend(title="Weather_ID", ncol=min(4, data.Weather_ID.nunique()))
    axes[-1].set_xlabel("Timestamp (UTC)")
    fig.suptitle(f"Weather variables — {weather_id or 'all stations'}")
    fig.tight_layout()
    return fig


def plot_variable_coverage_matplotlib(weather: pd.DataFrame):
    """Create an inline-friendly Matplotlib missing-data heatmap."""
    columns = variable_columns(weather)
    coverage = weather.groupby("Weather_ID", observed=True)[columns].apply(lambda group: group.isna().mean() * 100)
    fig, axis = plt.subplots(figsize=(max(10, len(columns) * 1.2), 4))
    image = axis.imshow(coverage.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    axis.set_xticks(range(len(columns)), columns, rotation=45, ha="right")
    axis.set_yticks(range(len(coverage.index)), coverage.index)
    axis.set_xlabel("Weather variable")
    axis.set_ylabel("Weather_ID")
    axis.set_title("Missing data (%) by weather variable and station")
    fig.colorbar(image, ax=axis, label="Missing (%)")
    fig.tight_layout()
    return fig


def plot_hourly_weather_variables(
    hourly: pd.DataFrame,
    weather_id: str | None = None,
) -> go.Figure:
    """Create a Plotly dashboard for all variables in the hourly source data."""
    return plot_all_variables(hourly, weather_id=weather_id)


def plot_15min_weather_variables(
    quarter_hourly: pd.DataFrame,
    weather_id: str | None = None,
) -> go.Figure:
    """Create a Plotly dashboard for the effective 15-minute weather series.

    The source-observation companion columns for hourly accumulations are omitted;
    their allocated 15-minute amount columns are plotted instead.
    """
    columns = [
        column for column in quarter_hourly.columns
        if column in {"Weather_ID", "Timestamp"}
        or not column.endswith("_hourly_observation_15min")
    ]
    return plot_all_variables(quarter_hourly[columns], weather_id=weather_id)


def save_plotly_figures(weather: pd.DataFrame, output_dir: str | Path, weather_id: str | None = None) -> list[Path]:
    """Save interactive HTML charts; no static plotting backend is used."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths = [target / "weather_variables.html", target / "weather_coverage.html"]
    selected_id = weather_id or str(sorted(weather.Weather_ID.astype(str).unique())[0])
    plot_all_variables(weather, selected_id).write_html(paths[0], include_plotlyjs="cdn")
    plot_variable_coverage(weather).write_html(paths[1], include_plotlyjs="cdn")
    return paths


def save_weather_granularity_figures(
    hourly: pd.DataFrame,
    quarter_hourly: pd.DataFrame,
    output_dir: str | Path,
    weather_id: str | None = None,
) -> list[Path]:
    """Write separate interactive hourly and 15-minute weather dashboards."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    hourly_path = target / "weather_variables_hourly.html"
    quarter_path = target / "weather_variables_15min.html"
    selected_id = weather_id or str(sorted(hourly.Weather_ID.astype(str).unique())[0])
    plot_hourly_weather_variables(hourly, selected_id).write_html(hourly_path, include_plotlyjs="cdn")
    plot_15min_weather_variables(quarter_hourly, selected_id).write_html(quarter_path, include_plotlyjs="cdn")
    return [hourly_path, quarter_path]
