"""Command line entry point: python -m src_weather [input_dir] [output_path]."""

import argparse
from pathlib import Path

from .io import read_weather_files
from .pipeline import write_weather_parquet
from .visualization import (
    plot_granularity_comparison,
    save_plotly_figures,
    save_weather_granularity_figures,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build quarter-hour weather parquet and Plotly charts")
    parser.add_argument("input_dir", nargs="?", default="data/weather_data_hourly")
    parser.add_argument("output_path", nargs="?", default="data/weather_data_15min/weather_data_15min.parquet")
    parser.add_argument("--plots", default="outputs/weather_15min")
    parser.add_argument("--weather-id", default=None)
    args = parser.parse_args()
    weather = write_weather_parquet(args.input_dir, args.output_path)
    save_plotly_figures(weather, args.plots, args.weather_id)
    hourly = read_weather_files(args.input_dir)
    granularity_paths = save_weather_granularity_figures(hourly, weather, args.plots, args.weather_id)
    comparison_station = args.weather_id or str(weather.Weather_ID.iloc[0])
    comparison_path = Path(args.plots) / "weather_granularity_comparison.html"
    plot_granularity_comparison(hourly, weather, comparison_station).write_html(
        comparison_path, include_plotlyjs="cdn"
    )
    print(f"Wrote {len(weather):,} rows to {Path(args.output_path)}")
    print(f"Stations: {weather.Weather_ID.nunique()}; source weather variables: {len(hourly.columns) - 2}")
    print(f"Hourly vs 15-minute comparison for {comparison_station}: {comparison_path}")
    print(f"Hourly weather graph: {granularity_paths[0]}")
    print(f"15-minute weather graph: {granularity_paths[1]}")


if __name__ == "__main__":
    main()
