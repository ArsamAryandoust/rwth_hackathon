import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from utils.forecast_level_0 import forecast_day, run_backtest


TIMEZONE = "Europe/Berlin"


class ForecastDayTests(unittest.TestCase):
    def test_uses_previous_week_slot_and_ignores_target_day_values(self) -> None:
        previous_week = pd.Timestamp("2024-01-08 12:00", tz=TIMEZONE).tz_convert("UTC")
        target_time = pd.Timestamp("2024-01-15 12:00", tz=TIMEZONE).tz_convert("UTC")
        history = pd.DataFrame(
            {
                "Timestamp": [previous_week, target_time],
                "kWh_received_Total": [2.5, 999.0],
            }
        )

        predictions = forecast_day(history, "2024-01-15")
        target_forecast = predictions.loc[predictions["Timestamp"] == target_time].iloc[0]

        self.assertEqual(len(predictions), 96)
        self.assertEqual(target_forecast["Forecast_kWh"], 2.5)
        self.assertEqual(target_forecast["ForecastMethod"], "same_slot_previous_week")

    def test_falls_back_to_median_of_earlier_weekly_slots(self) -> None:
        observations = [
            (pd.Timestamp("2024-01-01 12:00", tz=TIMEZONE), 4.0),
            (pd.Timestamp("2023-12-25 12:00", tz=TIMEZONE), 6.0),
            (pd.Timestamp("2024-01-15 12:00", tz=TIMEZONE), 999.0),
        ]
        history = pd.DataFrame(
            {
                "Timestamp": [timestamp.tz_convert("UTC") for timestamp, _ in observations],
                "kWh_received_Total": [value for _, value in observations],
            }
        )

        predictions = forecast_day(history, "2024-01-15")
        target_time = pd.Timestamp("2024-01-15 12:00", tz=TIMEZONE).tz_convert("UTC")
        target_forecast = predictions.loc[predictions["Timestamp"] == target_time].iloc[0]

        self.assertEqual(target_forecast["Forecast_kWh"], 5.0)
        self.assertEqual(target_forecast["ForecastMethod"], "median_previous_weeks")

    def test_local_day_has_correct_interval_count_across_dst(self) -> None:
        empty_history = pd.DataFrame(columns=["Timestamp", "kWh_received_Total"])

        spring_day = forecast_day(empty_history, "2024-03-31")
        autumn_day = forecast_day(empty_history, "2024-10-27")

        self.assertEqual(len(spring_day), 92)
        self.assertEqual(len(autumn_day), 100)


class BacktestTests(unittest.TestCase):
    def test_backtest_writes_predictions_metrics_and_daily_totals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            meter_dir = root / "data" / "15min"
            meter_dir.mkdir(parents=True)
            previous_week = pd.date_range(
                "2024-01-08 00:00", periods=96, freq="15min", tz=TIMEZONE
            )
            target_day = pd.date_range(
                "2024-01-15 00:00", periods=96, freq="15min", tz=TIMEZONE
            )
            meter = pd.DataFrame(
                {
                    "Timestamp": previous_week.append(target_day)
                    .tz_convert("UTC")
                    .astype(str),
                    "kWh_received_Total": [1.0] * 96 + [2.0] * 96,
                }
            )
            meter.to_csv(meter_dir / "123.csv", sep=";", index=False)

            output_paths = run_backtest(
                root / "data",
                root / "output",
                date(2024, 1, 15),
                date(2024, 1, 15),
            )

            with output_paths["metrics"].open(encoding="utf-8") as metrics_file:
                metrics = json.load(metrics_file)
            predictions = pd.read_csv(output_paths["predictions"])

            self.assertEqual(metrics["scored_intervals"], 96)
            self.assertEqual(metrics["interval_mae_kwh"], 1.0)
            self.assertEqual(len(predictions), 96)
            self.assertTrue(output_paths["daily_totals"].exists())
            self.assertTrue(output_paths["plot"].exists())


if __name__ == "__main__":
    unittest.main()