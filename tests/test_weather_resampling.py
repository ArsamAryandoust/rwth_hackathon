import unittest

import pandas as pd

from src_weather.resampling import (
    classify_variables,
    quarter_source_column_name,
    quarter_total_column_name,
    to_quarter_hourly,
)


class WeatherResamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hourly = pd.DataFrame(
            {
                "Weather_ID": ["station-a", "station-a"],
                "Timestamp": pd.date_range(
                    "2024-01-01 00:00",
                    periods=2,
                    freq="h",
                    tz="UTC",
                ),
                "Temperature_avg_hourly": [10.0, 14.0],
                "DewPoint_hourly": [2.0, 6.0],
                "Humidity_avg_hourly": [40.0, 60.0],
                "Precipitation_total_hourly": [8.0, 4.0],
                "Sunshine_duration_hourly": [1.0, 0.5],
                "WindSpeed_hourly": [2.0, 4.0],
            }
        )

    def test_classifies_by_variable_semantics(self) -> None:
        averages, instantaneous, totals = classify_variables(
            column
            for column in self.hourly.columns
            if column not in {"Weather_ID", "Timestamp"}
        )

        self.assertEqual(
            averages,
            ["Temperature_avg_hourly", "Humidity_avg_hourly", "WindSpeed_hourly"],
        )
        self.assertEqual(instantaneous, ["DewPoint_hourly"])
        self.assertEqual(
            totals,
            ["Precipitation_total_hourly", "Sunshine_duration_hourly"],
        )

    def test_hourly_points_are_preserved_and_continuous_values_interpolated(self) -> None:
        result = to_quarter_hourly(self.hourly)

        self.assertEqual(len(result), 8)
        self.assertEqual(
            result["Temperature_avg_15min"].iloc[:4].tolist(),
            [10.0, 11.0, 12.0, 13.0],
        )
        self.assertEqual(result["Temperature_avg_15min"].iloc[4], 14.0)
        self.assertTrue(result["Temperature_avg_15min"].iloc[5:].isna().all())
        self.assertEqual(result["DewPoint_15min"].iloc[:5].tolist(), [2, 3, 4, 5, 6])
        self.assertTrue(result["DewPoint_15min"].iloc[5:].isna().all())
        self.assertEqual(
            result.loc[result["Timestamp"].dt.minute == 0, "Temperature_avg_15min"].tolist(),
            self.hourly["Temperature_avg_hourly"].tolist(),
        )

    def test_hourly_totals_are_conserved_and_source_values_retained(self) -> None:
        result = to_quarter_hourly(self.hourly)

        for source_column in (
            "Precipitation_total_hourly",
            "Sunshine_duration_hourly",
        ):
            quarter_column = quarter_total_column_name(source_column)
            summed = result.groupby(result["Timestamp"].dt.floor("h"))[
                quarter_column
            ].sum(min_count=1)
            expected = self.hourly.set_index("Timestamp")[source_column]
            pd.testing.assert_series_equal(
                summed,
                expected,
                check_names=False,
                check_freq=False,
            )

            source_column_name = quarter_source_column_name(
                source_column,
                is_hourly_total=True,
            )
            retained = result.loc[result["Timestamp"].dt.minute == 0]
            self.assertEqual(
                retained[source_column_name].tolist(),
                self.hourly[source_column].tolist(),
            )

    def test_missing_hour_is_not_filled_or_interpolated_across(self) -> None:
        hourly = self.hourly.iloc[[0, 1]].copy()
        hourly.loc[1, "Timestamp"] = pd.Timestamp("2024-01-01 02:00", tz="UTC")
        hourly.loc[1, "Temperature_avg_hourly"] = 20.0
        hourly.loc[1, "DewPoint_hourly"] = 8.0
        result = to_quarter_hourly(hourly)

        missing_hour = result.loc[result["Timestamp"].dt.hour == 1]
        self.assertTrue(missing_hour["Temperature_avg_15min"].isna().all())
        self.assertTrue(missing_hour["DewPoint_15min"].isna().all())
        self.assertTrue(missing_hour["Precipitation_total_15min"].isna().all())

    def test_rejects_unknown_or_non_hourly_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "no defined quarter-hour semantics"):
            classify_variables(["Temperature_max_daily"])

        non_hourly = self.hourly.copy()
        non_hourly.loc[1, "Timestamp"] += pd.Timedelta(minutes=15)
        with self.assertRaisesRegex(ValueError, "whole UTC hours"):
            to_quarter_hourly(non_hourly)


if __name__ == "__main__":
    unittest.main()
