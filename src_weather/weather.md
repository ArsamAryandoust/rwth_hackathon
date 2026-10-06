# Weather preprocessing: hourly to 15-minute estimates

The weather station files contain hourly observations, not measured 15-minute
weather. The pipeline creates a UTC quarter-hour grid for aligning weather with
the smart-meter data. Sub-hour values are estimates; the source cannot reveal
the exact 15-minute variation.

## Resampling methods

The source dictionary identifies temperature, humidity, pressure, and wind as
hourly averages, dew point as an hourly instantaneous reading, and precipitation
and sunshine duration as hourly totals.

### Continuous readings: linear interpolation

For temperature, dew point, humidity, pressure, and wind, the pipeline estimates
intermediate values linearly between two valid hourly observations:

```text
x(t) = x0 + (x1 - x0) * (t - t0) / (t1 - t0)
```

For adjacent observations at 00:00 and 01:00, values at 00:15, 00:30, and 00:45
are 25%, 50%, and 75% of the way from `x0` to `x1`. The exact source values at
00:00 and 01:00 remain unchanged in the quarter-hour output. Linear interpolation
is a smooth estimate between hourly records; for variables that are hourly
averages, it does not mean each resulting value is a measured quarter-hour
average.

Interpolation is allowed only when both endpoints are present and exactly one
hour apart. The pipeline does not interpolate across missing values or gaps and
does not extrapolate before or after the available range.

### Precipitation and sunshine: conserve hourly totals

Precipitation and sunshine duration are accumulations, so linear interpolation
would be the wrong operation. For each complete hourly total `H`, the pipeline
assigns:

```text
quarter-hour estimate = H / 4
```

Thus an hourly precipitation total of 8 mm produces four estimated amounts of
2 mm, while an hourly sunshine duration of 0.8 hours produces four estimates of
0.2 hours. Summing the four quarter-hour estimates recovers the original hourly
total. Their units stay unchanged: mm per interval for precipitation and hours
of sunshine per interval for sunshine duration.

This equal allocation is a neutral conservation rule, not a claim that rain or
sunshine was uniform within that hour. The output also keeps the original hourly
total, unchanged at its source timestamp, in a sparse
`*_hourly_observation_15min` column. The derived estimates are in
`*_15min` columns. Use the hourly source column for source/audit purposes and the
quarter-hour estimate when a 15-minute aligned feature is needed.

No equation can recover actual within-hour rainfall or sunshine timing from one
hourly total. A higher-resolution sensor or upstream sub-hour source is needed
for measured quarter-hour totals.

## Missing data and time alignment

- Each station is resampled independently; values never cross stations.
- Timestamps are UTC and must be aligned to whole hours in the source.
- Missing source values remain missing. Missing hourly rows are represented on
  the quarter-hour grid but are not filled or interpolated across.
- Daily-only variables in the overview are not fabricated from hourly data.
- Each station's output runs from its first hourly timestamp through 45 minutes
  after its final hourly timestamp. There is no extrapolation beyond the last
  source observation.

## Build the Parquet file

From the repository root:

```bash
python -m src_weather
```

By default, this reads `data/weather_data_hourly`, writes
`data/weather_data_15min/weather_data_15min.parquet`, and creates visualizations
under `outputs/weather_15min`.

The `notebooks/weather_data_15min_showcase.ipynb` notebook demonstrates the
resampling and checks that hourly totals are preserved.
