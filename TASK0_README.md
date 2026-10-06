# Task 0 · Baseline day-ahead forecast: design

Implementation: [`task0_baseline.ipynb`](task0_baseline.ipynb). Run it from the repository root (≈ 2 min, peak memory ≈ 4–5 GB). The raw files in `data/` are only read, never modified.

## 1. Forecasting task

| | |
|---|---|
| **Target** | `kWh_received_Total`: energy drawn from the grid per household per 15-min interval |
| **Forecast horizon** | All 15-min intervals of delivery day **D** (local time, Europe/Berlin) |
| **Issue time** | 12:00 on **D-1** (day-ahead gate closure) |
| **Information rule** | A feature may only use data that exists at the issue time. Meter readings of D-1 are not available yet, so the latest complete day is **D-2**. |
| **Reported quantity** | The **portfolio** (sum over all households), because that is what is procured. Household-level errors are reported as well. |

## 2. Data selection

| Step | Rule | Result |
|---|---|---|
| Valid households | > 365 days between first and last reading (`smart_meter_data_15min_overview.csv`) **and** a row in `meta_data.csv` | 379 households |
| No usable target | Households whose `kWh_received_Total` is empty everywhere (they only have a heat-pump sub-meter) | 4 dropped → **375** |
| Missing readings | Rows with empty `kWh_received_Total` are dropped. **The target is never imputed.** | |
| Impossible values | Readings > 10 kWh per 15 min (= 40 kW, beyond a household connection) are dropped | 2 readings |
| Reading the CSVs | Columns are selected **by name**, because their order differs between files | |

## 3. Weather

| Step | Rule |
|---|---|
| Variables | `Temperature_avg_hourly`, `Sunshine_duration_hourly`, `Humidity_avg_hourly`, `WindSpeed_hourly`. Pressure is not used (missing at 4 stations, little relevance). |
| Time alignment | A weather timestamp marks the **end** of the hour it describes, so every value is moved 1 h back to the start of its hour. |
| Short gaps | Linear interpolation within a station, up to 6 consecutive hours, only between known values |
| Stations without sunshine (HbsbG, ceOxS, sV3mR) | Filled with the **mean of the other stations at the same hour**, plus the flag `sunshine_filled = 1`. Never filled with 0, because 0 means "overcast". |
| Daily mean | `temperature_day_mean` = mean temperature of the local day the hour belongs to |
| Hourly → 15 min | Each 15-min interval gets the value of the hour it falls in (step function) |
| Household → station | Through `Weather_ID` in `households.csv` |

### Weather mode (`WEATHER_MODE`, top of the notebook)

The dataset contains **no weather forecasts**, so the notebook supports two settings:

| Mode | Weather used for interval *t* on day D | Meaning |
|---|---|---|
| `"observed"` **(default)** | Measured weather at *t − 2 days* (same hour on D-2) and D-2's daily mean temperature | Only information that exists at 12:00 on D-1. **Honest lower bound.** |
| `"perfect_forecast"` | Measured weather at *t* (day D itself) | Stands in for a perfect weather forecast. **Optimistic upper bound.** |

An operational system with a real weather forecast would perform between the two.

## 4. Features

| Group | Feature | Source / definition |
|---|---|---|
| Weather | `temperature`, `temperature_day_mean`, `sunshine`, `humidity`, `wind_speed` | Section 3, shifted according to `WEATHER_MODE` |
| | `sunshine_filled` | 1 if sunshine was taken from other stations |
| Calendar (local time) | `quarter_hour` | 0–95: hour × 4 + minute // 15 |
| | `weekday` | 0 = Monday |
| | `month` | 1–12 |
| History | `kwh_lag_2d` | Same household, same 15 min, 2 days earlier |
| | `kwh_lag_7d` | Same household, same 15 min, 7 days earlier |
| Group / intervention | `is_treatment` | 1 if `Group == "treatment"` |
| | `after_visit` | 1 if `AffectsTimePoint == "after visit"` (heat pump has been optimised) |
| PV | `has_pv` | 1 / 0 from `Installation_HasPVSystem`; **NaN if unknown** |
| Metadata (static) | `Survey_Building_LivingArea`, `Survey_Building_Residents` | numeric |
| | `Survey_Building_Type`, `Survey_HeatPump_Installation_Type` | categorical |
| | All other `Survey_*` columns (heat distribution, hot-water production, dryer, freezer, EV) | True / False → 1 / 0, unanswered → NaN |

**Missing feature values** (unanswered survey questions, lags that fall into a data gap) are left as NaN. The model handles them natively. Nothing is filled with 0.

**Not used:** lags of D-1 or day D (not available at issue time), the household ID itself, and the PV generation (not in the data).

## 5. Train / test split

| Set | Period (local time) | Rows |
|---|---|---|
| Train | 2019-05-20 → 2023-02-28 | 13.7 M |
| Test | 2023-03-01 → 2024-02-28 (≈ 12 months, includes a full winter) | 12.5 M |

- The split is **by date, identical for all households**. There is no random split, so test-period weather and seasons cannot leak into training.
- Households that start after 2023-03-01 appear only in the test set and are forecast from weather, calendar and metadata (lags only once they have 2 or 7 days of history).

## 6. Model

| Setting | Value |
|---|---|
| Model | scikit-learn `HistGradientBoostingRegressor`, **one global model** for all households |
| Training data | Random sample of **3 M** training rows (`random_state=0`), spread over all households and dates |
| Hyperparameters | `max_iter=300`, `learning_rate=0.1`, `categorical_features="from_dtype"`, other settings default (squared-error loss; early stopping on a random 10 % of the sample) |
| Output | Prediction clipped at ≥ 0 |

**Baselines:**
- *Same time 2 days ago*: `kwh_lag_2d`.
- *Same time 7 days ago*: `kwh_lag_7d`.

## 7. Evaluation

All methods are scored on **the same test rows**: those where both baseline lags exist (≈ 98 % of the test rows).

**Levels:**

| Level | Aggregation |
|---|---|
| Household · 15 min | each reading |
| Portfolio · 15 min | sum over households per timestamp |
| Household · day | sum per household per local day, **only complete days** (≥ 92 readings: 96, or 92/100 on daylight-saving days) |
| Portfolio · day | sum of the complete household-days per local day |

**Metrics** (error = forecast − actual):

| Metric | Definition | Reading |
|---|---|---|
| MAE | mean \|error\| | kWh |
| RMSE | √ mean error² | kWh, penalises large misses |
| nMAE % | MAE / mean actual × 100 | comparable across levels |
| bias % | Σ error / Σ actual × 100 | > 0: forecast too high (over-buying), < 0: too low (under-buying) |

## 8. Results (test year, run on 2026-10-06)

Portfolio level, nMAE:

| Model | 15 min, `observed` | 15 min, `perfect_forecast` | daily total, `observed` | daily total, `perfect_forecast` |
|---|---|---|---|---|
| Same time 2 days ago | 17.3 % | 17.3 % | 12.2 % | 12.2 % |
| Same time 7 days ago | 20.6 % | 20.6 % | 17.1 % | 17.1 % |
| **Gradient boosting** | **14.5 %** | **7.6 %** | **11.3 %** | **4.2 %** |

Bias of gradient boosting: +1.9 % (`observed`), −0.3 % (`perfect_forecast`).

Household level, gradient boosting: nMAE 64 % / 62 % per 15 min and 24 % / 21 % per day (`observed` / `perfect_forecast`). Single households are very noisy (a heat pump switching on or off), so the portfolio numbers are the relevant ones.

### What the results mean

- **With only observed weather**, the model beats both naive baselines. On daily totals it is only slightly better than "2 days ago", because it reacts to weather changes about 2 days late. It misses the start of cold snaps (under-buying) and over-buys when they end.
- **With day D's weather**, the daily error drops from 11.3 % to 4.2 %. That is the value of a good weather forecast for procurement, and an argument for Level 2.
- The 15-min errors are larger than the daily ones because timing errors within a day cancel in the daily sum. Day-ahead products are bought per hour or 15 min, so both levels matter.

## 9. Known limitations

1. **No weather forecasts in the data.** Both modes are approximations (Section 3).
2. Weather is a step function within each hour. Temperature could be interpolated.
3. Sunshine says *how long* the sun shone, not *how strongly*; sun elevation is not yet a feature.
4. The target is not scaled per household, so large households weigh more in training.
5. No separate validation period: hyperparameters are defaults and early stopping uses a random (not time-based) 10 % of the training sample.
6. The PV flag is unknown for many households, and about 1 in 10 PV-flagged households shows no PV pattern in its data.

## 10. Next steps

1. Interpolate temperature to 15 min, add sun elevation and a heating-degree feature `max(0, 15 − temperature)`.
2. Scale the target per household by its typical load (computed on training data only).
3. Hold out the last winter before the test period as a validation set for tuning.
4. Level 1: PV / no-PV or clustered models, scored with this exact evaluation for comparison.
5. Level 2/3: cost-based metric for over- vs under-buying, and quantile forecasts (`loss="quantile"`).
