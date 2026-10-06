# %% [markdown]
# # Household preprocessing
# Run cell by cell in VS Code: click "Run Cell" above each `# %%`, or press Shift+Enter.
# Goal: one combined, cleaned household table at 15-min resolution with Total kWh and a PV label.

# %% STEP 0 - Setup: imports and paths
from pathlib import Path
import time

import pandas as pd

# Folder layout (paths are relative to the repo root, so this works on every teammate's laptop)
REPO_ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
DATA_DIR = REPO_ROOT / "data"
RAW_15MIN_DIR = DATA_DIR / "15min"
META_DIR = DATA_DIR / "smart_meter_meta_data"
OUT_DIR = REPO_ROOT / "data_processed"          # our outputs go here (ignored by Git)
OUT_DIR.mkdir(exist_ok=True)

# Set to a small number (e.g. 10) for a quick test run; None = all 410 households
N_FILES = None

print("Repo root:      ", REPO_ROOT)
print("15-min folder:  ", RAW_15MIN_DIR, "| exists:", RAW_15MIN_DIR.exists())
print("Output folder:  ", OUT_DIR)

# %% STEP 1a - Find all household files
files = sorted(RAW_15MIN_DIR.glob("*.csv"))
if N_FILES:
    files = files[:N_FILES]
print(f"Found {len(files)} household files")   # expect 410

# %% STEP 1b - Read every file into one table
# Column types: small types save a lot of memory (~26 million rows in total)
DTYPES = {
    "Household_ID": "int64",
    "AffectsTimePoint": "category",   # before visit / after visit / unknown
    "Group": "category",              # treatment / control
    "kWh_received_Total": "float32",
    "kWh_received_HeatPump": "float32",
    "kWh_received_Other": "float32",
}

start = time.time()
frames = []
for i, f in enumerate(files, 1):
    df = pd.read_csv(f, sep=";", dtype=DTYPES)
    # Timestamps are UTC strings like "2023-02-25 00:15:00+00:00"
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, format="%Y-%m-%d %H:%M:%S%z")
    frames.append(df)
    if i % 50 == 0 or i == len(files):
        print(f"  read {i}/{len(files)} files  ({time.time() - start:.0f} s)")

raw = pd.concat(frames, ignore_index=True)
del frames
# concat can turn categories back into text when files have different categories -> fix
for col in ["AffectsTimePoint", "Group"]:
    raw[col] = raw[col].astype("category")

print(f"Done in {time.time() - start:.0f} s")

# %% STEP 1c - First look at what we loaded
print("Rows:        ", f"{len(raw):,}")
print("Households:  ", raw["Household_ID"].nunique())
print("Time range:  ", raw["Timestamp"].min(), "->", raw["Timestamp"].max())
print("Memory:      ", f"{raw.memory_usage(deep=True).sum() / 1e9:.2f} GB")
print("\nColumn types:\n", raw.dtypes)
print("\nMissing values per column (%):\n", (raw.isna().mean() * 100).round(2))
print("\nAffectsTimePoint counts:\n", raw["AffectsTimePoint"].value_counts())
print("\nGroup counts (rows):\n", raw["Group"].value_counts())
raw.head()

# %% STEP 1d - Save a combined raw copy so later steps don't have to re-read 410 files
raw_path = OUT_DIR / "households_15min_raw.parquet"
raw.to_parquet(raw_path, index=False)
print("Saved:", raw_path, f"({raw_path.stat().st_size / 1e6:.0f} MB)")


# %% [markdown]
# ## A. Clean the structure

# %% STEP 2a - Load the saved raw table (only needed if you restarted VS Code / the kernel)
if "raw" not in globals():
    raw = pd.read_parquet(OUT_DIR / "households_15min_raw.parquet")
    print("Loaded raw table from parquet:", f"{len(raw):,} rows")

# %% STEP 2b - Timestamp check: households vs. weather
# The weather data is hourly (:00). The weather teammate splits each hour into :00/:15/:30/:45.
# The households must use exactly the same slots, in the same time zone (UTC), so the merge matches.
WEATHER_DIR = DATA_DIR / "weather_data_hourly"

# (1) Weather: are all timestamps on full hours, UTC, without gaps?
w_rows = []
for f in sorted(WEATHER_DIR.glob("*.csv")):
    wt = pd.to_datetime(pd.read_csv(f, sep=";", usecols=["Timestamp"])["Timestamp"], utc=True)
    w_rows.append({
        "Weather_ID": f.stem,
        "first": wt.min(), "last": wt.max(),
        "not_on_full_hour": int(((wt.dt.minute != 0) | (wt.dt.second != 0)).sum()),
        "duplicates": int(wt.duplicated().sum()),
        "missing_hours": int((wt.max() - wt.min()) / pd.Timedelta("1h")) + 1 - wt.nunique(),
    })
weather_check = pd.DataFrame(w_rows)
print("Weather timestamps per station:\n", weather_check.to_string(index=False))

# (2) Households: are all timestamps on 15-min slots (:00/:15/:30/:45)?
ts = raw["Timestamp"]
off_grid = (ts.dt.minute % 15 != 0) | (ts.dt.second != 0)
print("\nHousehold timezone:", ts.dt.tz, "| weather timezone: UTC")
print("Household timestamps not on a 15-min slot:", int(off_grid.sum()))   # expect 0

# (3) Does the weather cover the whole household period?
print("Households:", ts.min(), "->", ts.max())
print("Weather:   ", weather_check["first"].min(), "->", weather_check["last"].max())
covered = (ts.min() >= weather_check["first"].max()) and (ts.max() <= weather_check["last"].min())
print("Weather covers the full household period:", covered)

# (4) Every :00/:15/:30/:45 slot of a household falls inside one weather hour (floor to the hour)
#     -> e.g. 10:15 belongs to weather hour 10:00. Check that every household hour exists in the weather.
hh_hours = ts.dt.floor("h").drop_duplicates()
w_hours = pd.to_datetime(pd.read_csv(next(WEATHER_DIR.glob("*.csv")), sep=";", usecols=["Timestamp"])["Timestamp"], utc=True)
print("Household hours without a weather hour:", int((~hh_hours.isin(w_hours)).sum()))   # expect 0

# %% STEP 2c - Remove off-grid timestamps and duplicates (every removed row is logged)
removed_log = []   # collects removed rows + reason -> saved as CSV in step 2d
n_before = len(raw)

# (1) off-grid timestamps like 10:07 have no 15-min slot (and no weather slot) -> remove + log
if off_grid.any():
    removed_log.append(raw[off_grid].assign(removal_reason="off_15min_grid"))
    raw = raw[~off_grid]

# (2) exact duplicates: the same row (all columns identical) appears more than once
exact_dup = raw.duplicated()
print("Exact duplicate rows:             ", int(exact_dup.sum()))
if exact_dup.any():
    removed_log.append(raw[exact_dup].assign(removal_reason="exact_duplicate"))
raw = raw[~exact_dup]

# (3) same house + same timestamp, but different values -> keep the first one, log the others
key_dup = raw.duplicated(subset=["Household_ID", "Timestamp"], keep="first")
print("Same house+time, different values:", int(key_dup.sum()))
if key_dup.any():
    print("  affected households:", raw.loc[key_dup, "Household_ID"].nunique())
    removed_log.append(raw[key_dup].assign(removal_reason="conflicting_duplicate"))
raw = raw[~key_dup]

raw = raw.sort_values(["Household_ID", "Timestamp"]).reset_index(drop=True)
print(f"Rows: {n_before:,} -> {len(raw):,}  (removed {n_before - len(raw):,})")

# %% STEP 2d - Write the log of removed rows (so nothing disappears silently)
log_path = OUT_DIR / "step2_removed_rows.csv"
if removed_log:
    removed = pd.concat(removed_log, ignore_index=True)
    removed.to_csv(log_path, index=False, sep=";")
    print(f"Logged {len(removed):,} removed rows ->", log_path)
    print(removed["removal_reason"].value_counts())
else:
    pd.DataFrame(columns=list(raw.columns) + ["removal_reason"]).to_csv(log_path, index=False, sep=";")
    print("Nothing removed - empty log written ->", log_path)

# %% STEP 3 - Build the full 15-minute grid for every household
# Before: missing intervals are simply absent rows -> invisible.
# After:  every house has one row per 15 min from its first to its last reading;
#         intervals with no data become rows with NaN, flagged with row_was_missing = True.
STATIC_COLS = ["Group"]                 # never changes within a house -> copy into new rows
FILL_COLS = ["AffectsTimePoint"]        # changes only at the visit -> carry the last known value forward

def to_full_grid(house: pd.DataFrame) -> pd.DataFrame:
    # floor/ceil = safety: the grid always starts and ends exactly on a 15-min mark
    full_index = pd.date_range(house["Timestamp"].min().floor("15min"),
                               house["Timestamp"].max().ceil("15min"), freq="15min")
    g = house.set_index("Timestamp").reindex(full_index)
    g.index.name = "Timestamp"
    g["row_was_missing"] = g["Household_ID"].isna()   # True = this interval did not exist in the file
    g["Household_ID"] = house["Household_ID"].iloc[0]
    for c in STATIC_COLS:
        g[c] = house[c].iloc[0]
    g[FILL_COLS] = g[FILL_COLS].ffill()
    return g.reset_index()

start = time.time()
grid = pd.concat(
    (to_full_grid(h) for _, h in raw.groupby("Household_ID", sort=False)),
    ignore_index=True,
)
for col in ["AffectsTimePoint", "Group"]:
    grid[col] = grid[col].astype("category")
print(f"Built grid in {time.time() - start:.0f} s")

added = int(grid["row_was_missing"].sum())
print(f"Rows: {len(raw):,} -> {len(grid):,}  (added {added:,} empty rows for missing intervals)")
print("Households:", grid["Household_ID"].nunique())   # still 410
print("Total kWh missing now (%):", round(grid["kWh_received_Total"].isna().mean() * 100, 2))

# %% STEP 3b - Save the gridded table
grid_path = OUT_DIR / "households_15min_grid.parquet"
grid.to_parquet(grid_path, index=False)
print("Saved:", grid_path, f"({grid_path.stat().st_size / 1e6:.0f} MB)")


# %% STEP 4a - Load the grid (only needed if you restarted VS Code / the kernel)
if "grid" not in globals():
    grid = pd.read_parquet(OUT_DIR / "households_15min_grid.parquet")
    print("Loaded grid from parquet:", f"{len(grid):,} rows")

# %% STEP 4b - Helper columns for the completeness check (temporary, not saved into the grid)
# Implausible value: > 10 kWh in 15 min = more than 40 kW average power. A normal house with a
# heat pump stays far below that (99.99% of all readings are below ~3.3 kWh per 15 min).
IMPLAUSIBLE_KWH = 10.0
ZERO_KWH = 0.005          # treat values below 0.005 kWh as "zero" (meter rounding)

total = grid["kWh_received_Total"]
hid = grid["Household_ID"]
t = grid["Timestamp"]

is_nan = total.isna()
value_empty = is_nan & ~grid["row_was_missing"]          # row existed, but kWh was empty
is_zero = total.le(ZERO_KWH)                              # NaN counts as False
is_neg = total.lt(0)
is_high = total.gt(IMPLAUSIBLE_KWH)
summer_midday = t.dt.month.between(4, 9) & t.dt.hour.between(10, 13)   # Apr-Sep, 10:00-13:59 UTC

# %% STEP 4c - Gap lengths: how long is each run of consecutive missing intervals?
# A "gap" = consecutive NaN rows within the same house. We number each run and measure its length.
new_run = is_nan.ne(is_nan.shift()) | hid.ne(hid.shift())   # a new run starts when NaN-status or house changes
run_id = new_run.cumsum()
gaps = (pd.DataFrame({"Household_ID": hid, "run_id": run_id})[is_nan]
          .groupby(["Household_ID", "run_id"]).size()
          .rename("gap_len").reset_index())
gaps["gap_hours"] = gaps["gap_len"] / 4                     # 4 intervals = 1 hour
gap_stats = gaps.groupby("Household_ID").agg(
    n_gaps=("gap_len", "size"),
    n_short_gaps_le_1h=("gap_len", lambda x: int((x <= 4).sum())),   # these will be interpolated in step 9
    n_long_gaps_gt_1h=("gap_len", lambda x: int((x > 4).sum())),
    longest_gap_hours=("gap_hours", "max"),
)
print("Gaps found:", f"{len(gaps):,}", "| of which <= 1 h:", int((gaps["gap_len"] <= 4).sum()))

# %% STEP 4d - One row per household: the quality report
tmp = pd.DataFrame({
    "Household_ID": hid, "Timestamp": t,
    "is_nan": is_nan, "row_was_missing": grid["row_was_missing"], "value_empty": value_empty,
    "is_zero": is_zero, "is_neg": is_neg, "is_high": is_high,
    "summer_midday": summer_midday, "summer_midday_valid": summer_midday & ~is_nan,
    "kwh": total,
})
report = tmp.groupby("Household_ID").agg(
    first_timestamp=("Timestamp", "min"),
    last_timestamp=("Timestamp", "max"),
    n_intervals=("Timestamp", "size"),
    n_missing_rows=("row_was_missing", "sum"),      # interval did not exist in the file
    n_empty_values=("value_empty", "sum"),           # row existed, kWh empty
    n_missing_total=("is_nan", "sum"),
    n_zero=("is_zero", "sum"),
    n_negative=("is_neg", "sum"),
    n_implausible_high=("is_high", "sum"),
    max_kwh_15min=("kwh", "max"),
    mean_kwh_15min=("kwh", "mean"),
    n_summer_midday_valid=("summer_midday_valid", "sum"),   # needed for PV detection (step 6)
)
report["n_days"] = ((report["last_timestamp"] - report["first_timestamp"]).dt.total_seconds() / 86400).round(1)
report["pct_missing_total"] = (report["n_missing_total"] / report["n_intervals"] * 100).round(2)
n_valid = report["n_intervals"] - report["n_missing_total"]
report["pct_zero"] = (report["n_zero"] / n_valid.where(n_valid > 0) * 100).round(2)
report["annual_kwh_estimate"] = (report["mean_kwh_15min"] * 96 * 365).round(0)   # typical yearly consumption
report = report.join(gap_stats).fillna({"n_gaps": 0, "n_short_gaps_le_1h": 0,
                                         "n_long_gaps_gt_1h": 0, "longest_gap_hours": 0})

# Cross-check with the organisers' overview file (number of days per household)
overview = pd.read_csv(META_DIR / "smart_meter_data_15min_overview.csv", sep=";")
report = report.join(overview.set_index("Household_ID")[["SMD_15min_TimeAvailable_NumberDays",
                                                         "SMD_15min_MeasurementsAvailable_Total"]])
report["days_diff_vs_overview"] = (report["n_days"] - report["SMD_15min_TimeAvailable_NumberDays"]).round(1)

report = report.reset_index()
print("Households in report:", len(report))

# %% STEP 4e - Summary: what does the data quality look like?
print("Missing Total per house (%):\n", report["pct_missing_total"].describe(percentiles=[.5, .75, .9, .95]).round(2))
print("\nHouses with ...")
print("  0% missing:                 ", int((report["pct_missing_total"] == 0).sum()))
print("  > 20% missing:              ", int((report["pct_missing_total"] > 20).sum()))
print("  100% missing (no Total):    ", int((report["pct_missing_total"] == 100).sum()))
print("  < 365 days of data:         ", int((report["n_days"] < 365).sum()))
print("  a gap longer than 7 days:   ", int((report["longest_gap_hours"] > 24 * 7).sum()))
print("  negative values:            ", int((report["n_negative"] > 0).sum()))
print("  implausible values (>10 kWh):", int((report["n_implausible_high"] > 0).sum()))
print("  no summer-midday data at all:", int((report["n_summer_midday_valid"] == 0).sum()))
print("\nDays: our count vs overview file, biggest difference:", report["days_diff_vs_overview"].abs().max())
print("\nAnnual consumption estimate (kWh/year):\n", report["annual_kwh_estimate"].describe().round(0))

print("\nWorst 10 houses by % missing:")
print(report.sort_values("pct_missing_total", ascending=False)
      [["Household_ID", "n_days", "pct_missing_total", "longest_gap_hours", "n_gaps"]].head(10).to_string(index=False))

# %% STEP 4f - Save the quality report (open it in Excel: Data -> From Text/CSV, separator ";")
report_path = OUT_DIR / "household_quality_report.csv"
report.to_csv(report_path, index=False, sep=";")
print("Saved:", report_path)


# %% [markdown]
# ## B. PV labelling

# %% STEP 5a - Load the PV flag from the metadata (households.csv)
# Installation_HasPVSystem: True = PV, False = no PV, empty = nobody recorded it (unknown)
households = pd.read_csv(META_DIR / "households.csv", sep=";")
print("Households in metadata:", len(households))
print("Raw values:\n", households["Installation_HasPVSystem"].value_counts(dropna=False))

# Turn the text True/False into 1/0; unknown stays missing (<NA>), never 0!
pv_text = households["Installation_HasPVSystem"].astype("string").str.strip().str.lower()
households["pv_label"] = pv_text.map({"true": 1, "false": 0}).astype("Int8")      # Int8 allows <NA>
households["pv_label_source"] = households["pv_label"].notna().map({True: "metadata", False: "unknown"})

pv_lookup = households.set_index("Household_ID")[["pv_label", "pv_label_source"]]

# Safety check: every household in our data must exist in the metadata
missing_in_meta = set(grid["Household_ID"].unique()) - set(pv_lookup.index)
print("Households in our data but not in households.csv:", len(missing_in_meta))   # expect 0

# %% STEP 5b - Add the PV label to the quality report (one row per house)
report = report.drop(columns=["pv_label", "pv_label_source"], errors="ignore")   # safe to re-run
report = report.merge(pv_lookup, left_on="Household_ID", right_index=True, how="left")
print("PV label per house:\n", report["pv_label_source"].value_counts())
print("\n", report["pv_label"].value_counts(dropna=False).rename({1: "PV (1)", 0: "no PV (0)"}))

# %% STEP 5c - Add the PV label to every row of the 15-min grid
# map() looks up each row's Household_ID in the small lookup table -> fast, no big merge needed
grid["pv_label"] = grid["Household_ID"].map(pv_lookup["pv_label"]).astype("Int8")
grid["pv_label_source"] = grid["Household_ID"].map(pv_lookup["pv_label_source"]).astype("category")
print("Grid rows per PV label source:\n", grid["pv_label_source"].value_counts())

# %% STEP 5d - First look: does the known PV label show up in the data?
# Preview for step 6: PV houses should have many more zero readings (self-supplied at midday)
print("Share of zero readings (%) by PV label:")
print(report.groupby(report["pv_label"].astype("string").fillna("unknown"))["pct_zero"]
            .describe(percentiles=[.25, .5, .75]).round(1))

print("\nPV label vs. experiment group (houses):")
print(pd.crosstab(report["pv_label"].astype("string").fillna("unknown"),
                  report.merge(households[["Household_ID", "Group"]], on="Household_ID")["Group"]))

# %% STEP 5e - Save the updated report
report.to_csv(OUT_DIR / "household_quality_report.csv", index=False, sep=";")
print("Updated:", OUT_DIR / "household_quality_report.csv")


# %% STEP 6a - Count readings per house, month and hour (the building block for the PV feature)
# Idea: a house WITHOUT PV always imports a little power -> Total is (almost) never 0.
#       A house WITH PV supplies itself at sunny midday -> grid import drops to exactly 0.
# We count, per house / month / UTC-hour: how many valid readings, and how many are ~0.
# Only 3 columns are used: Household_ID, Timestamp, kWh_received_Total (pv_label is NOT used here).
valid_total = grid["kWh_received_Total"].notna()
counts = (pd.DataFrame({
              "Household_ID": grid["Household_ID"],
              "month": grid["Timestamp"].dt.month,
              "hour": grid["Timestamp"].dt.hour,            # UTC! 10 UTC = 12:00 German summer time
              "valid": valid_total,
              "zero": grid["kWh_received_Total"].lt(ZERO_KWH),   # NaN counts as False
          })[valid_total]
          .groupby(["Household_ID", "month", "hour"])[["valid", "zero"]].sum()
          .reset_index())
print("Count table rows:", f"{len(counts):,}", "(houses x months x hours)")

def zero_share(months, hours):
    """Per house: % of valid readings in the given months/UTC-hours that are ~0, plus how many readings were used."""
    sel = counts[counts["month"].isin(months) & counts["hour"].isin(hours)]
    agg = sel.groupby("Household_ID")[["valid", "zero"]].sum()
    return pd.DataFrame({"share": agg["zero"] / agg["valid"] * 100, "n_used": agg["valid"]})

# %% STEP 6b - The chosen setting and the PV feature per house
PV_MONTHS = list(range(1, 13))      # all months (tested below: as good as or better than summer only)
PV_HOURS = [10, 11, 12, 13]          # 10:00-13:59 UTC = 12:00-15:59 German summer time (peak PV)
PV_THRESHOLD = 4.0                   # % zero readings above which a house counts as PV (chosen in step 7)
MIN_READINGS = 96                    # need at least 96 valid readings in the window (~6 days) to judge

feat = zero_share(PV_MONTHS, PV_HOURS)
report = report.drop(columns=["pv_zero_share", "pv_n_readings_used"], errors="ignore")   # safe to re-run
report = report.merge(feat.rename(columns={"share": "pv_zero_share", "n_used": "pv_n_readings_used"}),
                      left_on="Household_ID", right_index=True, how="left")
report["pv_n_readings_used"] = report["pv_n_readings_used"].fillna(0).astype(int)
print("pv_zero_share (%) per house:\n", report["pv_zero_share"].describe().round(1))

# %% STEP 7a - Why this window? Compare windows on the houses with a KNOWN label only
# The unknown houses are NOT used here - otherwise the rule would be graded with its own guesses.
# Use the metadata label (if step 8 already ran, the original is kept in pv_label_metadata)
known_label = (report.set_index("Household_ID")["pv_label_metadata"] if "pv_label_metadata" in report
               else report.set_index("Household_ID")["pv_label"]).dropna().astype(bool)
THRESHOLDS = [1, 2, 3, 4, 5, 6, 7.5, 10, 15, 20]

def evaluate(share: pd.Series, threshold: float) -> dict:
    k = pd.DataFrame({"share": share}).join(known_label.rename("pv"), how="inner").dropna()
    pred = k["share"] > threshold
    tp, fp = int((pred & k["pv"]).sum()), int((pred & ~k["pv"]).sum())
    fn, tn = int((~pred & k["pv"]).sum()), int((~pred & ~k["pv"]).sum())
    return {"threshold_%": threshold, "houses_judged": len(k),
            "accuracy_%": round((tp + tn) / len(k) * 100, 1),
            "PV_found": f"{tp}/{tp + fn}", "false_PV": f"{fp}/{fp + tn}",
            "precision_%": round(tp / (tp + fp) * 100, 1) if tp + fp else None,   # said PV -> really PV
            "recall_%": round(tp / (tp + fn) * 100, 1) if tp + fn else None}      # real PV -> found

WINDOWS = {
    "Apr-Sep, 10-13 UTC":            (range(4, 10), range(10, 14)),
    "Apr-Sep, 9-14 UTC":             (range(4, 10), range(9, 15)),
    "Apr-Sep, 8-15 UTC":             (range(4, 10), range(8, 16)),
    "Apr-Sep, 7-16 UTC (evening)":   (range(4, 10), range(7, 17)),
    "May-Aug, 10-13 UTC":            (range(5, 9), range(10, 14)),
    "Mar-Oct, 10-13 UTC":            (range(3, 11), range(10, 14)),
    "All year, 10-13 UTC (chosen)":  (range(1, 13), range(10, 14)),
}
rows = []
for name, (months, hours) in WINDOWS.items():
    f = zero_share(list(months), list(hours))
    share = f["share"][f["n_used"] >= MIN_READINGS]
    best = max((evaluate(share, th) for th in THRESHOLDS), key=lambda r: r["accuracy_%"])
    rows.append({"window": name, **best})
window_table = pd.DataFrame(rows).rename(columns={"threshold_%": "best_threshold_%"})
print("Window comparison (best threshold per window, known houses only):")
print(window_table.to_string(index=False))

# %% STEP 7b - Which threshold? All thresholds for the chosen window
chosen_share = feat["share"][feat["n_used"] >= MIN_READINGS]
threshold_table = pd.DataFrame([evaluate(chosen_share, th) for th in THRESHOLDS])
print(f"Thresholds for the chosen window (months {PV_MONTHS[0]}-{PV_MONTHS[-1]}, "
      f"UTC hours {PV_HOURS[0]}-{PV_HOURS[-1]}):")
print(threshold_table.to_string(index=False))
print(f"\n-> Using PV_THRESHOLD = {PV_THRESHOLD}%")

# %% STEP 7c - Where does the rule get it wrong? (known houses only - good to show honestly)
k = report.set_index("Household_ID")[["pv_zero_share", "pv_n_readings_used"]].join(known_label.rename("pv"), how="inner")
k = k[k["pv_n_readings_used"] >= MIN_READINGS]
k["rule_says_pv"] = k["pv_zero_share"] > PV_THRESHOLD
print("Confusion matrix (rows = metadata label, columns = rule):")
print(pd.crosstab(k["pv"].map({True: "PV", False: "no PV"}), k["rule_says_pv"].map({True: "rule: PV", False: "rule: no PV"})))
print("\nKnown PV houses the rule misses (possible battery / small PV / wrong label):")
print(k[k["pv"] & ~k["rule_says_pv"]].sort_values("pv_zero_share")["pv_zero_share"].round(2).to_string())
print("\nKnown non-PV houses the rule calls PV:")
print(k[~k["pv"] & k["rule_says_pv"]]["pv_zero_share"].round(2).to_string())

# %% STEP 8 - Fill the unknown PV labels (known labels are NEVER overwritten)
# pv_label_metadata = original label from households.csv (1 / 0 / <NA>)
# pv_label_rule     = what the zero-share rule says, for every house with enough data (1 / 0 / <NA>)
# pv_label          = FINAL label: metadata if known, otherwise the rule
# pv_label_source   = metadata / inferred / unknown (unknown = no label and too little data to judge)
if "pv_label_metadata" not in report:
    report["pv_label_metadata"] = report["pv_label"]
enough = report["pv_n_readings_used"] >= MIN_READINGS
report["pv_label_rule"] = (report["pv_zero_share"] > PV_THRESHOLD).astype("Int8").where(enough)
report["pv_label"] = report["pv_label_metadata"].fillna(report["pv_label_rule"]).astype("Int8")
report["pv_label_source"] = "unknown"
report.loc[report["pv_label_metadata"].notna(), "pv_label_source"] = "metadata"
report.loc[report["pv_label_metadata"].isna() & report["pv_label_rule"].notna(), "pv_label_source"] = "inferred"

print("Final PV label source:\n", report["pv_label_source"].value_counts())
print("\nFinal PV label by source:")
print(pd.crosstab(report["pv_label_source"], report["pv_label"].astype("string").fillna("<NA>")))

# Copy the final label to every row of the 15-min grid
lookup = report.set_index("Household_ID")
grid["pv_label"] = grid["Household_ID"].map(lookup["pv_label"]).astype("Int8")
grid["pv_label_source"] = grid["Household_ID"].map(lookup["pv_label_source"]).astype("category")
print("\nGrid rows per final PV label:\n", grid["pv_label"].value_counts(dropna=False))

# %% STEP 8b - Save report + the PV tables (for the slides)
report.to_csv(OUT_DIR / "household_quality_report.csv", index=False, sep=";")
window_table.to_csv(OUT_DIR / "pv_window_comparison.csv", index=False, sep=";")
threshold_table.to_csv(OUT_DIR / "pv_threshold_comparison.csv", index=False, sep=";")
print("Saved: household_quality_report.csv, pv_window_comparison.csv, pv_threshold_comparison.csv in", OUT_DIR)


# %% STEP 8d - Save the ONE shared household file (state after step 8)
# This is the single file the team works with. Later steps (9-12) load it, add their columns,
# and save it again under the same name -> there is always exactly one current version.
SHARED_PATH = OUT_DIR / "households_15min.parquet"
SUMMARY_PATH = OUT_DIR / "household_summary.csv"

grid.to_parquet(SHARED_PATH, index=False)
report.to_csv(SUMMARY_PATH, index=False, sep=";")

check = pd.read_parquet(SHARED_PATH)                       # re-load to make sure the file is complete
print("Saved:", SHARED_PATH, f"({SHARED_PATH.stat().st_size / 1e6:.0f} MB)")
print("Rows:", f"{len(check):,}", "| households:", check["Household_ID"].nunique())
print("Columns:", list(check.columns))
print("PV label per household:\n", check.drop_duplicates("Household_ID")["pv_label"].value_counts(dropna=False))
print("Saved:", SUMMARY_PATH, "(one row per household)")
del check

# %% [markdown]
# ## Figures for steps 1-8 (saved as PNG in data_processed/figures)

# %% STEP 8c - Figure setup (shared style: PV = orange, no PV = blue, everywhere)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

C_PV, C_NOPV = "#eb6834", "#2a78d6"          # colour-blind safe pair (validated)
C_TEXT, C_GRID, C_WINDOW = "#52514e", "#e6e5e1", "#f3f2ee"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": C_TEXT, "axes.labelcolor": C_TEXT,
    "xtick.color": C_TEXT, "ytick.color": C_TEXT,
    "axes.grid": True, "grid.color": C_GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False,
})

def save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print("Saved:", path)
    plt.show()

# known labels from the metadata (used ONLY for colouring / validation plots)
meta = report.set_index("Household_ID")["pv_label_metadata"]
known_pv = meta[meta == 1].index
known_nopv = meta[meta == 0].index

# %% STEP 8c-1 - Data availability timeline: one row per house, one column per day
daily = (grid.assign(day=grid["Timestamp"].dt.floor("D"),
                     has=grid["kWh_received_Total"].notna())
             .groupby(["Household_ID", "day"])["has"].mean()
             .unstack("day"))                                    # rows = houses, columns = days
daily = daily.loc[report.sort_values("first_timestamp")["Household_ID"]]   # sort houses by start date

fig, ax = plt.subplots(figsize=(10, 5.5))
cmap = plt.get_cmap("Blues").copy()
cmap.set_bad("#f3f2ee")                                          # not measured at all = light grey
days = mdates.date2num(daily.columns.tz_localize(None).to_pydatetime())
im = ax.imshow(daily.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap=cmap,
               vmin=0, vmax=1, extent=[days[0], days[-1], len(daily), 0])
ax.xaxis_date(); ax.grid(False)
ax.set_ylabel("Households (sorted by start of measurement)")
ax.set_title("Data availability per household and day", loc="left", fontweight="bold")
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
cb.set_label("Share of 15-min readings with a value")
ax.text(0, -0.1, "Grey = outside the household's measurement period. White/light streaks = outages.",
        transform=ax.transAxes, color=C_TEXT, fontsize=9)
save(fig, "01_data_availability.png")

# %% STEP 8c-2 - Histogram: % missing Total per house
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(report["pct_missing_total"].dropna(), bins=np.arange(0, 102.5, 2.5),
        color=C_NOPV, edgecolor="white", linewidth=1.5)
med = report["pct_missing_total"].median()
ax.axvline(20, color=C_TEXT, linestyle="--", linewidth=1)
ax.text(20.8, ax.get_ylim()[1] * 0.9, "20 % (flag threshold)", color=C_TEXT, fontsize=9)
ax.set_xlabel("Missing Total readings per household (%)")
ax.set_ylabel("Households")
ax.set_title(f"Most households are almost complete (median {med:.1f} % missing)", loc="left", fontweight="bold")
save(fig, "02_missing_per_house.png")

# %% STEP 8c-3 - One sunny day: a PV house vs a house without PV (same weather station)
# Pick a clear PV house and a clear non-PV house from the same weather station,
# then the summer day on which the PV house has the most zero readings (both houses complete that day).
wid = households.set_index("Household_ID")["Weather_ID"]
cand = report.assign(Weather_ID=report["Household_ID"].map(wid))
cand = cand[cand["pv_n_readings_used"] >= 2000]
pv_c = cand[cand["Household_ID"].isin(known_pv)].sort_values("pv_zero_share", ascending=False)
station = pv_c["Weather_ID"].iloc[0]
house_pv = pv_c[pv_c["Weather_ID"] == station]["Household_ID"].iloc[len(pv_c[pv_c["Weather_ID"] == station]) // 4]
nopv_c = cand[cand["Household_ID"].isin(known_nopv) & (cand["Weather_ID"] == station)]
house_nopv = nopv_c.sort_values("pv_zero_share")["Household_ID"].iloc[len(nopv_c) // 2]

two = grid[grid["Household_ID"].isin([house_pv, house_nopv])].copy()
two["day"] = two["Timestamp"].dt.floor("D")
summer = two[two["Timestamp"].dt.month.between(5, 8)]
per_day = summer.pivot_table(index="day", columns="Household_ID", values="kWh_received_Total", aggfunc="count")
complete_days = per_day[(per_day[house_pv] == 96) & (per_day[house_nopv] == 96)].index
zeros = (summer[(summer["Household_ID"] == house_pv) & summer["day"].isin(complete_days)]
         .assign(z=lambda d: d["kWh_received_Total"].lt(ZERO_KWH)).groupby("day")["z"].sum())
best_day = zeros.idxmax()

fig, ax = plt.subplots(figsize=(9, 4))
ax.axvspan(10, 14, color=C_WINDOW, zorder=0)
ax.text(12, 0.02, "PV window\n10-14 UTC", ha="center", va="bottom", color=C_TEXT, fontsize=8, transform=ax.get_xaxis_transform())
for hid_, col, lab in [(house_nopv, C_NOPV, "no PV"), (house_pv, C_PV, "PV")]:
    d = two[(two["Household_ID"] == hid_) & (two["day"] == best_day)]
    x = d["Timestamp"].dt.hour + d["Timestamp"].dt.minute / 60
    ax.step(x, d["kWh_received_Total"], where="post", color=col, linewidth=2)
    ax.text(24.2, d["kWh_received_Total"].iloc[-1], f"{lab} (house {hid_})", color=col, va="center", fontsize=9)
ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 3))
ax.set_xlabel("Hour of day (UTC; German summer time = UTC + 2)")
ax.set_ylabel("Grid import per 15 min (kWh)")
ax.set_title(f"Sunny day {best_day.date()}: the PV house imports exactly 0 at midday", loc="left", fontweight="bold")
save(fig, "03_sunny_day_pv_vs_nopv.png")

# %% STEP 8c-4 - Zero share by hour of day: PV vs no PV (known labels, all months)
hourly = counts.groupby(["Household_ID", "hour"])[["valid", "zero"]].sum()
hourly["share"] = hourly["zero"] / hourly["valid"] * 100
hourly = hourly["share"].unstack("hour")

fig, ax = plt.subplots(figsize=(9, 4))
ax.axvspan(9.5, 13.5, color=C_WINDOW, zorder=0)
for idx, col, lab in [(known_nopv, C_NOPV, "no PV"), (known_pv, C_PV, "PV")]:
    m = hourly.loc[hourly.index.intersection(idx)].median()
    ax.plot(m.index, m.values, color=col, linewidth=2, marker="o", markersize=4)
    if lab == "PV":
        ax.text(15.3, m.loc[15], "  PV houses", color=col, fontsize=10, fontweight="bold", va="bottom")
    else:
        ax.text(19, m.max() + 2, "houses without PV", color=col, fontsize=10, fontweight="bold", va="bottom")
ax.set_xticks(range(0, 24, 2)); ax.set_xlim(0, 24)
ax.set_xlabel("Hour of day (UTC)"); ax.set_ylabel("Median share of zero readings (%)")
ax.set_title("PV houses hit zero import around midday; houses without PV never do", loc="left", fontweight="bold")
ax.text(11.5, 0.03, "chosen window\n10-13 UTC", ha="center", va="bottom", color=C_TEXT, fontsize=8,
        transform=ax.get_xaxis_transform())
save(fig, "04_zero_share_by_hour.png")

# %% STEP 8c-5 - Distribution of pv_zero_share for known PV vs known no-PV, with the 4 % threshold
bins = np.arange(0, 102, 2)
fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
for ax, idx, col, lab in [(axes[0], known_nopv, C_NOPV, "Known no PV"), (axes[1], known_pv, C_PV, "Known PV")]:
    vals = report.set_index("Household_ID").loc[idx, "pv_zero_share"].dropna()
    ax.hist(vals, bins=bins, color=col, edgecolor="white", linewidth=1.5)
    ax.axvline(PV_THRESHOLD, color=C_TEXT, linestyle="--", linewidth=1.2)
    ax.text(0.99, 0.85, f"{lab}  (n = {len(vals)})", transform=ax.transAxes, ha="right", color=col, fontweight="bold")
    ax.set_ylabel("Households")
axes[0].text(PV_THRESHOLD + 1, axes[0].get_ylim()[1] * 0.6, f"threshold {PV_THRESHOLD:g} %", color=C_TEXT, fontsize=9)
axes[1].set_xlabel("pv_zero_share: zero readings at 10-13 UTC (%)")
axes[0].set_title("Two clearly separated groups: a 4 % threshold splits them", loc="left", fontweight="bold")
save(fig, "05_zero_share_distribution.png")

# %% STEP 8c-6 (optional) - Accuracy, precision and recall vs threshold
fig, ax = plt.subplots(figsize=(8, 4))
for col_name, colr, style in [("accuracy_%", C_TEXT, "-"), ("precision_%", "#1baf7a", "--"), ("recall_%", "#4a3aa7", ":")]:
    ax.plot(threshold_table["threshold_%"], threshold_table[col_name], color=colr, linestyle=style, linewidth=2, marker="o", markersize=4)
    ax.text(20.4, threshold_table[col_name].iloc[-1], col_name.replace("_%", ""), color=colr, va="center", fontsize=9)
ax.axvline(PV_THRESHOLD, color=C_PV, linewidth=1.2)
ax.set_xlabel("Threshold (%)"); ax.set_ylabel("Percent of known households")
ax.set_xlim(0, 22)
ax.set_title(f"Accuracy peaks at {PV_THRESHOLD:g} %", loc="left", fontweight="bold")
save(fig, "06_threshold_tradeoff.png")

# %% STEP 8c-7 (optional) - Labelled "no PV" but PV appears later: zero share per quarter
for case in [721074, 796869]:
    if case not in set(grid["Household_ID"]):
        continue
    d = grid[(grid["Household_ID"] == case) & grid["Timestamp"].dt.hour.isin(PV_HOURS)].dropna(subset=["kWh_received_Total"])
    q = d.groupby(d["Timestamp"].dt.tz_localize(None).dt.to_period("Q"))["kWh_received_Total"].apply(lambda x: (x < ZERO_KWH).mean() * 100)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(q.index.astype(str), q.values, color=C_PV, edgecolor="white", linewidth=1.5)
    ax.axhline(PV_THRESHOLD, color=C_TEXT, linestyle="--", linewidth=1)
    ax.set_ylabel("Zero readings at 10-13 UTC (%)")
    ax.set_title(f"House {case}: labelled 'no PV', but a PV pattern appears later", loc="left", fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    save(fig, f"07_pv_appears_later_{case}.png")


# %% [markdown]
# ## C. Finish: fix spikes, helper columns, save the shared file

# %% STEP 9a - Load the shared file if needed (only if you restarted VS Code / the kernel)
SHARED_PATH = OUT_DIR / "households_15min.parquet"
SUMMARY_PATH = OUT_DIR / "household_summary.csv"
if "grid" not in globals() or "pv_label" not in grid.columns:
    grid = pd.read_parquet(SHARED_PATH)
    print("Loaded shared file:", f"{len(grid):,} rows")
if "report" not in globals() or "pv_label" not in report.columns:
    report = pd.read_csv(SUMMARY_PATH, sep=";")
    print("Loaded summary:", len(report), "households")
households = pd.read_csv(META_DIR / "households.csv", sep=";")

# %% STEP 9b - Fix physically impossible spikes (> 10 kWh per 15 min = more than ~40 kW)
# A single isolated reading far above what a house connection can deliver = meter / transmission error.
# Fix: replace with the average of the reading before and after (same house).
# If a neighbour is missing (or also implausible), the value becomes NaN instead - no guessing.
# Every change is logged with the original value. Re-running this cell is safe (nothing left to fix).
import numpy as np

KWH_COLS = ["kWh_received_Total", "kWh_received_HeatPump", "kWh_received_Other"]
IMPLAUSIBLE_KWH = 10.0
hid_arr = grid["Household_ID"].to_numpy()
if "is_implausible" not in grid.columns:
    grid["is_implausible"] = False

log_rows = []
for col in KWH_COLS:
    values = grid[col].to_numpy()                      # float32 array, same order as the grid
    bad_pos = np.flatnonzero(values > IMPLAUSIBLE_KWH)
    for i in bad_pos:
        same_prev = i > 0 and hid_arr[i - 1] == hid_arr[i]
        same_next = i < len(values) - 1 and hid_arr[i + 1] == hid_arr[i]
        before = values[i - 1] if same_prev else np.nan
        after = values[i + 1] if same_next else np.nan
        ok = (not np.isnan(before) and not np.isnan(after)
              and before <= IMPLAUSIBLE_KWH and after <= IMPLAUSIBLE_KWH)
        new_value = (before + after) / 2 if ok else np.nan
        log_rows.append({
            "Household_ID": hid_arr[i], "Timestamp": grid["Timestamp"].iat[i], "column": col,
            "original_value": float(values[i]), "value_before": float(before), "value_after": float(after),
            "replaced_with": float(new_value),
            "method": "average of neighbours" if ok else "set to NaN (neighbour missing)",
            "reason": f"> {IMPLAUSIBLE_KWH:g} kWh per 15 min (physically impossible for a household)",
        })
        grid.iat[i, grid.columns.get_loc(col)] = new_value
        grid.iat[i, grid.columns.get_loc("is_implausible")] = True

implausible_log = pd.DataFrame(log_rows)
log_path = OUT_DIR / "step9_implausible_values.csv"
if len(implausible_log):
    implausible_log.to_csv(log_path, index=False, sep=";")
    print(f"Fixed {len(implausible_log)} implausible readings -> logged in {log_path}")
    print(implausible_log[["Household_ID", "Timestamp", "column", "original_value",
                           "value_before", "value_after", "replaced_with"]].to_string(index=False))
else:
    print("No (new) implausible readings found - nothing changed.")
print("Rows flagged is_implausible:", int(grid["is_implausible"].sum()))

# %% STEP 10 - Helper columns: after_visit and Weather_ID
# after_visit: 1 = after the heat-pump optimisation; 0 = before visit, during visit (visit day) or unknown (control)
grid["after_visit"] = grid["AffectsTimePoint"].astype("string").eq("after visit").astype("int8")
print("after_visit by AffectsTimePoint:")
print(pd.crosstab(grid["AffectsTimePoint"], grid["after_visit"]))

# Weather_ID: key for the merge with the weather data (Weather_ID + Timestamp)
wid_lookup = households.set_index("Household_ID")["Weather_ID"]
grid["Weather_ID"] = grid["Household_ID"].map(wid_lookup).astype("category")
print("\nRows without Weather_ID:", int(grid["Weather_ID"].isna().sum()))     # expect 0
print("Households per weather station:\n", grid.drop_duplicates("Household_ID")["Weather_ID"].value_counts())

# %% STEP 11 - Save the ONE shared household file + the per-household summary
# Update the summary with Weather_ID and the number of fixed spikes per house
report = report.drop(columns=["Weather_ID", "n_implausible_fixed"], errors="ignore")
report = report.merge(wid_lookup.rename("Weather_ID"), left_on="Household_ID", right_index=True, how="left")
fixed_per_house = grid.loc[grid["is_implausible"], "Household_ID"].value_counts()
report["n_implausible_fixed"] = report["Household_ID"].map(fixed_per_house).fillna(0).astype(int)

grid.to_parquet(SHARED_PATH, index=False)
report.to_csv(SUMMARY_PATH, index=False, sep=";")
print("Saved:", SHARED_PATH, f"({SHARED_PATH.stat().st_size / 1e6:.0f} MB)")
print("Saved:", SUMMARY_PATH, f"({len(report)} households)")

# %% STEP 12 - Final check: reload the saved file and verify it
final = pd.read_parquet(SHARED_PATH)
per_house = final.drop_duplicates("Household_ID")

checks = {
    "410 households":                         final["Household_ID"].nunique() == 410,
    "29,724,288 rows (full 15-min grid)":     len(final) == 29_724_288,
    "no duplicate house+timestamp":           not final.duplicated(["Household_ID", "Timestamp"]).any(),
    "PV labels: 259 no PV / 150 PV / 1 NA":   (int((per_house["pv_label"] == 0).sum()), int((per_house["pv_label"] == 1).sum()),
                                               int(per_house["pv_label"].isna().sum())) == (259, 150, 1),
    "no value above 10 kWh left":             not (final[KWH_COLS] > IMPLAUSIBLE_KWH).any().any(),
    "every row has a Weather_ID":             final["Weather_ID"].notna().all(),
    "after_visit only 0/1":                   set(final["after_visit"].unique()) <= {0, 1},
    "no negative kWh":                        not (final[KWH_COLS] < 0).any().any(),
}
for name, ok in checks.items():
    print(("PASS  " if ok else "FAIL  ") + name)
print("\nColumns:", list(final.columns))
print("Rows flagged is_implausible:", int(final["is_implausible"].sum()))
print("Total kWh missing (%):", round(final["kWh_received_Total"].isna().mean() * 100, 2))
del final, per_house
print("\nDone. Next: commit preprocessing/01_households.py and share", SHARED_PATH.name, "with the team.")


# %%
