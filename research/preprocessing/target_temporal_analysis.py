"""
research/preprocessing/target_temporal_analysis.py
Investigates target construction, window overlap, autocorrelation,
and evaluates the 4 distinct prediction tasks:
Task A: 24h prospective prediction
Task B: Lapse continuation vs onset
Task C: New lapse after abstinence (Quiet periods: 7-day, 14-day)
Task D: Long-horizon degradation (72h, 168h, 336h)
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "dataset"
REPORTS_DIR = ROOT / "research" / "reports"

parquet_path = DATA_DIR / "features_0lag_v3.parquet"
if not parquet_path.exists():
    parquet_path = ROOT / "AI-HealthCare-" / "dataset" / "features_0lag_v3.parquet"

print(f"Loading {parquet_path} for temporal analysis...")
df = pd.read_parquet(parquet_path)
df["y"] = df["lapse"].astype(str).str.strip().str.lower().isin(["yes", "1", "true"]).astype("int8")
df["dttm_label"] = pd.to_datetime(df["dttm_label"])
df = df.sort_values(["subid", "dttm_label"]).reset_index(drop=True)

# 1. Target Overlap & Autocorrelation
# Find consecutive positive runs (episodes)
df["y_prev"] = df.groupby("subid")["y"].shift(1).fillna(0).astype("int8")
df["new_pos_run"] = ((df["y"] == 1) & (df["y_prev"] == 0)).astype("int8")
total_pos_runs = int(df["new_pos_run"].sum())
total_pos_windows = int(df["y"].sum())
mean_run_length = total_pos_windows / max(1, total_pos_runs)

print(f"Total positive windows: {total_pos_windows}")
print(f"Distinct continuous positive runs (clusters): {total_pos_runs}")
print(f"Mean continuous positive run length: {mean_run_length:.1f} hours")

# Autocorrelation of target
autocorr_1h = float(df.groupby("subid")["y"].apply(lambda s: s.autocorr(lag=1)).mean())
autocorr_6h = float(df.groupby("subid")["y"].apply(lambda s: s.autocorr(lag=6)).mean())
autocorr_24h = float(df.groupby("subid")["y"].apply(lambda s: s.autocorr(lag=24)).mean())
autocorr_168h = float(df.groupby("subid")["y"].apply(lambda s: s.autocorr(lag=168)).mean())

print(f"Target Autocorrelation: 1h={autocorr_1h:.3f}, 6h={autocorr_6h:.3f}, 24h={autocorr_24h:.3f}, 168h={autocorr_168h:.3f}")

# 2. Analyze Past-Use Feature Association (Continuation vs Onset)
# Check: when p12.l0.rratecount.count.lapse > 0, what is the lapse rate?
p12_col = "p12.l0.rratecount.count.lapse"
p24_col = "p24.l0.rratecount.count.lapse"
p168_col = "p168.l0.rratecount.count.lapse"

prob_y_given_p12_drank = float(df[df[p12_col] > 0]["y"].mean())
prob_y_given_p12_clean = float(df[df[p12_col] <= 0]["y"].mean())

prob_y_given_p24_drank = float(df[df[p24_col] > 0]["y"].mean())
prob_y_given_p24_clean = float(df[df[p24_col] <= 0]["y"].mean())

prob_y_given_p168_drank = float(df[df[p168_col] > 0]["y"].mean())
prob_y_given_p168_clean = float(df[df[p168_col] <= 0]["y"].mean())

print(f"P(y=1 | drank in past 12h): {prob_y_given_p12_drank:.3%} vs clean: {prob_y_given_p12_clean:.3%}")
print(f"P(y=1 | drank in past 24h): {prob_y_given_p24_drank:.3%} vs clean: {prob_y_given_p24_clean:.3%}")
print(f"P(y=1 | drank in past 168h): {prob_y_given_p168_drank:.3%} vs clean: {prob_y_given_p168_clean:.3%}")

# 3. Clean Abstinence Subsets (Quiet Periods)
# Quiet 168h: No lapse in past 7 days (p168 <= 0)
df_quiet168 = df[df[p168_col] <= 0]
q168_rows = len(df_quiet168)
q168_pos = int(df_quiet168["y"].sum())
q168_prev = q168_pos / q168_rows
q168_parts = df_quiet168["subid"].nunique()

results = {
    "target_overlap": {
        "total_positive_windows": total_pos_windows,
        "distinct_positive_runs": total_pos_runs,
        "mean_run_length_hours": mean_run_length,
        "autocorr_1h": autocorr_1h,
        "autocorr_6h": autocorr_6h,
        "autocorr_24h": autocorr_24h,
        "autocorr_168h": autocorr_168h
    },
    "continuation_vs_onset": {
        "prob_y_given_p12_drank": prob_y_given_p12_drank,
        "prob_y_given_p12_clean": prob_y_given_p12_clean,
        "prob_y_given_p24_drank": prob_y_given_p24_drank,
        "prob_y_given_p24_clean": prob_y_given_p24_clean,
        "prob_y_given_p168_drank": prob_y_given_p168_drank,
        "prob_y_given_p168_clean": prob_y_given_p168_clean
    },
    "quiet168_subset": {
        "rows": q168_rows,
        "positive_windows": q168_pos,
        "prevalence": q168_prev,
        "participants": q168_parts
    }
}

out_json = REPORTS_DIR / "target_temporal_metrics.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved temporal metrics to {out_json}")
