"""
research/preprocessing/dataset_audit.py
Independent, reproducible dataset audit for Anchor ML Research.
Calculates exact row counts, columns, participant distributions, positive rates,
missingness profiles, duplicates, and constant features across feature datasets and raw EMA.
"""
import sys
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "dataset"
ALL_CSV_DIR = DATA_DIR / "completedatasetwithALLcsv"
REPORTS_DIR = ROOT / "research" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Executing dataset audit from ROOT: {ROOT}")

# 1. Audit features_0lag_v3 (using fast Parquet copy)
parquet_path = DATA_DIR / "features_0lag_v3.parquet"
if not parquet_path.exists():
    parquet_path = ROOT / "AI-HealthCare-" / "dataset" / "features_0lag_v3.parquet"

print(f"Reading {parquet_path}...")
df0 = pd.read_parquet(parquet_path)

n_rows = len(df0)
n_cols = len(df0.columns)
n_participants = df0["subid"].nunique()
lapse_series = df0["lapse"].astype(str).str.strip().str.lower()
y = lapse_series.isin(["yes", "1", "true"]).astype(int)
pos_count = int(y.sum())
neg_count = n_rows - pos_count
prevalence = pos_count / n_rows

print(f"0-lag: rows={n_rows}, cols={n_cols}, participants={n_participants}, positives={pos_count} ({prevalence:.4%})")

# Participant-level distributions
part_g = df0.groupby("subid")
part_rows = part_g.size()
part_pos = part_g.apply(lambda g: (g["lapse"].astype(str).str.strip().str.lower().isin(["yes", "1", "true"])).sum())
part_rate = part_pos / part_rows

n_lapsing_parts = int((part_pos > 0).sum())
n_zero_lapsing_parts = int((part_pos == 0).sum())

# Date range
df0["dttm_label"] = pd.to_datetime(df0["dttm_label"])
min_date = str(df0["dttm_label"].min())
max_date = str(df0["dttm_label"].max())

# Consecutive spacing
df0 = df0.sort_values(["subid", "dttm_label"]).reset_index(drop=True)
spacing_hours = df0.groupby("subid")["dttm_label"].diff().dt.total_seconds() / 3600.0
pct_1h = float((spacing_hours == 1.0).sum() / spacing_hours.notna().sum())
max_gap = float(spacing_hours.max())

# Missingness
missing_cells = int(df0.isna().sum().sum())
total_cells = n_rows * n_cols
pct_missing = missing_cells / total_cells
cols_with_missing = int((df0.isna().sum() > 0).sum())

# Duplicates
dup_rows = int(df0.duplicated(subset=["subid", "dttm_label"]).sum())

# Constant columns
numeric_cols = df0.select_dtypes(include=[np.number]).columns
constant_cols = [c for c in numeric_cols if df0[c].nunique(dropna=True) <= 1]

# 2. Audit raw EMA file
ema_path = ALL_CSV_DIR / "ema.csv"
if not ema_path.exists():
    ema_path = ROOT / "AI-HealthCare-" / "dataset" / "completedatasetwithALLcsv" / "ema.csv"

print(f"Reading {ema_path}...")
df_ema = pd.read_csv(ema_path, low_memory=False)
ema_rows = len(df_ema)
ema_cols = len(df_ema.columns)
ema_participants = df_ema["subid"].nunique()
ema_drink_reports = int((df_ema["ema_1"].astype(str).str.strip().str.lower() == "yes").sum())
ema_part_drinks = df_ema[df_ema["ema_1"].astype(str).str.strip().str.lower() == "yes"]["subid"].nunique()

results = {
    "feature_0lag": {
        "rows": n_rows,
        "columns": n_cols,
        "participants": n_participants,
        "positive_windows": pos_count,
        "negative_windows": neg_count,
        "prevalence": prevalence,
        "lapsing_participants": n_lapsing_parts,
        "zero_lapse_participants": n_zero_lapsing_parts,
        "min_date": min_date,
        "max_date": max_date,
        "pct_1h_spacing": pct_1h,
        "max_gap_hours": max_gap,
        "missing_cells": missing_cells,
        "pct_missing_cells": pct_missing,
        "cols_with_missing": cols_with_missing,
        "duplicate_timestamps": dup_rows,
        "constant_columns": constant_cols,
        "participant_rows_min": int(part_rows.min()),
        "participant_rows_median": float(part_rows.median()),
        "participant_rows_mean": float(part_rows.mean()),
        "participant_rows_max": int(part_rows.max()),
        "participant_pos_median": float(part_pos[part_pos > 0].median()),
        "participant_pos_mean": float(part_pos[part_pos > 0].mean()),
        "participant_pos_max": int(part_pos.max()),
        "heaviest_participant_share": float(part_pos.max() / pos_count)
    },
    "raw_ema": {
        "rows": ema_rows,
        "columns": ema_cols,
        "participants": ema_participants,
        "drinking_reports": ema_drink_reports,
        "participants_reporting_drinking": ema_part_drinks,
        "drinking_rate": ema_drink_reports / ema_rows
    }
}

out_json = REPORTS_DIR / "dataset_audit_metrics.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved audit metrics to {out_json}")
print("Phase 1 calculations completed successfully.")
