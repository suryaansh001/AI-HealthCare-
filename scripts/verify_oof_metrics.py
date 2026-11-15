"""
Standalone OOF Parquet inspector and metric recomputer.
Read-only: does not modify any files.
"""
import json, sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss
)

OOF = r"research\validation\oof_predictions_24lag.parquet"

print("=== SECTION 1: LOAD AND SCHEMA ===")
try:
    df = pd.read_parquet(OOF)
except Exception as e:
    print(f"FAIL: Cannot load Parquet: {e}")
    sys.exit(1)

print(f"Rows: {len(df)}")
print(f"Columns ({len(df.columns)}): {list(df.columns)}")
print(f"Dtypes:\n{df.dtypes.to_string()}")

print("\n=== SECTION 2: LABEL DISTRIBUTION ===")
label_col = "y"
print(f"Label column: '{label_col}'")
print(f"Value counts:\n{df[label_col].value_counts().to_string()}")
print(f"Missing labels: {df[label_col].isna().sum()}")
n_pos = int((df[label_col] == 1).sum())
n_neg = int((df[label_col] == 0).sum())
prevalence = n_pos / len(df)
print(f"Positive: {n_pos}, Negative: {n_neg}, Prevalence: {prevalence:.6f}")

pred_cols = [c for c in df.columns if c not in ("subid", "y")]
print(f"\n=== SECTION 3: PREDICTION COLUMNS ===")
print(f"Prediction columns: {pred_cols}")

for c in pred_cols:
    print(f"\n--- {c} ---")
    print(f"  Missing: {df[c].isna().sum()}")
    desc = df[c].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print(f"  Stats:\n{desc.to_string()}")

print(f"\n=== SECTION 4: PARTICIPANTS ===")
n_participants = df["subid"].nunique()
print(f"Unique participants: {n_participants}")

print(f"\n=== SECTION 5: METRIC RECOMPUTATION ===")
y = df[label_col].to_numpy()
groups = df["subid"].to_numpy()

for c in pred_cols:
    p = df[c].to_numpy()
    auroc = roc_auc_score(y, p)
    auprc = average_precision_score(y, p)
    brier = brier_score_loss(y, p)
    ll = log_loss(y, np.clip(p, 1e-15, 1-1e-15))
    print(f"\n{c}:")
    print(f"  AUROC:  {auroc:.6f}")
    print(f"  AUPRC:  {auprc:.6f}")
    print(f"  Brier:  {brier:.6f}")
    print(f"  LogLoss:{ll:.6f}")

    # Within-person AUROC (>=1 pos, >=1 neg)
    aucs_1 = []
    for pid in np.unique(groups):
        mask = groups == pid
        yp = y[mask]
        pp = p[mask]
        if yp.sum() >= 1 and (len(yp) - yp.sum()) >= 1:
            try:
                aucs_1.append(roc_auc_score(yp, pp))
            except:
                pass
    wp1 = float(np.mean(aucs_1)) if aucs_1 else float("nan")
    print(f"  Within-person AUROC (>=1/1): {wp1:.6f} (n={len(aucs_1)})")

    # Within-person AUROC (>=3 pos, >=3 neg)
    aucs_3 = []
    for pid in np.unique(groups):
        mask = groups == pid
        yp = y[mask]
        pp = p[mask]
        if yp.sum() >= 3 and (len(yp) - yp.sum()) >= 3:
            try:
                aucs_3.append(roc_auc_score(yp, pp))
            except:
                pass
    wp3 = float(np.mean(aucs_3)) if aucs_3 else float("nan")
    print(f"  Within-person AUROC (>=3/3): {wp3:.6f} (n={len(aucs_3)})")

print("\n=== DONE ===")
