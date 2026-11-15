"""
research/experiments/train_and_validate.py
Trains and evaluates candidate algorithms under strict 5-fold StratifiedGroupKFold
on Lag 24 Hours. Compares HistGradientBoosting, Random Forest, and Calibrated Logistic Regression.
Evaluates both the Full feature set (G) and the Anchor Compact Mobile set (K).
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    log_loss, recall_score, precision_score, balanced_accuracy_score,
    confusion_matrix
)
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "preprocessing"))
from pipeline import build_feature_sets, make_fold_safe_pipeline

DATA_DIR = ROOT / "dataset"
ALL_CSV_DIR = DATA_DIR / "completedatasetwithALLcsv"
VAL_DIR = ROOT / "research" / "validation"
VAL_DIR.mkdir(parents=True, exist_ok=True)

# Load 24lag dataset (the primary operational prospective horizon)
csv_24 = ALL_CSV_DIR / "features_24lag_v3.csv"
if not csv_24.exists():
    csv_24 = ROOT / "AI-HealthCare-" / "dataset" / "completedatasetwithALLcsv" / "features_24lag_v3.csv"

print(f"Loading 24-hour lag dataset from {csv_24}...")
# Use stride=6 for standard benchmark comparability
df = pd.read_csv(csv_24, low_memory=False)
df["y"] = df["lapse"].astype(str).str.strip().str.lower().isin(["yes", "1", "true"]).astype("int8")
df = df.sort_values(["subid", "dttm_label"]).reset_index(drop=True)

# Apply stride 6
STRIDE = 6
df_thinned = df[df.groupby("subid").cumcount() % STRIDE == 0].reset_index(drop=True)
print(f"Thinned dataset: {len(df_thinned):,} rows across {df_thinned['subid'].nunique()} participants. Positive rate: {df_thinned['y'].mean():.3%}")

feature_sets = build_feature_sets(df_thinned.columns.tolist())
y = df_thinned["y"].to_numpy()
groups = df_thinned["subid"].to_numpy()

# Models to evaluate
CANDIDATES = [
    {
        "name": "HGB_Full_G",
        "set_name": "G_full_features",
        "model_type": "hgb",
        "estimator": HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.06, max_leaf_nodes=12,
            min_samples_leaf=100, l2_regularization=1.0, random_state=42
        )
    },
    {
        "name": "HGB_Compact_K",
        "set_name": "K_compact_mobile",
        "model_type": "hgb",
        "estimator": HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.06, max_leaf_nodes=12,
            min_samples_leaf=100, l2_regularization=1.0, random_state=42
        )
    },
    {
        "name": "Logit_Calibrated_K",
        "set_name": "K_compact_mobile",
        "model_type": "logit",
        "estimator": LogisticRegression(C=0.1, max_iter=500, random_state=42)
    },
    {
        "name": "RandomForest_Compact_K",
        "set_name": "K_compact_mobile",
        "model_type": "hgb",
        "estimator": RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_leaf=50, n_jobs=-1, random_state=42
        )
    }
]

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# Compute within-person AUROC
def compute_within_person_auc(y_true, y_pred, grps, min_pos=3):
    aucs = []
    for pid in np.unique(grps):
        mask = (grps == pid)
        y_p = y_true[mask]
        p_p = y_pred[mask]
        if y_p.sum() >= min_pos and (len(y_p) - y_p.sum()) >= min_pos:
            aucs.append(roc_auc_score(y_p, p_p))
    return float(np.mean(aucs)) if aucs else 0.5

results = {}
oof_predictions = {"subid": groups, "y": y}

for cand in CANDIDATES:
    name = cand["name"]
    feats = feature_sets[cand["set_name"]]
    print(f"\nEvaluating candidate {name} ({len(feats)} features)...")
    
    oof_prob = np.zeros(len(df_thinned), dtype=np.float32)
    t0 = time.time()
    
    for fold, (tr, te) in enumerate(cv.split(df_thinned, y, groups)):
        pipeline = make_fold_safe_pipeline(feats, df_thinned, model_type=cand["model_type"], estimator=cand["estimator"])
        pipeline.fit(df_thinned.iloc[tr][feats], y[tr])
        oof_prob[te] = pipeline.predict_proba(df_thinned.iloc[te][feats])[:, 1]
    
    elapsed = time.time() - t0
    oof_predictions[name] = oof_prob
    
    # Calculate performance metrics
    auroc = float(roc_auc_score(y, oof_prob))
    auprc = float(average_precision_score(y, oof_prob))
    brier = float(brier_score_loss(y, oof_prob))
    wp_auc = compute_within_person_auc(y, oof_prob, groups)
    
    # Threshold at top 10% highest risk
    thresh_top10 = float(np.percentile(oof_prob, 90))
    y_pred_top10 = (oof_prob >= thresh_top10).astype(int)
    recall_top10 = float(recall_score(y, y_pred_top10))
    prec_top10 = float(precision_score(y, y_pred_top10))
    
    # Balanced accuracy at Youden's J statistic threshold
    # Evaluate across candidate thresholds
    threshs = np.linspace(0.02, 0.50, 49)
    best_j, best_thresh = -1, 0.08
    for t_cand in threshs:
        y_bin = (oof_prob >= t_cand).astype(int)
        cm = confusion_matrix(y, y_bin)
        sens = cm[1, 1] / (cm[1, 0] + cm[1, 1])
        spec = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        j = sens + spec - 1
        if j > best_j:
            best_j = j
            best_thresh = float(t_cand)
            
    y_pred_cal = (oof_prob >= best_thresh).astype(int)
    cm_opt = confusion_matrix(y, y_pred_cal)
    sens_opt = float(cm_opt[1, 1] / (cm_opt[1, 0] + cm_opt[1, 1]))
    spec_opt = float(cm_opt[0, 0] / (cm_opt[0, 0] + cm_opt[0, 1]))
    bal_acc = float(balanced_accuracy_score(y, y_pred_cal))
    
    results[name] = {
        "features": len(feats),
        "AUROC": auroc,
        "AUPRC": auprc,
        "within_person_AUROC": wp_auc,
        "Brier_score": brier,
        "Recall_at_top10pct": recall_top10,
        "Precision_at_top10pct": prec_top10,
        "Optimal_operating_threshold": best_thresh,
        "Sensitivity_at_optimal": sens_opt,
        "Specificity_at_optimal": spec_opt,
        "Balanced_accuracy": bal_acc,
        "training_time_seconds": elapsed
    }
    print(f"  AUROC: {auroc:.3f} | AUPRC: {auprc:.3f} | Within-Person: {wp_auc:.3f} | Brier: {brier:.4f}")
    print(f"  Recall @ Top 10%: {recall_top10:.3%} | Prec @ Top 10%: {prec_top10:.3%} | Bal Acc: {bal_acc:.3%}")

# Save validation summary
val_summary_path = VAL_DIR / "validation_results.json"
with open(val_summary_path, "w") as f:
    json.dump(results, f, indent=2)

# Save OOF predictions parquet
oof_df = pd.DataFrame(oof_predictions)
oof_df.to_parquet(VAL_DIR / "oof_predictions_24lag.parquet", index=False)

print(f"\nSaved cross-validation results to {val_summary_path}")
print(f"Saved OOF predictions to {VAL_DIR / 'oof_predictions_24lag.parquet'}")
