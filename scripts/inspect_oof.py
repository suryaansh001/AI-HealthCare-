import pandas as pd, json, sys
path = r"research/validation/oof_predictions_24lag.parquet"
try:
    df = pd.read_parquet(path)
except Exception as e:
    print('FAIL', e)
    sys.exit(1)
# schema
schema = {col: str(df[col].dtype) for col in df.columns}
row_count = len(df)
label_col = 'label' if 'label' in df.columns else None
label_dist = df[label_col].value_counts().to_dict() if label_col else {}
pred_cols = [c for c in df.columns if c.startswith('pred_')]
missing_counts = {c: df[c].isna().sum() for c in pred_cols}
stats = {}
for c in pred_cols:
    s = df[c].describe(percentiles=[0.25,0.5,0.75])
    stats[c] = {"min": s['min'], "25%": s['25%'], "50%": s['50%'], "75%": s['75%'], "max": s['max'], "mean": s['mean']}
participants = df['participant_id'].nunique() if 'participant_id' in df.columns else None
prevalence = label_dist.get(1,0)/row_count if row_count>0 else None
output = {
    "row_count": row_count,
    "schema": schema,
    "label_column": label_col,
    "label_distribution": label_dist,
    "prediction_columns": pred_cols,
    "missing_counts": missing_counts,
    "prediction_stats": stats,
    "participants": participants,
    "prevalence": prevalence
}
print(json.dumps(output, indent=2))
