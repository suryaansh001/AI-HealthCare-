import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from pandas.api.types import is_object_dtype, is_string_dtype

ROOT = Path(__file__).parent
DATA = ROOT / "dataset" / "completedatasetwithALLcsv"
GRAPH_DIR = ROOT / "static" / "graphs"
FILES = sorted(DATA.glob("features_*lag_v3.csv"), key=lambda p: int(re.search(r"(\d+)lag", p.stem).group(1)))
FOLDS = 5
REPEATS = 2
STRIDE = 6
MAX_TEST_ROWS = 3000
SEED = 0

META = {"label_num", "subid", "dttm_label", "lapse", "y", "hours_since_start"}
PATTERN = re.compile(
    r"^(?:ema_(?P<item>\d+)\.)?p(?P<period>\d+)\.l(?P<lag>\d+)\."
    r"(?P<kind>[rd])(?P<stat>ratecount|recent|min|max|median)"
    r"(?:_response|\.count\.(?P<target>lapse|ema))$"
)


def family(column):
    if column in META:
        return "meta"
    if column.startswith("demo_") or column == "label_day":
        return "static"
    match = PATTERN.match(column)
    if not match:
        return "unparsed"
    data = match.groupdict()
    kind = "change" if data["kind"] == "d" else "raw"
    if data["stat"] == "ratecount":
        return "past_use" if data["target"] == "lapse" else "compliance"
    if data["stat"] == "recent":
        return f"current_{kind}"
    return f"aggregate_{kind}"


def run_file(path):
    lag = int(re.search(r"(\d+)lag", path.stem).group(1))
    df = pd.read_csv(path, low_memory=False)
    df["y"] = df["lapse"].astype(str).str.strip().str.lower().eq("yes").astype("int8")
    df = df.sort_values(["subid", "dttm_label"]).reset_index(drop=True)
    df = df[df.groupby("subid").cumcount() % STRIDE == 0].reset_index(drop=True)
    families = {column: family(column) for column in df.columns}
    static_raw = [column for column, value in families.items() if value == "static"]
    categorical = [
        column for column in static_raw
        if not pd.api.types.is_numeric_dtype(df[column])
    ]
    dummies = pd.get_dummies(df[categorical], dtype="float32")
    df = pd.concat([df.drop(columns=categorical), dummies], axis=1)
    static = [column for column in static_raw if column not in categorical] + list(dummies.columns)
    feature_columns = [
        column for column, value in families.items()
        if column not in META and value != "static"
    ]
    features = static + [column for column in feature_columns if column not in static]
    X = df[features].to_numpy(dtype="float32")
    y = df.y.to_numpy()
    groups = df.subid.to_numpy()
    rows = []
    for repeat in range(REPEATS):
        cv = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED + repeat)
        for fold, (train, test) in enumerate(cv.split(X, y, groups)):
            if len(test) > MAX_TEST_ROWS:
                rng = np.random.default_rng(SEED + repeat * 100 + fold)
                test = rng.choice(test, size=MAX_TEST_ROWS, replace=False)
            model = HistGradientBoostingClassifier(
                max_iter=150, learning_rate=0.06, max_leaf_nodes=12,
                min_samples_leaf=100, l2_regularization=1.0,
                random_state=SEED + repeat, early_stopping=False,
            )
            model.fit(X[train], y[train])
            baseline = roc_auc_score(y[test], model.predict_proba(X[test])[:, 1])
            importance = permutation_importance(
                model, X[test], y[test], scoring="roc_auc", n_repeats=1,
                random_state=SEED + repeat * 100 + fold, n_jobs=-1,
            )
            for name, value in zip(features, importance.importances_mean):
                rows.append({
                    "lag_hours": lag,
                    "feature": name,
                    "family": families.get(name, "static_encoded"),
                    "importance_auc_drop": value,
                    "baseline_test_auroc": baseline,
                    "repeat": repeat + 1,
                    "fold": fold + 1,
                })
            print(f"lag={lag} repeat={repeat + 1} fold={fold + 1} baseline={baseline:.3f}")
    return pd.DataFrame(rows)


def main():
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    importance = pd.concat([run_file(path) for path in FILES], ignore_index=True)
    summary = (
        importance.groupby(["lag_hours", "feature", "family"], as_index=False)
        .agg(mean_importance_auc_drop=("importance_auc_drop", "mean"),
             sd_importance_auc_drop=("importance_auc_drop", "std"))
        .fillna(0)
    )
    summary = summary.sort_values(["lag_hours", "mean_importance_auc_drop"], ascending=[True, False])
    summary.to_csv(GRAPH_DIR / "feature_importance_by_lag.csv", index=False)

    top_features = set()
    for lag in summary.lag_hours.unique():
        top_features.update(summary[summary.lag_hours.eq(lag)].nlargest(12, "mean_importance_auc_drop").feature)
    plot_data = summary[summary.feature.isin(top_features)]
    lags = sorted(summary.lag_hours.unique())
    fig, axes = plt.subplots(len(lags), 1, figsize=(14, 3.2 * len(lags)), squeeze=False)
    for axis, lag in zip(axes[:, 0], lags):
        part = plot_data[plot_data.lag_hours.eq(lag)].nlargest(12, "mean_importance_auc_drop").sort_values("mean_importance_auc_drop")
        labels = [f"{row.feature} [{row.family}]" for row in part.itertuples()]
        colors = ["#b23a48" if "change" in row.family else "#1768ac" for row in part.itertuples()]
        axis.barh(labels, part.mean_importance_auc_drop, color=colors)
        axis.axvline(0, color="#333333", linewidth=0.8)
        axis.set_title(f"{lag}-hour lag: top validation permutation importance")
        axis.set_xlabel("Decrease in validation AUROC after shuffling feature")
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("Most important features for lapse prediction", fontsize=16, y=0.995)
    fig.tight_layout()
    fig.savefig(GRAPH_DIR / "feature_importance_by_lag.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {GRAPH_DIR / 'feature_importance_by_lag.png'}")
    print(f"saved {GRAPH_DIR / 'feature_importance_by_lag.csv'}")


if __name__ == "__main__":
    main()
