import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from pandas.api.types import is_object_dtype, is_string_dtype


BASE = Path(__file__).parent / "dataset" / "completedatasetwithALLcsv"
FILES = sorted(BASE.glob("features_*lag_v3.csv"))
STRIDE = 6
FOLDS = 5
REPEATS = 2
BOOT = 300
SEED = 0

META = {"label_num", "subid", "dttm_label", "lapse", "y"}
PAT = re.compile(
    r"^(?:ema_(?P<item>\d+)\.)?p(?P<period>\d+)\.l(?P<lag>\d+)\."
    r"(?P<kind>[rd])(?P<stat>ratecount|recent|min|max|median)"
    r"(?:_response|\.count\.(?P<target>lapse|ema))$"
)


def family(column):
    if column in META:
        return "meta"
    if column.startswith("demo_") or column == "label_day":
        return "static"
    match = PAT.match(column)
    if not match:
        return "unparsed"
    data = match.groupdict()
    kind = "change" if data["kind"] == "d" else "raw"
    if data["stat"] == "ratecount":
        return "past_use" if data["target"] == "lapse" else "compliance"
    if data["stat"] == "recent":
        return f"current_{kind}"
    return f"aggregate_{kind}"


def pct(value):
    return f"{100 * value:.2f}%"


def bootstrap_summary(predictions, y, groups, within, contrasts):
    rng = np.random.default_rng(SEED)
    positions = {pid: np.flatnonzero(groups == pid) for pid in np.unique(groups)}
    people = np.array(list(positions))
    draws = {name: {metric: [] for metric in ("AUROC", "AUPRC", "within")} for name in predictions}
    contrast_draws = {name: {metric: [] for metric in ("AUROC", "AUPRC", "within")} for name in contrasts}
    for _ in range(BOOT):
        chosen = rng.choice(people, size=len(people), replace=True)
        indices = np.concatenate([positions[pid] for pid in chosen])
        for name, pred in predictions.items():
            draws[name]["AUROC"].append(roc_auc_score(y[indices], pred[indices]))
            draws[name]["AUPRC"].append(average_precision_score(y[indices], pred[indices]))
            values = [within[name][pid] for pid in chosen if pid in within[name]]
            draws[name]["within"].append(np.mean(values) if values else np.nan)
        for name, (left, right) in contrasts.items():
            for metric in ("AUROC", "AUPRC", "within"):
                contrast_draws[name][metric].append(draws[left][metric][-1] - draws[right][metric][-1])
    intervals = {
        name: {metric: np.nanpercentile(draws[name][metric], [2.5, 97.5]) for metric in ("AUROC", "AUPRC", "within")}
        for name in predictions
    }
    contrast_intervals = {
        name: {metric: np.nanpercentile(contrast_draws[name][metric], [2.5, 97.5]) for metric in ("AUROC", "AUPRC", "within")}
        for name in contrasts
    }
    return intervals, contrast_intervals


def analyze(path):
    df = pd.read_csv(path, low_memory=False, parse_dates=["dttm_label"])
    df["y"] = df["lapse"].astype(str).str.strip().str.lower().eq("yes").astype("int8")
    df = df.sort_values(["subid", "dttm_label"]).reset_index(drop=True)
    original_rows = len(df)
    original_people = df.subid.nunique()
    original_positive = int(df.y.sum())
    original_prevalence = df.y.mean()
    gaps = df.groupby("subid").dttm_label.diff().dt.total_seconds().div(3600)
    per = df.groupby("subid").agg(rows=("y", "size"), positives=("y", "sum"))
    families = {column: family(column) for column in df.columns}
    unparsed = [column for column, value in families.items() if value == "unparsed"]
    if unparsed:
        raise ValueError(f"Unparsed columns in {path.name}: {unparsed[:5]}")
    early_hours = (df.dttm_label - df.groupby("subid").dttm_label.transform("min")).dt.total_seconds() / 3600
    feature_columns = [column for column, value in families.items() if value != "meta"]
    missing = df[feature_columns].isna().mean()
    family_series = pd.Series({column: families[column] for column in feature_columns})
    missing_by_family = (100 * missing.groupby(family_series).mean()).sort_index()

    df = df[df.groupby("subid").cumcount() % STRIDE == 0].reset_index(drop=True)
    static_raw = [column for column, value in families.items() if value == "static"]
    categorical = [
        column for column in static_raw
        if is_object_dtype(df[column]) or is_string_dtype(df[column])
    ]
    dummies = pd.get_dummies(df[categorical], dtype="float32")
    df = pd.concat([df.drop(columns=categorical), dummies], axis=1)
    static = [column for column in static_raw if column not in categorical] + list(dummies.columns)

    def columns(*names):
        return [column for column, value in families.items() if value in names and column in df.columns]

    past_use = columns("past_use")
    base = static + past_use
    sets = {
        "A static": static,
        "B past use": past_use,
        "C static + past use": base,
        "D + current raw": base + columns("current_raw"),
        "E + raw state": base + columns("current_raw", "aggregate_raw"),
        "F change only": base + columns("current_change", "aggregate_change"),
        "G full + change": base + columns("current_raw", "aggregate_raw", "current_change", "aggregate_change"),
        "H all + compliance": static + past_use + columns("compliance", "current_raw", "aggregate_raw", "current_change", "aggregate_change"),
    }
    y = df.y.to_numpy()
    groups = df.subid.to_numpy()
    predictions = {name: np.zeros(len(df), dtype=float) for name in sets}
    feature_arrays = {name: df[features].to_numpy(dtype="float32") for name, features in sets.items()}
    for repeat in range(REPEATS):
        cv = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED + repeat)
        for train, test in cv.split(df, y, groups):
            for name, features in sets.items():
                model = HistGradientBoostingClassifier(
                    max_iter=150, learning_rate=0.06, max_leaf_nodes=12,
                    min_samples_leaf=100, l2_regularization=1.0,
                    random_state=SEED + repeat,
                )
                model.fit(feature_arrays[name][train], y[train])
                predictions[name][test] += model.predict_proba(feature_arrays[name][test])[:, 1] / REPEATS

    by_person = {pid: np.flatnonzero(groups == pid) for pid in np.unique(groups)}
    within = {name: {} for name in sets}
    for name, pred in predictions.items():
        for pid, indices in by_person.items():
            labels = y[indices]
            if labels.sum() >= 3 and (1 - labels).sum() >= 3:
                within[name][pid] = roc_auc_score(labels, pred[indices])
    contrasts = {
        "D minus C": ("D + current raw", "C static + past use"),
        "E minus D": ("E + raw state", "D + current raw"),
        "G minus E": ("G full + change", "E + raw state"),
        "F minus E": ("F change only", "E + raw state"),
    }
    intervals, contrast_intervals = bootstrap_summary(predictions, y, groups, within, contrasts)
    summary = {}
    for name, pred in predictions.items():
        summary[name] = {
            "features": len(sets[name]),
            "auroc": roc_auc_score(y, pred),
            "auprc": average_precision_score(y, pred),
            "within": np.mean(list(within[name].values())),
            "brier": brier_score_loss(y, np.clip(pred, 0, 1)),
            "n_within": len(within[name]),
        }

    lag_match = re.search(r"features_(\d+)lag", path.stem)
    lag = lag_match.group(1) if lag_match else "unknown"
    report = [
        f"# Lag {lag} hours: exploratory lapse-forecasting results\n",
        "## Analysis scope\n",
        "This report analyzes one engineered Curtin-lab feature file. The outcome is a goal-inconsistent drinking window, not clinical relapse. Rows were sampled at every sixth hourly row, participants were kept together in five-fold stratified group cross-validation, and uncertainty intervals were bootstrapped over participants. These are exploratory results.\n",
        "## 1. Basic EDA\n",
        "| Quantity | Result |",
        "|---|---:|",
        f"| Labelled windows before thinning | {original_rows:,} |",
        f"| Participants | {original_people:,} |",
        f"| Positive windows | {original_positive:,} |",
        f"| Positive-window rate | {pct(original_prevalence)} |",
        f"| Date range | {df.dttm_label.min()} to {df.dttm_label.max()} |",
        f"| Rows used in modelling after stride 6 | {len(df):,} |",
        f"| Participants with at least one positive window | {(per.positives > 0).sum()} |",
        f"| Participants with no positive window | {(per.positives == 0).sum()} |",
        f"| Median rows per participant before thinning | {per.rows.median():.0f} |",
        f"| Positive windows per participant: median / mean / maximum | {per.positives.median():.0f} / {per.positives.mean():.1f} / {per.positives.max():.0f} |",
        f"| Consecutive gaps exactly 1 hour | {pct(gaps.eq(1).mean())} |",
        f"| Largest consecutive gap | {gaps.max():.0f} hours |",
        "\nFeature missingness was summarized by category. The category labels come from the column names; no item meanings were inferred from correlations.\n",
        "| Feature category | Number of columns | Mean missingness |",
        "|---|---:|---:|",
    ]
    for category in ["static", "past_use", "compliance", "current_raw", "current_change", "aggregate_raw", "aggregate_change"]:
        count = sum(value == category for value in families.values())
        report.append(f"| {category} | {count} | {missing_by_family.get(category, 0):.2f}% |")
    report.extend([
        f"\nRows in each participant's first 24, 72, and 168 hours were {pct((early_hours < 24).mean())}, {pct((early_hours < 72).mean())}, and {pct((early_hours < 168).mean())}, respectively.\n",
        "## 2. Category-wise predictive results\n",
        "AUPRC should be compared with the positive rate above, while AUROC should be compared with 0.50. The within-person metric averages AUROC only for participants with at least three positive and three negative rows after thinning.\n",
        "| Feature set | Features | AUROC | AUPRC | Within-person AUROC | Eligible people | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for name, values in summary.items():
        report.append(f"| {name} | {values['features']} | {values['auroc']:.3f} | {values['auprc']:.3f} | {values['within']:.3f} | {values['n_within']} | {values['brier']:.3f} |")
    report.extend([
        "\n### Paired differences with 95% participant-bootstrap intervals\n",
        "| Contrast | AUROC difference | AUPRC difference | Within-person difference |",
        "|---|---:|---:|---:|",
    ])
    for name, (left, right) in contrasts.items():
        values = []
        for metric, key in [("AUROC", "auroc"), ("AUPRC", "auprc"), ("within", "within")]:
            point = summary[left][key] - summary[right][key]
            interval = contrast_intervals[name][metric]
            values.append(f"{point:+.3f} [{interval[0]:+.3f}, {interval[1]:+.3f}]")
        report.append(f"| {name} | {values[0]} | {values[1]} | {values[2]} |")
    d_c = summary["D + current raw"]["auroc"] - summary["C static + past use"]["auroc"]
    g_e = summary["G full + change"]["auroc"] - summary["E + raw state"]["auroc"]
    report.extend([
        "\n## 3. Hypotheses and plain-language results\n",
        "### H1: Past drinking history is useful for prediction\n",
        f"**Result: supported, with an important caution.** Past-use-only AUROC was {summary['B past use']['auroc']:.3f}, compared with 0.50 for chance. However, the source data contain overlapping outcome windows, so past-use performance may partly reflect persistence from a recent lapse rather than genuinely early warning.\n",
        "**Situation example:** If someone has recently reported drinking, many nearby hourly windows are also labelled positive. A model can score those windows correctly using recent drinking history, even if it is not predicting a completely new episode far in advance.\n",
        "### H2: Current EMA state improves prediction beyond static traits and past use\n",
        f"**Result: see the D-minus-C interval above.** Adding current raw EMA variables changed AUROC by {d_c:+.3f}.\n",
        "**Situation example:** A person with the same past drinking history may have a different current craving or stress response. The model uses that current response to assign different risk scores.\n",
        "### H3: Personal-baseline change variables add information beyond raw state\n",
        f"**Result: see the G-minus-E interval above.** The estimated AUROC change was {g_e:+.3f}. If its interval contains zero, this file does not show clear added discrimination. This is not proof that change variables are useless; it means the added value is uncertain in this sample.\n",
        "**Situation example:** Two people may both report stress=8, but one person normally reports stress=8 and the other normally reports stress=2. A change feature tries to distinguish those situations.\n",
        "### H4: Longer lags reduce forecasting performance\n",
        "**Result: this report estimates one lag only.** Compare the same row of each lag-specific report rather than comparing this file alone. Differences across lag files are exploratory and may also reflect different row counts and positive rates.\n",
        "**Situation example:** A signal recorded now may help predict drinking tomorrow but be less useful for predicting drinking two weeks later.\n",
        "## 4. Honest interpretation\n",
        f"The model's pooled AUROC was {summary['G full + change']['auroc']:.3f}, while the average within-person AUROC was {summary['G full + change']['within']:.3f} among {summary['G full + change']['n_within']} eligible participants. These are different questions: pooled AUROC can benefit from stable differences between people, while within-person AUROC asks whether high-risk hours are ranked above safer hours for the same person. Results do not establish causality, clinical relapse prediction, or generalization beyond this sample.\n",
        "## 5. Main limitations\n",
        "- Positive hourly windows overlap, so they are not independent drinking episodes.\n- Participant grouping prevents person-level leakage, but does not remove label persistence.\n- The analysis uses one general model; it does not test personalized random-effects or partial-pooling models.\n- Feature names identify raw, change, aggregate, current, and lag categories, but do not by themselves prove the exact time endpoint or absence of all leakage.\n",
    ])
    output = path.with_name(f"{path.stem}_results.md")
    output.write_text("\n".join(report), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    for input_file in FILES:
        analyze(input_file)