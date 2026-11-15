"""Run fixed-v2 benchmark, personalization feature, and baseline experiments."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.anchor_v2.data import build_feature_sets, load_feature_matrix
from research.anchor_v2.pipeline import make_model_pipeline
from research.anchor_v2.validation import (
    grouped_partitions,
    metrics_at_probabilities,
    participant_auc_summary,
    participant_bootstrap_intervals,
    select_youden_threshold,
)
from research.anchor_v3 import VERSION
from research.anchor_v3.diagnostics import (
    history_strata_metrics,
    paired_participant_bootstrap,
    participant_metrics,
    prediction_variance_diagnostics,
    participant_token,
)
from research.anchor_v3.features import add_personal_history_features
from research.anchor_v3.visualize import save_diagnostic_figures


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CSV = ROOT / "dataset" / "completedatasetwithALLcsv" / "features_24lag_v3.csv"
V1_SCHEMA = ROOT / "research" / "models" / "feature_schema.json"
V2_SCHEMA = ROOT / "research" / "models" / "feature_schema_v2.json"
V2_METRICS = ROOT / "research" / "validation" / "model_comparison_v2.json"
V2_MODEL = ROOT / "research" / "models" / "anchor_lapse_model_v2.joblib"
RESULTS_DIR = ROOT / "research" / "anchor_v3" / "results"
FIGURES_DIR = ROOT / "research" / "anchor_v3" / "figures"
OUTPUT_FILES = [
    RESULTS_DIR / "comparison_metrics_v3.json",
    RESULTS_DIR / "development_oof_v3.parquet",
    RESULTS_DIR / "participant_metrics_v3.csv",
    RESULTS_DIR / "history_strata_v3.csv",
    RESULTS_DIR / "feature_importance_v3.csv",
    RESULTS_DIR / "personalization_diagnostics_v3.json",
]
FIGURE_FILES = [
    FIGURES_DIR / "participant_auc_v3.png",
    FIGURES_DIR / "participant_mean_risk_v3.png",
    FIGURES_DIR / "history_amount_metrics_v3.png",
    FIGURES_DIR / "v2_feature_importance_v3.png",
    FIGURES_DIR / "calibration_diagnostics_v3.png",
]


def _to_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _log_loss(target: np.ndarray, probability: np.ndarray) -> float:
    return float(log_loss(target, np.clip(probability, 1e-15, 1 - 1e-15), labels=[0, 1]))


def _cv_metrics(
    frame: pd.DataFrame,
    target: np.ndarray,
    groups: np.ndarray,
    features: list[str],
    model_name: str,
    parameters: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = np.full(len(frame), np.nan, dtype=float)
    for train_index, validation_index in splitter.split(frame, target, groups):
        if set(groups[train_index]).intersection(groups[validation_index]):
            raise RuntimeError("Participant leakage detected in development folds")
        model = make_model_pipeline(
            frame,
            features,
            model_name,
            parameters,
            seed=seed,
        )
        model.fit(frame.iloc[train_index][features], target[train_index])
        scores[validation_index] = model.predict_proba(
            frame.iloc[validation_index][features]
        )[:, 1]
    if not np.isfinite(scores).all():
        raise RuntimeError("Development OOF predictions are incomplete")
    result = metrics_at_probabilities(target, scores)
    result["within_person"] = participant_auc_summary(target, scores, groups)
    result["feature_count"] = len(features)
    return result, scores


def _rank_candidates(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(
        [
            {
                "name": name,
                "auroc": metric["auroc"],
                "auprc": metric["auprc"],
                "brier": metric["brier"],
                "log_loss": metric["log_loss"],
                "feature_count": metric["feature_count"],
            }
            for name, metric in results.items()
        ]
    )
    frame["auroc_rank"] = frame["auroc"].rank(ascending=False, method="average")
    frame["auprc_rank"] = frame["auprc"].rank(ascending=False, method="average")
    frame["brier_rank"] = frame["brier"].rank(ascending=True, method="average")
    frame["log_loss_rank"] = frame["log_loss"].rank(ascending=True, method="average")
    frame["mean_rank"] = frame[
        ["auroc_rank", "auprc_rank", "brier_rank", "log_loss_rank"]
    ].mean(axis=1)
    return frame.sort_values(
        ["mean_rank", "feature_count", "name"],
        kind="stable",
    ).to_dict(orient="records")


def _make_model(
    frame: pd.DataFrame,
    target: np.ndarray,
    features: list[str],
    estimator: str,
    parameters: dict[str, Any],
    seed: int,
) -> Any:
    model = make_model_pipeline(frame, features, estimator, parameters, seed=seed)
    return model.fit(frame[features], target)


def _fit_candidate_predictions(
    candidate_name: str,
    train: pd.DataFrame,
    target: np.ndarray,
    threshold_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_map: dict[str, list[str]],
    definitions: dict[str, tuple[str, str, dict[str, Any]]],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    feature_set, estimator, parameters = definitions[candidate_name]
    features = feature_map[feature_set]
    model = _make_model(train, target, features, estimator, parameters, seed)
    threshold_probability = model.predict_proba(threshold_frame[features])[:, 1]
    test_probability = model.predict_proba(test_frame[features])[:, 1]
    return threshold_probability, test_probability


def _history_presence_probabilities(
    train: pd.DataFrame,
    target: np.ndarray,
    threshold_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    column: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    present_train = pd.to_numeric(train[column], errors="coerce").fillna(0).gt(0).to_numpy()
    global_prevalence = float(target.mean())
    estimates: dict[str, float] = {}
    for present in (False, True):
        mask = present_train == present
        positives = int(target[mask].sum())
        count = int(mask.sum())
        estimates["present" if present else "absent"] = (
            (positives + 0.5) / (count + 1.0)
            if count
            else global_prevalence
        )
    def apply(frame: pd.DataFrame) -> np.ndarray:
        present = pd.to_numeric(frame[column], errors="coerce").fillna(0).gt(0).to_numpy()
        return np.where(present, estimates["present"], estimates["absent"]).astype(float)
    return apply(threshold_frame), apply(test_frame), estimates


def _feature_importance(model: Any, features: list[str]) -> tuple[pd.DataFrame, dict[str, float]]:
    names = [str(name) for name in model.named_steps["preprocessor"].get_feature_names_out()]
    values = getattr(model.named_steps["model"], "feature_importances_", None)
    if values is None:
        return pd.DataFrame(columns=["feature", "importance"]), {}
    table = pd.DataFrame({"feature": names, "importance": values}).sort_values(
        "importance", ascending=False
    )
    table["family"] = np.where(
        table["feature"].str.startswith("p"),
        "historical_count",
        np.where(
            table["feature"].str.startswith("ema_"),
            "EMA",
            np.where(
                table["feature"].str.startswith("demo_"),
                "demographic",
                "derived_personal_or_calendar",
            ),
        ),
    )
    family_sum = table.groupby("family")["importance"].sum().to_dict()
    table["source_features"] = len(features)
    return table, {name: float(value) for name, value in family_sum.items()}


def _participant_mean_table(
    groups: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"participant": groups, "y": target, "probability": prediction}
    )
    table = frame.groupby("participant", sort=False).agg(
        mean_probability=("probability", "mean"),
        prevalence=("y", "mean"),
        n=("y", "size"),
    ).reset_index()
    table["participant_token"] = table["participant"].map(participant_token)
    return table.drop(columns="participant").sort_values("participant_token").reset_index(drop=True)


def run(seed: int = 42, bootstrap_replicates: int = 1000) -> dict[str, Any]:
    missing_inputs = [
        path
        for path in (SOURCE_CSV, V1_SCHEMA, V2_SCHEMA, V2_METRICS, V2_MODEL)
        if not path.exists()
    ]
    if missing_inputs:
        raise FileNotFoundError(f"Required fixed benchmark inputs are missing: {missing_inputs}")
    existing = [path for path in OUTPUT_FILES + FIGURE_FILES if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite v3 results: {existing}")

    frame, input_schema = load_feature_matrix(SOURCE_CSV, V1_SCHEMA, stride=6)
    frame, derived_families = add_personal_history_features(frame)
    target = frame["y"].to_numpy(dtype=np.int8)
    group_ids = frame["subid"].astype(str).to_numpy()
    partition = grouped_partitions(
        target,
        group_ids,
        n_splits=7,
        development_folds=3,
        seed=42,
    )
    development = frame.iloc[partition["development"]].reset_index(drop=True)
    threshold = frame.iloc[partition["threshold_selection"]].reset_index(drop=True)
    test = frame.iloc[partition["locked_test"]].reset_index(drop=True)
    dev_y = development["y"].to_numpy(dtype=np.int8)
    test_y = test["y"].to_numpy(dtype=np.int8)
    test_groups = test["subid"].astype(str).to_numpy()
    threshold_y = threshold["y"].to_numpy(dtype=np.int8)

    compact = input_schema["features"]
    feature_sets = build_feature_sets(frame.columns.tolist(), compact)
    feature_map: dict[str, list[str]] = {
        "compact_22": list(compact),
        "historical_lapse_only": feature_sets["historical_lapse_only"],
        "ema_only": feature_sets["ema_only"],
        "compact_plus_personal_ema": list(compact) + derived_families["personal_ema_baseline"],
        "compact_plus_history_rates": list(compact) + derived_families["history_rate_contrast"],
        "full_personalized_hybrid": list(compact) + derived_families["personalized_hybrid"],
    }
    definitions: dict[str, tuple[str, str, dict[str, Any]]] = {
        "pooled_rf_compact": (
            "compact_22",
            "random_forest",
            {"max_depth": 8, "n_estimators": 40},
        ),
        "logistic_compact": ("compact_22", "logistic", {"C": 0.1}),
        "logistic_history_only": ("historical_lapse_only", "logistic", {"C": 0.1}),
        "random_forest_ema_only": (
            "ema_only",
            "random_forest",
            {"max_depth": 8, "n_estimators": 40},
        ),
        "personalized_ema_hybrid_rf": (
            "compact_plus_personal_ema",
            "random_forest",
            {"max_depth": 8, "n_estimators": 40},
        ),
        "history_rate_augmented_rf": (
            "compact_plus_history_rates",
            "random_forest",
            {"max_depth": 8, "n_estimators": 40},
        ),
        "full_personalized_rf": (
            "full_personalized_hybrid",
            "random_forest",
            {"max_depth": 8, "n_estimators": 40},
        ),
    }

    development_results: dict[str, dict[str, Any]] = {}
    development_oof: dict[str, np.ndarray] = {}
    for index, (candidate, (feature_set, estimator, parameters)) in enumerate(
        definitions.items(), start=1
    ):
        print(f"[{index}/{len(definitions)}] development grouped OOF: {candidate}", flush=True)
        result, score = _cv_metrics(
            development,
            dev_y,
            development["subid"].astype(str).to_numpy(),
            feature_map[feature_set],
            estimator,
            parameters,
            seed,
        )
        development_results[candidate] = result
        development_oof[candidate] = score

    ranking = _rank_candidates(development_results)
    selected_candidate = ranking[0]["name"]

    v2_model = joblib.load(V2_MODEL)
    v2_schema = json.loads(V2_SCHEMA.read_text(encoding="utf-8"))
    if v2_schema["features"] != compact:
        raise RuntimeError("The fixed v2 schema differs from the original compact feature order")
    v2_probability = v2_model["model"].predict_proba(test[compact])[:, 1]
    v2_partition_metrics = json.loads(V2_METRICS.read_text(encoding="utf-8"))
    stored_locked = v2_partition_metrics["threshold_analysis"]["locked_test_metrics"]
    reproduced = metrics_at_probabilities(test_y, v2_probability)
    for metric in ("auroc", "auprc", "brier", "log_loss"):
        if not np.isclose(reproduced[metric], stored_locked[metric], atol=1e-10):
            raise RuntimeError(f"Fixed v2 benchmark {metric} did not reproduce")
    v2_threshold = float(v2_partition_metrics["threshold_analysis"]["threshold"])

    prediction_map: dict[str, np.ndarray] = {"v2_random_forest": v2_probability}
    threshold_map: dict[str, float | None] = {"v2_random_forest": v2_threshold}
    selection_counts: dict[str, Any] = {}
    dev_prevalence = float(dev_y.mean())
    prediction_map["constant_prevalence"] = np.full(len(test), dev_prevalence, dtype=float)
    threshold_map["constant_prevalence"] = 0.5

    history_column = "p24.l24.rratecount.count.lapse"
    history_t, history_test, history_rates = _history_presence_probabilities(
        development,
        dev_y,
        threshold,
        test,
        history_column,
    )
    history_threshold = select_youden_threshold(threshold_y, history_t)
    prediction_map["history_presence_prevalence"] = history_test
    threshold_map["history_presence_prevalence"] = history_threshold["threshold"]
    selection_counts["history_presence_prevalence"] = history_rates

    # This is intentionally a label-persistence diagnostic, not a model or prospective baseline.
    previous_label = frame.groupby("subid", sort=False)["y"].shift(1)
    previous_lookup = pd.Series(
        previous_label.to_numpy(),
        index=pd.MultiIndex.from_arrays(
            [frame["subid"].astype(str).to_numpy(), frame["dttm_label"].to_numpy()]
        ),
    )
    def previous_scores(partition_frame: pd.DataFrame) -> np.ndarray:
        keys = pd.MultiIndex.from_arrays(
            [
                partition_frame["subid"].astype(str).to_numpy(),
                partition_frame["dttm_label"].to_numpy(),
            ]
        )
        values = previous_lookup.reindex(keys).to_numpy(dtype=float)
        return np.where(np.isfinite(values), values, dev_prevalence)
    prediction_map["previous_stored_label_diagnostic"] = previous_scores(test)
    threshold_map["previous_stored_label_diagnostic"] = 0.5

    fitted_models: dict[str, Any] = {}
    for candidate, (feature_set, estimator, parameters) in definitions.items():
        threshold_probability, test_probability = _fit_candidate_predictions(
            candidate,
            development,
            dev_y,
            threshold,
            test,
            feature_map,
            definitions,
            seed,
        )
        operating_point = select_youden_threshold(threshold_y, threshold_probability)
        prediction_map[candidate] = test_probability
        threshold_map[candidate] = operating_point["threshold"]
        selection_counts[candidate] = {
            "threshold": operating_point,
            "feature_count": len(feature_map[feature_set]),
        }
        fitted_models[candidate] = _make_model(
            development,
            dev_y,
            feature_map[feature_set],
            estimator,
            parameters,
            seed,
        )

    results: dict[str, Any] = {}
    for name, probability in prediction_map.items():
        operating_threshold = threshold_map[name]
        result = metrics_at_probabilities(
            test_y,
            probability,
            operating_threshold,
        )
        result["within_person"] = participant_auc_summary(test_y, probability, test_groups)
        result["participant_bootstrap_95ci"] = participant_bootstrap_intervals(
            test_y,
            probability,
            test_groups,
            replicates=bootstrap_replicates,
            seed=seed,
        )
        result["threshold_selection_source"] = {
            "v2_random_forest": "fixed v2 threshold",
            "constant_prevalence": "fixed 0.5",
            "previous_stored_label_diagnostic": "fixed 0.5; diagnostic only",
            "history_presence_prevalence": "separate v2 threshold-selection participants",
        }.get(name, "separate v2 threshold-selection participants")
        if name == "previous_stored_label_diagnostic":
            result["interpretation"] = (
                "Autocorrelation diagnostic only; target-label availability and window overlap are unverified."
            )
        results[name] = result

    candidate_probabilities = {
        name: scores
        for name, scores in prediction_map.items()
        if name not in {
            "constant_prevalence",
            "history_presence_prevalence",
            "previous_stored_label_diagnostic",
            "v2_random_forest",
        }
    }
    paired_delta = paired_participant_bootstrap(
        test_y,
        test_groups,
        v2_probability,
        candidate_probabilities,
        replicates=bootstrap_replicates,
        seed=seed,
    )

    thresholds_for_people = threshold_map
    history_rows = test["v3_prior_sample_count"].to_numpy(dtype=np.int64)
    long_history = pd.to_numeric(
        test["p168.l24.rratecount.count.lapse"],
        errors="coerce",
    ).to_numpy(dtype=float)
    participant_table = participant_metrics(
        test_y,
        test_groups,
        prediction_map,
        history_rows,
        long_history,
        thresholds_for_people,
    )
    history_table = history_strata_metrics(
        test_y,
        {
            "v2_random_forest": v2_probability,
            "personalized_ema_hybrid_rf": prediction_map["personalized_ema_hybrid_rf"],
            "full_personalized_rf": prediction_map["full_personalized_rf"],
        },
        test_groups,
        history_rows,
    )

    v2_feature_table, v2_importance_families = _feature_importance(
        v2_model["model"].estimator.estimator
        if hasattr(v2_model["model"], "estimator")
        and hasattr(v2_model["model"].estimator, "estimator")
        else v2_model["model"],
        compact,
    )
    hybrid_importance_table, hybrid_importance_families = _feature_importance(
        fitted_models["full_personalized_rf"],
        feature_map["full_personalized_hybrid"],
    )
    v2_feature_table["model"] = "v2_random_forest"
    hybrid_importance_table["model"] = "full_personalized_rf"
    importance_table = pd.concat(
        [v2_feature_table, hybrid_importance_table],
        ignore_index=True,
    )
    mean_table = _participant_mean_table(test_groups, test_y, v2_probability)
    variance_diagnostics = prediction_variance_diagnostics(
        test_y,
        test_groups,
        v2_probability,
    )

    participants_eligible = int(
        participant_table["within_person_auc_eligible"].sum()
    )
    participant_training_coverage = {
        "locked_test_participants": int(test["subid"].nunique()),
        "participants_with_training_labels_in_development": 0,
        "eligible_for_separate_person_models_under_fixed_unseen_participant_split": 0,
        "reason": (
            "The grouped split holds each locked-test participant entirely out of training. "
            "Fitting an individual estimator from locked-test labels would change the fixed benchmark."
        ),
        "chronological_personalized_modeling": "not run; target interval and follow-up are unverified",
    }

    history_summary = (
        history_table.groupby(["history_band", "model"], sort=False)
        .apply(
            lambda rows: {
                "rows": int(rows["rows"].iloc[0]),
                "participants": int(rows["participants"].iloc[0]),
                "prevalence": float(rows["prevalence"].iloc[0]),
                "auroc": rows["auroc"].iloc[0],
                "auprc": rows["auprc"].iloc[0],
                "brier": rows["brier"].iloc[0],
            },
            include_groups=False,
        )
        .to_dict()
    )
    metrics_document = {
        "version": VERSION,
        "target_status": "NOT_VERIFIED",
        "test_set_policy": (
            "Reused the exact fixed v2 locked-test participants and rows. No v3 test rows were substituted."
        ),
        "data": {
            "source": str(SOURCE_CSV.relative_to(ROOT)),
            "source_rows_before_stride": 270911,
            "rows_after_stride": int(len(frame)),
            "participants": int(frame["subid"].nunique()),
            "stride": 6,
            "locked_test_rows": int(len(test)),
            "locked_test_participants": int(test["subid"].nunique()),
            "locked_test_positives": int(test_y.sum()),
            "locked_test_prevalence": float(test_y.mean()),
        },
        "fixed_v2_benchmark": {
            "stored_metrics": stored_locked,
            "independently_reproduced_metrics": reproduced,
            "same_partitions": True,
            "same_v2_artifact_and_feature_order": True,
        },
        "development_candidate_selection": {
            "method": (
                "Mean rank of 5-fold participant-grouped development OOF AUROC, AUPRC, "
                "Brier, and log loss; smaller feature count breaks ties."
            ),
            "selected_experimental_candidate": selected_candidate,
            "does_not_include_fixed_v2_artifact_in_dev_ranking": True,
            "candidates": [
                {
                    "name": row["name"],
                    "mean_rank": float(row["mean_rank"]),
                    "feature_count": int(row["feature_count"]),
                    **development_results[row["name"]],
                }
                for row in ranking
            ],
        },
        "locked_test_results": results,
        "paired_participant_bootstrap_delta_vs_v2": paired_delta,
        "threshold_policy": {
            "candidate_thresholds": selection_counts,
            "independent_of_locked_test": True,
            "decision_threshold_is_not_recommended_for_use": True,
            "reason": "Alert consequences and target semantics are unverified.",
        },
        "within_person_diagnostics": {
            "v2_eligible_participants": participants_eligible,
            "v2_total_test_participants": int(test["subid"].nunique()),
            "minimum_each_class": 3,
            "prediction_variance": variance_diagnostics,
            "performance_by_prior_sample_count": history_summary,
            "prior_sample_count_note": (
                "Counts prior stride-six rows, not a verified number of hours; feature timing is unresolved."
            ),
        },
        "historical_feature_diagnostics": {
            "v2_random_forest_importance_by_family": v2_importance_families,
            "v2_top_features": v2_feature_table.head(15)[
                ["feature", "importance"]
            ].to_dict(orient="records"),
            "hybrid_random_forest_importance_by_family": hybrid_importance_families,
            "v2_historical_counter_importance_share": float(
                v2_feature_table.loc[
                    v2_feature_table["family"] == "historical_count", "importance"
                ].sum()
            ),
            "meaning_limit": (
                "Impurity importance is associational and cannot establish causal influence or target leakage."
            ),
        },
        "personalization": {
            "engineered_feature_families": {
                name: columns for name, columns in derived_families.items()
            },
            "construction": (
                "Per-person rolling summaries use only the preceding 24 sampled rows, exclude current row, "
                "and require at least six prior nonmissing readings for the mean/std."
            ),
            "individual_models": participant_training_coverage,
            "identifier_policy": "participant IDs excluded from saved outputs; diagnostic CSV uses SHA-256 tokens",
        },
        "chronological_future_prediction": {
            "status": "BLOCKED_PENDING_TARGET_AND_FEATURE_TIMING_VERIFICATION",
            "why": (
                "No verified prediction origin, outcome start/end interval, endpoint rule, or complete-follow-up field."
            ),
            "personal_baseline_features": (
                "Past-row-only by computation order, but source measurement timestamps and target overlap remain unverified."
            ),
        },
        "selection_decision": {
            "v3_model_artifact_created": False,
            "reason": (
                "Personalization must be demonstrated on a valid known-participant chronological evaluation; "
                "that evaluation is blocked. No locked-test winner is promoted."
            ),
            "fixed_benchmark_retained": "anchor_lapse_model_v2.joblib",
        },
        "reproducibility": {
            "seed": int(seed),
            "bootstrap_replicates": int(bootstrap_replicates),
            "v3_random_forest_estimators": 40,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    diagnostics_document = {
        "participant_level_rows": int(len(participant_table)),
        "eligible_within_person_auc": participants_eligible,
        "participant_mean_risk_vs_prevalence": variance_diagnostics,
        "errors": {
            "false_alert_definition": "predicted positive at a threshold selected outside locked test",
            "missed_positive_definition": "stored-label positive below that threshold",
        },
        "history_strata": _to_json(history_summary),
        "caution": (
            "Participant-level statistics have small denominators and are descriptive only. "
            "Target and feature timing remain unverified."
        ),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if any(path.exists() for path in OUTPUT_FILES):
        raise FileExistsError("A v3 output appeared during execution; refusing to overwrite it")
    oof = pd.DataFrame(
        {
            "participant_token": development["subid"].astype(str).map(participant_token),
            "y": dev_y,
            **{
                f"{name}_probability": score.astype(np.float32)
                for name, score in development_oof.items()
            },
        }
    )
    (RESULTS_DIR / "comparison_metrics_v3.json").write_text(
        json.dumps(_to_json(metrics_document), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    oof.to_parquet(RESULTS_DIR / "development_oof_v3.parquet", index=False)
    participant_table.to_csv(RESULTS_DIR / "participant_metrics_v3.csv", index=False)
    history_table.to_csv(RESULTS_DIR / "history_strata_v3.csv", index=False)
    importance_table.to_csv(RESULTS_DIR / "feature_importance_v3.csv", index=False)
    (RESULTS_DIR / "personalization_diagnostics_v3.json").write_text(
        json.dumps(_to_json(diagnostics_document), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    participant_mean_table = _participant_mean_table(test_groups, test_y, v2_probability)
    generated_figures = save_diagnostic_figures(
        FIGURES_DIR,
        participant_table,
        history_table,
        v2_feature_table,
        participant_mean_table,
        test_y,
        prediction_map,
    )
    print("Selected experimental candidate by development OOF:", selected_candidate)
    print("Fixed v2 metrics reproduced:", reproduced)
    print("Fixed v2 test participant summary:", results["v2_random_forest"]["within_person"])
    print("Personalized EMA hybrid test metrics:", results["personalized_ema_hybrid_rf"])
    print("Paired bootstrap deltas:", paired_delta)
    print("Figures:", generated_figures)
    print("Outputs:", [str(path) for path in OUTPUT_FILES])
    return _to_json(metrics_document)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    arguments = parser.parse_args()
    run(seed=arguments.seed, bootstrap_replicates=arguments.bootstrap_replicates)


if __name__ == "__main__":
    main()
