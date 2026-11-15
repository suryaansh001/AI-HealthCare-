"""Train, compare, calibrate, and serialize a research-only ANCHOR v2 model."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.anchor_v2 import MODEL_VERSION
from research.anchor_v2.data import (
    build_feature_sets,
    feature_type_map,
    load_feature_matrix,
)
from research.anchor_v2.pipeline import candidate_specs, make_model_pipeline
from research.anchor_v2.validation import (
    grouped_partitions,
    metrics_at_probabilities,
    participant_auc_summary,
    participant_bootstrap_intervals,
    select_youden_threshold,
)


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "dataset" / "completedatasetwithALLcsv" / "features_24lag_v3.csv"
SCHEMA_PATH = ROOT / "research" / "models" / "feature_schema.json"
OUTPUTS = {
    "model": ROOT / "research" / "models" / "anchor_lapse_model_v2.joblib",
    "schema": ROOT / "research" / "models" / "feature_schema_v2.json",
    "card": ROOT / "research" / "models" / "model_card_v2.json",
    "metrics": ROOT / "research" / "validation" / "model_comparison_v2.json",
    "oof": ROOT / "research" / "validation" / "oof_predictions_v2.parquet",
}


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _rank_candidates(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table = pd.DataFrame(results)
    table["rank_auroc"] = table["auroc"].rank(ascending=False, method="average")
    table["rank_auprc"] = table["auprc"].rank(ascending=False, method="average")
    table["rank_brier"] = table["brier"].rank(ascending=True, method="average")
    table["rank_log_loss"] = table["log_loss"].rank(ascending=True, method="average")
    table["mean_metric_rank"] = table[
        ["rank_auroc", "rank_auprc", "rank_brier", "rank_log_loss"]
    ].mean(axis=1)
    table = table.sort_values(
        ["mean_metric_rank", "feature_count", "name"],
        ascending=[True, True, True],
        kind="stable",
    )
    return table.to_dict(orient="records")


def _score_candidate(
    frame: pd.DataFrame,
    target: np.ndarray,
    groups: np.ndarray,
    candidate: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    features = candidate["features"]
    probabilities = np.full(len(frame), np.nan, dtype=float)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    started = time.monotonic()
    for train_index, validation_index in splitter.split(frame, target, groups):
        if set(groups[train_index]).intersection(groups[validation_index]):
            raise RuntimeError("Participant appeared in both sides of an internal validation fold")
        model = make_model_pipeline(
            frame,
            features,
            candidate["model_name"],
            candidate["parameters"],
            seed=seed,
        )
        model.fit(frame.iloc[train_index][features], target[train_index])
        probabilities[validation_index] = model.predict_proba(
            frame.iloc[validation_index][features]
        )[:, 1]
    if not np.isfinite(probabilities).all():
        raise RuntimeError(f"Candidate {candidate['name']} did not produce complete OOF scores")

    metrics = metrics_at_probabilities(target, probabilities)
    if metrics["auroc"] is None or metrics["auprc"] is None:
        raise ValueError("Development folds must contain both target classes")
    metrics.update(
        {
            "name": candidate["name"],
            "model_name": candidate["model_name"],
            "feature_set": candidate["feature_set"],
            "feature_count": len(features),
            "within_person": participant_auc_summary(target, probabilities, groups),
            "elapsed_seconds": float(time.monotonic() - started),
        }
    )
    return metrics, probabilities


def _fit_calibrators(
    model: Any,
    frame: pd.DataFrame,
    target: np.ndarray,
    features: list[str],
    participant_ids: np.ndarray,
) -> tuple[dict[str, Any], dict[str, str]]:
    fitted: dict[str, Any] = {"uncalibrated": model}
    statuses = {"uncalibrated": "available"}
    if np.unique(target).size != 2:
        return fitted, {**statuses, "sigmoid": "not fitted: calibration data require both classes"}

    fitted["sigmoid"] = CalibratedClassifierCV(
        FrozenEstimator(model),
        method="sigmoid",
    ).fit(frame[features], target)
    statuses["sigmoid"] = "fit on participant-disjoint calibration-fit partition"

    positives = int(target.sum())
    negatives = int(len(target) - positives)
    unique_participants = len(np.unique(participant_ids))
    if positives >= 100 and negatives >= 100 and unique_participants >= 20:
        fitted["isotonic"] = CalibratedClassifierCV(
            FrozenEstimator(model),
            method="isotonic",
        ).fit(frame[features], target)
        statuses["isotonic"] = (
            "fit; calibration-fit partition met minimum 100 observations per class "
            "and 20 participants"
        )
    else:
        statuses["isotonic"] = (
            "not fitted: calibration-fit partition did not meet minimum sample criteria"
        )
    return fitted, statuses


def _choose_calibrator(
    models: dict[str, Any],
    frame: pd.DataFrame,
    target: np.ndarray,
    features: list[str],
) -> tuple[str, dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    metrics: dict[str, dict[str, Any]] = {}
    probabilities: dict[str, np.ndarray] = {}
    for name, model in models.items():
        predicted = model.predict_proba(frame[features])[:, 1]
        probabilities[name] = predicted
        metrics[name] = metrics_at_probabilities(target, predicted)

    ranked = pd.DataFrame(
        [
            {
                "name": name,
                "brier": values["brier"],
                "log_loss": values["log_loss"],
            }
            for name, values in metrics.items()
        ]
    )
    ranked["brier_rank"] = ranked["brier"].rank(ascending=True, method="average")
    ranked["log_loss_rank"] = ranked["log_loss"].rank(ascending=True, method="average")
    ranked["mean_rank"] = ranked[["brier_rank", "log_loss_rank"]].mean(axis=1)
    ranked["preference"] = ranked["name"].map(
        {"uncalibrated": 0, "sigmoid": 1, "isotonic": 2}
    )
    selected = ranked.sort_values(
        ["mean_rank", "preference"], kind="stable"
    ).iloc[0]["name"]
    for name, values in metrics.items():
        values["mean_calibration_selection_rank"] = float(
            ranked.loc[ranked["name"] == name, "mean_rank"].iloc[0]
        )
    return str(selected), metrics, probabilities


def _global_importance(
    model: Any,
    feature_names: list[str],
    frame: pd.DataFrame,
    target: np.ndarray,
) -> tuple[str, list[dict[str, Any]]]:
    estimator = model.named_steps["model"]
    preprocessor = model.named_steps["preprocessor"]
    transformed_names = [str(name) for name in preprocessor.get_feature_names_out()]
    coefficients = getattr(estimator, "coef_", None)
    if coefficients is not None:
        weights = coefficients[0]
        order = np.argsort(np.abs(weights))[::-1][:20]
        return (
            "transformed logistic-regression coefficients",
            [
                {
                    "term": transformed_names[index],
                    "coefficient": float(weights[index]),
                    "magnitude": float(abs(weights[index])),
                    "interpretation": "associational model weight, not a causal effect",
                }
                for index in order
            ],
        )

    importances = getattr(estimator, "feature_importances_", None)
    if importances is not None:
        order = np.argsort(importances)[::-1][:20]
        return (
            "tree impurity importance",
            [
                {
                    "term": transformed_names[index],
                    "importance": float(importances[index]),
                    "interpretation": "global model importance, not a causal effect",
                }
                for index in order
            ],
        )

    from sklearn.inspection import permutation_importance

    sample_count = min(len(frame), 2000)
    sample = frame.iloc[:sample_count][feature_names]
    sample_target = target[:sample_count]
    importance = permutation_importance(
        model,
        sample,
        sample_target,
        scoring="average_precision",
        n_repeats=2,
        random_state=42,
        n_jobs=-1,
    )
    order = np.argsort(importance.importances_mean)[::-1][:20]
    return (
        "permutation importance on calibration-selection data",
        [
            {
                "term": feature_names[index],
                "importance_mean": float(importance.importances_mean[index]),
                "importance_std": float(importance.importances_std[index]),
                "interpretation": "predictive association, not a causal effect",
            }
            for index in order
        ],
    )


def train(seed: int = 42, stride: int = 6, bootstrap_replicates: int = 500) -> dict[str, Any]:
    if sklearn.__version__ < "1.6":
        raise RuntimeError("ANCHOR v2 requires scikit-learn >= 1.6 for frozen-estimator calibration")
    existing = [str(path) for path in OUTPUTS.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite versioned outputs: " + ", ".join(existing)
        )

    frame, input_schema = load_feature_matrix(DATA_PATH, SCHEMA_PATH, stride=stride)
    sets = build_feature_sets(frame.columns.tolist(), input_schema["features"])
    target = frame["y"].to_numpy(dtype=np.int8)
    groups = frame["subid"].to_numpy()
    partitions = grouped_partitions(target, groups, n_splits=7, development_folds=3, seed=seed)
    dev = frame.iloc[partitions["development"]].reset_index(drop=True)
    dev_y = dev["y"].to_numpy(dtype=np.int8)
    dev_groups = dev["subid"].to_numpy()
    partition_frames = {
        name: frame.iloc[indices].reset_index(drop=True)
        for name, indices in partitions.items()
        if name != "development"
    }
    if not all(
        np.unique(partition["y"]).size == 2
        for partition in partition_frames.values()
    ):
        raise ValueError("Each calibration, threshold, and locked-test partition must contain both classes")

    candidates = []
    for spec in candidate_specs(sets):
        candidates.append({**spec, "features": sets[spec["feature_set"]]})

    cv_results: list[dict[str, Any]] = []
    oof_columns: dict[str, Any] = {
        "participant_id": dev_groups,
        "y": dev_y,
    }
    for index, candidate in enumerate(candidates, start=1):
        print(
            f"[{index}/{len(candidates)}] grouped CV: {candidate['name']} "
            f"({len(candidate['features'])} inputs)",
            flush=True,
        )
        metrics, probabilities = _score_candidate(
            dev,
            dev_y,
            dev_groups,
            candidate,
            seed,
        )
        cv_results.append(metrics)
        oof_columns[candidate["name"]] = probabilities.astype(np.float32)

    ranked_candidates = _rank_candidates(cv_results)
    selected_result = ranked_candidates[0]
    selected_candidate = next(
        candidate for candidate in candidates if candidate["name"] == selected_result["name"]
    )
    selected_features = selected_candidate["features"]
    selected_model = make_model_pipeline(
        dev,
        selected_features,
        selected_candidate["model_name"],
        selected_candidate["parameters"],
        seed=seed,
    ).fit(dev[selected_features], dev_y)

    calibration_fit = partition_frames["calibration_fit"]
    calibration_selection = partition_frames["calibration_selection"]
    threshold_selection = partition_frames["threshold_selection"]
    locked_test = partition_frames["locked_test"]
    fitted_calibrators, calibration_status = _fit_calibrators(
        selected_model,
        calibration_fit,
        calibration_fit["y"].to_numpy(dtype=np.int8),
        selected_features,
        calibration_fit["subid"].to_numpy(),
    )
    selected_calibrator, calibration_selection_metrics, _ = _choose_calibrator(
        fitted_calibrators,
        calibration_selection,
        calibration_selection["y"].to_numpy(dtype=np.int8),
        selected_features,
    )
    selected_threshold = select_youden_threshold(
        threshold_selection["y"].to_numpy(dtype=np.int8),
        fitted_calibrators[selected_calibrator].predict_proba(
            threshold_selection[selected_features]
        )[:, 1],
    )

    test_y = locked_test["y"].to_numpy(dtype=np.int8)
    test_groups = locked_test["subid"].to_numpy()
    test_calibration_metrics: dict[str, dict[str, Any]] = {}
    test_probabilities: dict[str, np.ndarray] = {}
    for name, calibrated_model in fitted_calibrators.items():
        predicted = calibrated_model.predict_proba(locked_test[selected_features])[:, 1]
        test_probabilities[name] = predicted
        test_calibration_metrics[name] = metrics_at_probabilities(test_y, predicted)
    selected_test_probabilities = test_probabilities[selected_calibrator]
    threshold = selected_threshold["threshold"]
    selected_test_metrics = metrics_at_probabilities(
        test_y,
        selected_test_probabilities,
        threshold if threshold is not None else None,
    )
    selected_test_metrics["participant_auc_summary"] = participant_auc_summary(
        test_y,
        selected_test_probabilities,
        test_groups,
    )
    selected_test_metrics["participant_bootstrap_95pct"] = participant_bootstrap_intervals(
        test_y,
        selected_test_probabilities,
        test_groups,
        replicates=bootstrap_replicates,
        seed=seed,
    )

    explanation_method, global_importance = _global_importance(
        selected_model,
        selected_features,
        calibration_selection[selected_features],
        calibration_selection["y"].to_numpy(dtype=np.int8),
    )
    feature_schema = {
        "model_version": MODEL_VERSION,
        "feature_count": len(selected_features),
        "features": list(selected_features),
        "feature_types": feature_type_map(dev, selected_features),
        "feature_semantics_status": "available input names/types; upstream feature timing and meaning not fully verified",
    }
    group_counts = {
        name: {
            "rows": int(len(frame.iloc[indices])),
            "participants": int(frame.iloc[indices]["subid"].nunique()),
            "positives": int(frame.iloc[indices]["y"].sum()),
        }
        for name, indices in partitions.items()
    }
    candidate_summaries = []
    for result in ranked_candidates:
        candidate_summaries.append(
            {
                key: value
                for key, value in result.items()
                if key != "within_person"
            }
            | {"within_person": result["within_person"]}
        )

    comparison = {
        "model_version": MODEL_VERSION,
        "experiment_id": "anchor-v2-grouped-evaluation-seed-42",
        "target_status": "NOT_VERIFIED",
        "target_column": "lapse",
        "target_note": "All metrics are conditional on supplied labels; they do not establish future-event semantics.",
        "data": {
            "source": str(DATA_PATH.relative_to(ROOT)),
            "rows_after_stride": int(len(frame)),
            "participants": int(frame["subid"].nunique()),
            "positive_rows": int(target.sum()),
            "positive_rate": float(target.mean()),
            "stride": int(stride),
            "missing_compact_feature_cells": int(
                frame[input_schema["features"]].isna().sum().sum()
            ),
            "duplicate_participant_timestamps": 0,
        },
        "partition_counts": group_counts,
        "candidate_selection": {
            "method": "mean rank across development-only grouped OOF AUROC, AUPRC, Brier, and log loss; smaller feature count breaks ties",
            "selected": selected_candidate["name"],
            "selected_model": selected_candidate["model_name"],
            "selected_feature_set": selected_candidate["feature_set"],
            "selected_parameters": selected_candidate["parameters"],
            "candidates": candidate_summaries,
        },
        "calibration": {
            "fit_status": calibration_status,
            "selection_partition_metrics": calibration_selection_metrics,
            "selected_method": selected_calibrator,
            "independent_locked_test_metrics": test_calibration_metrics,
        },
        "threshold_analysis": {
            **selected_threshold,
            "selection_partition": "threshold_selection",
            "decision_use_approved": False,
            "reason": "Target meaning and alert consequences are not verified; inference returns probability only.",
            "locked_test_metrics": selected_test_metrics,
        },
        "participant_reporting": {
            "test_participant_auc": selected_test_metrics["participant_auc_summary"],
            "bootstrap_unit": "participant",
            "subgroup_metrics": "deferred: demographic field meanings/codebook are unavailable",
        },
        "chronological_validation": {
            "status": "PENDING_TARGET_VERIFICATION",
            "utility_implemented": "make_purged_chronological_split",
            "guard": "requires explicit verified prediction, outcome-start, outcome-end, and follow-up columns",
        },
        "features": {
            "comparison_sets": {name: len(values) for name, values in sets.items()},
            "selected_schema": feature_schema,
            "new_derived_features": [],
            "reason": "No additional temporal features were created because event definitions and source feature timing are unresolved.",
        },
        "reproducibility": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "git_head": _git_head(),
            "seed": int(seed),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        },
    }

    # Keep participant-level reporting de-identified in persisted results.
    person_metrics = []
    for participant in np.unique(test_groups):
        mask = test_groups == participant
        participant_y = test_y[mask]
        if participant_y.sum() >= 3 and (len(participant_y) - participant_y.sum()) >= 3:
            person_metrics.append(
                {
                    "participant_token": hashlib.sha256(
                        f"{participant}:{seed}".encode("utf-8")
                    ).hexdigest()[:12],
                    "n": int(mask.sum()),
                    "positives": int(participant_y.sum()),
                    "auroc": float(
                        roc_auc_score(participant_y, selected_test_probabilities[mask])
                    ),
                    "auprc": float(
                        average_precision_score(participant_y, selected_test_probabilities[mask])
                    ),
                }
            )
    comparison["participant_reporting"]["eligible_participant_metrics"] = person_metrics
    comparison = _jsonable(comparison)

    artifact = {
        "model_version": MODEL_VERSION,
        "model": fitted_calibrators[selected_calibrator],
        "feature_schema": feature_schema,
        "target_status": "NOT_VERIFIED",
        "decision_threshold": None,
        "decision_use_approved": False,
        "explanation_scope": "global",
        "explanation_method": explanation_method,
        "global_importance": global_importance,
        "metadata": {
            "experiment_id": comparison["experiment_id"],
            "target_column": "lapse",
            "selected_candidate": selected_candidate["name"],
            "selected_calibration": selected_calibrator,
            "threshold_analysis": selected_threshold,
            "final_test_metrics": selected_test_metrics,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "git_head": comparison["reproducibility"]["git_head"],
            "seed": seed,
            "research_only": True,
            "clinical_use": False,
        },
    }
    model_card = {
        "name": "ANCHOR research model v2",
        "version": MODEL_VERSION,
        "status": "research_only",
        "target_status": "NOT_VERIFIED",
        "intended_use": "Exploratory probability estimation against the supplied, unverified lapse labels.",
        "not_intended_for": [
            "clinical decisions",
            "autonomous interventions",
            "diagnosis",
            "claims of prospective or personalized forecasting",
        ],
        "selected_candidate": selected_candidate["name"],
        "calibration": selected_calibrator,
        "decision_threshold": None,
        "required_history": "Feature timing is not verified; p168-named fields exceed a 72-hour warm-up.",
        "frontend_integration": "Not implemented; partner-owned frontend/backend files were not changed.",
        "model_artifact": OUTPUTS["model"].name,
    }

    existing = [str(path) for path in OUTPUTS.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite versioned outputs: " + ", ".join(existing)
        )
    oof_frame = pd.DataFrame(oof_columns)
    OUTPUTS["metrics"].write_text(
        json.dumps(comparison, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    oof_frame.to_parquet(OUTPUTS["oof"], index=False)
    OUTPUTS["schema"].write_text(
        json.dumps(feature_schema, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    OUTPUTS["card"].write_text(
        json.dumps(model_card, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    joblib.dump(artifact, OUTPUTS["model"])
    print("\nSelected model:", selected_candidate["name"])
    print("Selected calibration:", selected_calibrator)
    print("Independent locked-test metrics:", json.dumps(selected_test_metrics, indent=2))
    print("New artifact:", OUTPUTS["model"])
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    args = parser.parse_args()
    train(
        seed=args.seed,
        stride=args.stride,
        bootstrap_replicates=args.bootstrap_replicates,
    )


if __name__ == "__main__":
    main()
