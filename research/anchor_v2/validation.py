"""Grouped metrics and guarded chronological split utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold


class TargetTimingUnverified(ValueError):
    """Raised when chronological validation lacks a verified target contract."""


@dataclass(frozen=True)
class TargetTimingSpec:
    verified: bool
    prediction_time_column: str
    outcome_start_column: str
    outcome_end_column: str
    followup_end_column: str
    endpoint_convention: str


def grouped_partitions(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 7,
    development_folds: int = 3,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Split participants into development, calibration, threshold, and test sets."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    if len(y) != len(groups):
        raise ValueError("y and groups must have the same number of rows")
    if n_splits != development_folds + 4:
        raise ValueError("Need four separate folds for calibration-fit, calibration-selection, threshold, and test")
    if len(unique_groups) < n_splits:
        raise ValueError(f"Need at least {n_splits} participants for grouped partitions")

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [
        validation_indices
        for _, validation_indices in splitter.split(np.zeros(len(y)), y, groups)
    ]
    result = {
        "development": np.concatenate(folds[:development_folds]),
        "calibration_fit": folds[development_folds],
        "calibration_selection": folds[development_folds + 1],
        "threshold_selection": folds[development_folds + 2],
        "locked_test": folds[development_folds + 3],
    }
    names = list(result)
    for index, left_name in enumerate(names):
        left_groups = set(groups[result[left_name]])
        for right_name in names[index + 1 :]:
            if left_groups.intersection(groups[result[right_name]]):
                raise RuntimeError(f"Participant leakage between {left_name} and {right_name}")
    if len(np.unique(np.concatenate(list(result.values())))) != len(y):
        raise RuntimeError("Grouped partitions did not assign every row exactly once")
    return result


def metrics_at_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Compute probability metrics and optional threshold metrics."""
    y_true = np.asarray(y_true, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(y_true) == 0 or len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have equal, non-zero lengths")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite values between zero and one")

    result: dict[str, Any] = {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "auroc": (
            float(roc_auc_score(y_true, probabilities))
            if np.unique(y_true).size == 2
            else None
        ),
        "auprc": (
            float(average_precision_score(y_true, probabilities))
            if y_true.sum() > 0
            else None
        ),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, np.clip(probabilities, 1e-15, 1 - 1e-15), labels=[0, 1])),
    }
    if threshold is None:
        return result

    predicted = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if tp + fn else None
    specificity = float(tn / (tn + fp)) if tn + fp else None
    result.update(
        {
            "threshold": float(threshold),
            "precision": float(precision_score(y_true, predicted, zero_division=0)),
            "recall": float(recall_score(y_true, predicted, zero_division=0)),
            "f1": float(f1_score(y_true, predicted, zero_division=0)),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "balanced_accuracy": (
                float(balanced_accuracy_score(y_true, predicted))
                if sensitivity is not None and specificity is not None
                else None
            ),
            "alert_rate": float(predicted.mean()),
            "false_alerts_per_100_rows": float(100 * fp / len(y_true)),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }
    )
    return result


def participant_auc_summary(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    minimum_each_class: int = 3,
) -> dict[str, Any]:
    """Summarize within-participant ranking without treating ineligible people as 0.5."""
    frame = pd.DataFrame({"y": y_true, "probability": probabilities, "participant": groups})
    aucs: list[float] = []
    for _, participant in frame.groupby("participant", sort=False):
        positive = int(participant["y"].sum())
        negative = len(participant) - positive
        if positive >= minimum_each_class and negative >= minimum_each_class:
            aucs.append(float(roc_auc_score(participant["y"], participant["probability"])))
    return {
        "eligible_participants": len(aucs),
        "total_participants": int(frame["participant"].nunique()),
        "macro_auroc": float(np.mean(aucs)) if aucs else None,
        "median_auroc": float(np.median(aucs)) if aucs else None,
        "participant_auroc_25th": float(np.quantile(aucs, 0.25)) if aucs else None,
        "participant_auroc_75th": float(np.quantile(aucs, 0.75)) if aucs else None,
    }


def participant_bootstrap_intervals(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    replicates: int = 500,
    seed: int = 42,
) -> dict[str, list[float] | None]:
    """Bootstrap participant clusters for pooled probability and within-person metrics."""
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    row_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    within_values: dict[Any, float] = {}
    for group, indices in row_indices.items():
        target = y_true[indices]
        if target.sum() >= 3 and (len(target) - target.sum()) >= 3:
            within_values[group] = float(roc_auc_score(target, probabilities[indices]))

    samples: dict[str, list[float]] = {
        "auroc": [],
        "auprc": [],
        "brier": [],
        "log_loss": [],
        "within_person_auroc": [],
    }
    for _ in range(replicates):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([row_indices[group] for group in sampled_groups])
        target = y_true[indices]
        predicted = probabilities[indices]
        values = metrics_at_probabilities(target, predicted)
        for key in ("auroc", "auprc", "brier", "log_loss"):
            if values[key] is not None:
                samples[key].append(float(values[key]))
        sampled_within = [within_values[group] for group in sampled_groups if group in within_values]
        if sampled_within:
            samples["within_person_auroc"].append(float(np.mean(sampled_within)))

    return {
        key: (
            [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
            if values
            else None
        )
        for key, values in samples.items()
    }


def select_youden_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Find an exploratory Youden-J threshold; this does not establish alert utility."""
    y_true = np.asarray(y_true, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=float)
    if np.unique(y_true).size != 2:
        return {"threshold": None, "policy": "exploratory_youden_j", "status": "both classes required"}
    thresholds = np.unique(probabilities)
    candidates = np.concatenate(([0.0], thresholds, [1.0]))
    best_threshold = None
    best_j = -np.inf
    for threshold in candidates:
        metrics = metrics_at_probabilities(y_true, probabilities, float(threshold))
        sensitivity = metrics["sensitivity"]
        specificity = metrics["specificity"]
        if sensitivity is None or specificity is None:
            continue
        score = sensitivity + specificity - 1
        if score > best_j:
            best_j = score
            best_threshold = float(threshold)
    return {
        "threshold": best_threshold,
        "youden_j": float(best_j),
        "policy": "exploratory_youden_j",
        "status": "operating point only; not a clinically justified decision rule",
    }


@dataclass(frozen=True)
class ChronologicalSplit:
    train_indices: np.ndarray
    test_indices: np.ndarray
    purged_training_rows: int
    incomplete_training_rows: int
    incomplete_test_rows: int


def make_purged_chronological_split(
    frame: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    specification: TargetTimingSpec,
) -> ChronologicalSplit:
    """Split only when explicit, verified target intervals and follow-up are supplied."""
    if not specification.verified:
        raise TargetTimingUnverified(
            "Chronological evaluation is blocked until local target timing and endpoint rules are verified"
        )
    if not specification.endpoint_convention.strip():
        raise ValueError("An explicit inclusive/exclusive endpoint convention is required")
    columns = [
        specification.prediction_time_column,
        specification.outcome_start_column,
        specification.outcome_end_column,
        specification.followup_end_column,
    ]
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Verified target interval columns are missing: {missing}")

    times = {
        column: pd.to_datetime(frame[column], utc=True, errors="coerce")
        for column in columns
    }
    required_times = columns[:-1]
    if any(times[column].isna().any() for column in required_times):
        raise ValueError("Prediction and outcome interval timestamps must be valid")
    prediction = times[specification.prediction_time_column]
    outcome_start = times[specification.outcome_start_column]
    outcome_end = times[specification.outcome_end_column]
    followup_end = times[specification.followup_end_column]
    cutoff_time = pd.to_datetime(cutoff, utc=True)
    if (outcome_start < prediction).any() or (outcome_end < outcome_start).any():
        raise ValueError("Outcome intervals must begin after prediction and have non-negative duration")

    complete = followup_end.notna() & (followup_end >= outcome_end)
    train_mask = (prediction < cutoff_time) & (outcome_end < cutoff_time) & complete
    test_mask = (prediction >= cutoff_time) & (outcome_start >= cutoff_time) & complete
    purged = int(((prediction < cutoff_time) & (outcome_end >= cutoff_time)).sum())
    incomplete_training = int(((prediction < cutoff_time) & ~complete).sum())
    incomplete_test = int(((prediction >= cutoff_time) & ~complete).sum())
    train_indices = np.flatnonzero(train_mask.to_numpy())
    test_indices = np.flatnonzero(test_mask.to_numpy())
    if len(train_indices) == 0 or len(test_indices) == 0:
        raise ValueError("Chronological cutoff produced an empty train or fully observed test set")
    return ChronologicalSplit(
        train_indices=train_indices,
        test_indices=test_indices,
        purged_training_rows=purged,
        incomplete_training_rows=incomplete_training,
        incomplete_test_rows=incomplete_test,
    )


def make_multiple_chronological_splits(
    frame: pd.DataFrame,
    cutoffs: Sequence[str | pd.Timestamp],
    specification: TargetTimingSpec,
) -> list[ChronologicalSplit]:
    """Apply the same verified temporal contract at several independent cutoffs."""
    if len(cutoffs) < 2:
        raise ValueError("At least two chronological cutoffs are required")
    return [
        make_purged_chronological_split(frame, cutoff, specification)
        for cutoff in cutoffs
    ]
