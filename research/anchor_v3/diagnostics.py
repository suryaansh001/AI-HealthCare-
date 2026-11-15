"""Participant-level diagnostics with pseudonymous output identifiers."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def participant_token(participant_id: Any) -> str:
    """Create a stable, non-reversible token for internal report tables."""
    return hashlib.sha256(f"anchor-v3:{participant_id}".encode("utf-8")).hexdigest()[:12]


def participant_metrics(
    target: np.ndarray,
    group_ids: np.ndarray,
    probabilities: dict[str, np.ndarray],
    history_rows: np.ndarray,
    long_history: np.ndarray,
    threshold_by_model: dict[str, float | None],
    minimum_each_class: int = 3,
) -> pd.DataFrame:
    """Return one anonymized row per participant with discrimination and error counts."""
    base = pd.DataFrame(
        {
            "participant": group_ids,
            "y": target,
            "history_rows": history_rows,
            "long_history": long_history,
        }
    )
    records: list[dict[str, Any]] = []
    for participant, rows in base.groupby("participant", sort=False):
        y = rows["y"].to_numpy(dtype=np.int8)
        positives = int(y.sum())
        negatives = int(len(y) - positives)
        record: dict[str, Any] = {
            "participant_token": participant_token(participant),
            "n_rows": int(len(y)),
            "positive_rows": positives,
            "negative_rows": negatives,
            "prevalence": float(y.mean()),
            "median_prior_sample_rows": float(rows["history_rows"].median()),
            "median_p168_count": (
                float(rows["long_history"].median())
                if rows["long_history"].notna().any()
                else None
            ),
            "within_person_auc_eligible": bool(
                positives >= minimum_each_class and negatives >= minimum_each_class
            ),
        }
        for model_name, scores in probabilities.items():
            score_frame = pd.DataFrame(
                {"participant": group_ids, "score": scores}
            )
            person_scores = score_frame.loc[
                score_frame["participant"] == participant, "score"
            ].to_numpy()
            record[f"{model_name}_auc"] = (
                float(roc_auc_score(y, person_scores))
                if positives >= minimum_each_class and negatives >= minimum_each_class
                else None
            )
            record[f"{model_name}_auprc"] = (
                float(average_precision_score(y, person_scores))
                if positives > 0
                else None
            )
            record[f"{model_name}_brier"] = float(
                brier_score_loss(y, person_scores)
            )
            threshold = threshold_by_model.get(model_name)
            if threshold is not None:
                predicted = person_scores >= threshold
                record[f"{model_name}_false_positives"] = int(
                    np.sum(predicted & (y == 0))
                )
                record[f"{model_name}_false_negatives"] = int(
                    np.sum(~predicted & (y == 1))
                )
        records.append(record)
    return pd.DataFrame(records).sort_values("participant_token").reset_index(drop=True)


def prediction_variance_diagnostics(
    target: np.ndarray,
    group_ids: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    """Separate participant-mean score variation from within-person score variation."""
    frame = pd.DataFrame(
        {"participant": group_ids, "y": target, "probability": probabilities}
    )
    means = frame.groupby("participant", sort=False).agg(
        mean_probability=("probability", "mean"),
        prevalence=("y", "mean"),
        n=("y", "size"),
    )
    aligned_means = frame["participant"].map(means["mean_probability"]).to_numpy()
    between = float(np.var(aligned_means))
    within = float(np.mean((probabilities - aligned_means) ** 2))
    variance_share = between / (between + within) if between + within > 0 else None
    if len(means) >= 3 and means["mean_probability"].nunique() > 1 and means["prevalence"].nunique() > 1:
        correlation = float(
            spearmanr(means["mean_probability"], means["prevalence"]).statistic
        )
    else:
        correlation = None
    if np.var(probabilities) > 0:
        centered = probabilities - aligned_means
        centered_auc = (
            float(roc_auc_score(target, centered))
            if np.unique(target).size == 2
            else None
        )
    else:
        centered_auc = None
    return {
        "participant_mean_probability_variance_share": variance_share,
        "spearman_participant_mean_probability_vs_prevalence": correlation,
        "auroc_of_participant_mean_centered_scores": centered_auc,
        "participant_mean_probability_range": [
            float(means["mean_probability"].min()),
            float(means["mean_probability"].max()),
        ],
        "participant_prevalence_range": [
            float(means["prevalence"].min()),
            float(means["prevalence"].max()),
        ],
        "interpretation": (
            "Descriptive decomposition only. It cannot distinguish true personal risk "
            "from label/follow-up differences or establish forecasting."
        ),
    }


def history_strata_metrics(
    target: np.ndarray,
    predictions: dict[str, np.ndarray],
    group_ids: np.ndarray,
    history_rows: np.ndarray,
) -> pd.DataFrame:
    """Summarize row-level metrics by prior sampled-row count (not elapsed hours)."""
    bands = [
        ("0-5 prior rows", history_rows <= 5),
        ("6-23 prior rows", (history_rows >= 6) & (history_rows < 24)),
        ("24+ prior rows", history_rows >= 24),
    ]
    records = []
    for band_name, mask in bands:
        y = target[mask]
        for model_name, scores in predictions.items():
            selected = scores[mask]
            records.append(
                {
                    "history_band": band_name,
                    "model": model_name,
                    "rows": int(mask.sum()),
                    "participants": int(np.unique(group_ids[mask]).size),
                    "positives": int(y.sum()),
                    "prevalence": float(y.mean()) if len(y) else None,
                    "auroc": (
                        float(roc_auc_score(y, selected))
                        if len(y) and np.unique(y).size == 2
                        else None
                    ),
                    "auprc": (
                        float(average_precision_score(y, selected))
                        if len(y) and y.sum()
                        else None
                    ),
                    "brier": (
                        float(brier_score_loss(y, selected))
                        if len(y)
                        else None
                    ),
                }
            )
    return pd.DataFrame(records)


def paired_participant_bootstrap(
    target: np.ndarray,
    groups: np.ndarray,
    reference: np.ndarray,
    candidates: dict[str, np.ndarray],
    replicates: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, list[float] | None]]:
    """Estimate paired participant-cluster bootstrap intervals for metric deltas."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    indices = {person: np.flatnonzero(groups == person) for person in unique}
    eligible = [
        person
        for person in unique
        if target[indices[person]].sum() >= 3
        and (len(indices[person]) - target[indices[person]].sum()) >= 3
    ]
    names = ["auroc", "auprc", "brier", "log_loss", "within_person_auroc"]
    output: dict[str, dict[str, list[float] | None]] = {}
    for name, candidate in candidates.items():
        differences = {metric: [] for metric in names}
        for _ in range(replicates):
            sampled = rng.choice(unique, size=len(unique), replace=True)
            row_indices = np.concatenate([indices[person] for person in sampled])
            sample_y = target[row_indices]
            if np.unique(sample_y).size < 2:
                continue
            reference_values = {
                "auroc": roc_auc_score(sample_y, reference[row_indices]),
                "auprc": average_precision_score(sample_y, reference[row_indices]),
                "brier": brier_score_loss(sample_y, reference[row_indices]),
                "log_loss": _log_loss(sample_y, reference[row_indices]),
            }
            candidate_values = {
                "auroc": roc_auc_score(sample_y, candidate[row_indices]),
                "auprc": average_precision_score(sample_y, candidate[row_indices]),
                "brier": brier_score_loss(sample_y, candidate[row_indices]),
                "log_loss": _log_loss(sample_y, candidate[row_indices]),
            }
            for metric in names[:-1]:
                differences[metric].append(
                    float(candidate_values[metric] - reference_values[metric])
                )
            sampled_eligible = rng.choice(eligible, size=len(eligible), replace=True) if eligible else []
            if len(sampled_eligible):
                per_person_delta = []
                for person in sampled_eligible:
                    person_rows = indices[person]
                    person_y = target[person_rows]
                    per_person_delta.append(
                        roc_auc_score(person_y, candidate[person_rows])
                        - roc_auc_score(person_y, reference[person_rows])
                    )
                differences["within_person_auroc"].append(
                    float(np.mean(per_person_delta))
                )
        output[name] = {
            metric: (
                [
                    float(np.quantile(values, 0.025)),
                    float(np.quantile(values, 0.975)),
                ]
                if values
                else None
            )
            for metric, values in differences.items()
        }
    return output


def _log_loss(target: np.ndarray, probabilities: np.ndarray) -> float:
    from sklearn.metrics import log_loss

    return float(log_loss(target, np.clip(probabilities, 1e-15, 1 - 1e-15), labels=[0, 1]))
