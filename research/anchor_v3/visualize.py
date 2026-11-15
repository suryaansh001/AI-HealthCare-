"""Save de-identified diagnostics plots for ANCHOR v3."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve


def save_diagnostic_figures(
    output_dir: Path,
    participants: pd.DataFrame,
    history_metrics: pd.DataFrame,
    importances: pd.DataFrame,
    participant_means: pd.DataFrame,
    test_target: np.ndarray,
    probabilities: dict[str, np.ndarray],
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    eligible = participants.loc[participants["v2_random_forest_auc"].notna()].sort_values(
        "v2_random_forest_auc"
    )
    fig, axis = plt.subplots(figsize=(10, 5))
    if not eligible.empty:
        axis.barh(eligible["participant_token"], eligible["v2_random_forest_auc"], color="#4776A8")
        axis.axvline(0.5, color="black", linestyle="--", linewidth=1)
        axis.set_xlabel("Within-participant AUROC")
        axis.set_ylabel("Pseudonymous participant token")
        axis.set_title("v2 locked-test within-participant discrimination")
    else:
        axis.text(0.5, 0.5, "No participants met the pre-specified class-count rule", ha="center")
        axis.axis("off")
    fig.tight_layout()
    path = output_dir / "participant_auc_v3.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    created.append(path.name)

    fig, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(
        participant_means["prevalence"],
        participant_means["mean_probability"],
        s=30 + 8 * np.sqrt(participant_means["n"]),
        alpha=0.8,
        color="#D47B32",
    )
    axis.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Observed participant label prevalence")
    axis.set_ylabel("Mean v2 predicted probability")
    axis.set_title("Participant mean scores and stored-label prevalence")
    fig.tight_layout()
    path = output_dir / "participant_mean_risk_v3.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    created.append(path.name)

    plotted = history_metrics.loc[
        history_metrics["model"].isin(
            ["v2_random_forest", "personalized_ema_hybrid_rf", "full_personalized_rf"]
        )
    ]
    fig, axis = plt.subplots(figsize=(9, 5))
    positions = np.arange(3)
    width = 0.25
    history_bands = ["0-5 prior rows", "6-23 prior rows", "24+ prior rows"]
    models = list(plotted["model"].drop_duplicates())
    for model_index, model in enumerate(models):
        vals = []
        for band in history_bands:
            selected = plotted.loc[
                (plotted["model"] == model) & (plotted["history_band"] == band),
                "auprc",
            ]
            vals.append(
                float(selected.iloc[0])
                if len(selected) and pd.notna(selected.iloc[0])
                else np.nan
            )
        axis.bar(positions + model_index * width, vals, width=width, label=model)
    axis.set_xticks(positions + width * (max(len(models), 1) - 1) / 2)
    axis.set_xticklabels(history_bands)
    axis.set_ylabel("AUPRC")
    axis.set_title("Test-set AUPRC by number of prior sampled rows")
    axis.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "history_amount_metrics_v3.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    created.append(path.name)

    top = importances.head(15).sort_values("importance")
    fig, axis = plt.subplots(figsize=(10, 6))
    if not top.empty:
        axis.barh(top["feature"], top["importance"], color="#579B79")
        axis.set_xlabel("Random Forest impurity importance")
        axis.set_title("v2 model global feature importance (associational)")
    else:
        axis.text(0.5, 0.5, "Feature importance unavailable", ha="center")
        axis.axis("off")
    fig.tight_layout()
    path = output_dir / "v2_feature_importance_v3.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    created.append(path.name)

    fig, axis = plt.subplots(figsize=(7, 6))
    for name in ("v2_random_forest", "personalized_ema_hybrid_rf", "full_personalized_rf"):
        if name not in probabilities:
            continue
        observed, predicted = calibration_curve(
            test_target,
            probabilities[name],
            n_bins=10,
            strategy="quantile",
        )
        axis.plot(predicted, observed, marker="o", label=name)
    axis.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed positive fraction")
    axis.set_title("Locked-test reliability diagnostics")
    axis.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "calibration_diagnostics_v3.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    created.append(path.name)
    return created
