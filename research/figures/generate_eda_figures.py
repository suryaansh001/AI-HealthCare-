"""
research/figures/generate_eda_figures.py
Generates publication-quality figures for the Anchor Healthcare AI Research Reports.
Uses matplotlib to produce vector-quality PNGs in research/figures/.
"""
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "research" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Set clean scientific publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 200
})

# -------------------------------------------------------------
# Figure 1: Participant Lapse Clustering & Skew
# -------------------------------------------------------------
print("Generating Figure 1: Participant Lapse Clustering...")
parquet_path = ROOT / "dataset" / "features_0lag_v3.parquet"
if not parquet_path.exists():
    parquet_path = ROOT / "AI-HealthCare-" / "dataset" / "features_0lag_v3.parquet"

df = pd.read_parquet(parquet_path, columns=["subid", "lapse"])
df["y"] = df["lapse"].astype(str).str.strip().str.lower().isin(["yes", "1", "true"]).astype(int)
part_pos = df.groupby("subid")["y"].sum().sort_values(ascending=False).values

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1A: Ranked positive counts per participant
ax1.bar(range(len(part_pos)), part_pos, color="#2b5c8f", width=0.8)
ax1.axvline(x=84, color="#d9534f", linestyle="--", linewidth=1.5, label="84 Lapsers / 67 Abstinent")
ax1.set_title("A. Distribution of Positive Windows by Participant")
ax1.set_xlabel("Enrolled Participants (Ranked by Lapse Count)")
ax1.set_ylabel("Total Positive Windows (Hours)")
ax1.legend(loc="upper right")

# Plot 1B: Cumulative Share of Total Lapses
cum_share = np.cumsum(part_pos) / np.sum(part_pos) * 100
ax2.plot(range(len(part_pos)), cum_share, color="#d9534f", linewidth=2.5)
ax2.axhline(y=50, color="#666666", linestyle=":", label="50% of All Positive Labels")
ax2.axvline(x=12, color="#666666", linestyle=":", label="Top 12 Participants (8% Cohort)")
ax2.set_title("B. Cumulative Share of Dataset Positive Labels")
ax2.set_xlabel("Enrolled Participants (Ranked)")
ax2.set_ylabel("Cumulative Percentage of All Lapse Windows (%)")
ax2.set_ylim([0, 105])
ax2.legend(loc="lower right")

fig.tight_layout()
fig.savefig(FIG_DIR / "figure1_participant_lapse_clustering.png")
plt.close(fig)

# -------------------------------------------------------------
# Figure 2: Lag Horizon Performance Trajectory
# -------------------------------------------------------------
print("Generating Figure 2: Lag Horizon Trajectory...")
lags = ["0h (Now)", "24h (1 Day)", "72h (3 Days)", "168h (1 Wk)", "336h (2 Wks)"]
auroc = [0.908, 0.891, 0.876, 0.866, 0.830]
auprc = [0.658, 0.471, 0.415, 0.345, 0.240]
within = [0.623, 0.554, 0.500, 0.511, 0.454]
prevalence = [0.077, 0.077, 0.076, 0.076, 0.077]

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(lags, auroc, marker="o", linewidth=2.5, color="#1f77b4", label="Population AUROC (Discrimination)")
ax.plot(lags, auprc, marker="s", linewidth=2.5, color="#ff7f0e", label="Precision-Recall AUPRC (Precision)")
ax.plot(lags, within, marker="^", linewidth=2.5, color="#2ca02c", label="Within-Person AUROC (Personal Timing)")
ax.axhline(y=0.50, color="#888888", linestyle="--", linewidth=1.2, label="Chance Discrimination (0.50)")
ax.plot(lags, prevalence, color="#d62728", linestyle=":", linewidth=1.5, label="Base Prevalence (~7.7%)")

ax.set_title("Predictive Discrimination Across Lead Time Horizons (0h to 336h)")
ax.set_xlabel("Prediction Horizon (Lead Time Ahead of 1-Week Outcome Window)")
ax.set_ylabel("Metric Score")
ax.set_ylim([0.0, 1.0])
ax.legend(loc="upper right", frameon=True)

# Annotate the collapse
ax.annotate("Within-Person Chance (0.500)", xy=(2, 0.500), xytext=(2.2, 0.58),
            arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6))
ax.annotate("Sub-Chance Collapse (0.454)", xy=(4, 0.454), xytext=(3.3, 0.38),
            arrowprops=dict(facecolor="red", shrink=0.08, width=1, headwidth=6))

fig.tight_layout()
fig.savefig(FIG_DIR / "figure2_lag_horizon_degradation.png")
plt.close(fig)

# -------------------------------------------------------------
# Figure 3: Feature Family Ablation Comparison (Lag 24h)
# -------------------------------------------------------------
print("Generating Figure 3: Feature Family Ablation...")
models = ["Static Only\n(21 feats)", "Past Use\n(10 feats)", "Static + Past\n(31 feats)", 
          "Compact Mobile\n(25 feats)", "C + Current\n(40 feats)", "Full Model\n(283 feats)"]
auroc_vals = [0.504, 0.865, 0.858, 0.881, 0.885, 0.891]
auprc_vals = [0.072, 0.438, 0.419, 0.459, 0.465, 0.471]
within_vals = [0.564, 0.476, 0.512, 0.542, 0.548, 0.554]

x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(11, 5.5))
rects1 = ax.bar(x - width, auroc_vals, width, label="AUROC", color="#2b5c8f")
rects2 = ax.bar(x, auprc_vals, width, label="AUPRC", color="#e27c3e")
rects3 = ax.bar(x + width, within_vals, width, label="Within-Person AUROC", color="#4ba173")

ax.set_ylabel("Validation Score (5-Fold Grouped CV)")
ax.set_title("Ablation Comparison Across Model Configurations (Lag 24h)")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim([0, 1.05])
ax.legend(loc="upper left")

# Highlight compact model
ax.annotate("Anchor Compact\n(99% performance,\n4 check-in items)", xy=(3, 0.881), xytext=(2.6, 0.95),
            arrowprops=dict(facecolor="#2b5c8f", shrink=0.05, width=1, headwidth=5))

fig.tight_layout()
fig.savefig(FIG_DIR / "figure3_feature_family_ablation.png")
plt.close(fig)

# -------------------------------------------------------------
# Figure 4: Continuation vs Clean Onset (Quiet Period Impact)
# -------------------------------------------------------------
print("Generating Figure 4: Continuation vs Clean Onset...")
fig, ax = plt.subplots(figsize=(8, 5))
categories = ["Default Stream\n(Continuation Included)", "Quiet 168h\n(Clean 7-Day Abstinence)", "Quiet + No Past Use\n(Pure Psychological Onset)"]
auroc_c = [0.908, 0.780, 0.740]
auprc_c = [0.658, 0.088, 0.068]
prev_c = [0.077, 0.022, 0.022]

xc = np.arange(len(categories))
wc = 0.28
ax.bar(xc - wc/2, auroc_c, wc, label="AUROC", color="#1f77b4")
ax.bar(xc + wc/2, auprc_c, wc, label="AUPRC", color="#d9534f")
ax.plot(xc, prev_c, color="black", marker="D", linestyle="none", markersize=8, label="Base Prevalence")

for i in range(len(categories)):
    ax.text(xc[i] - wc/2, auroc_c[i] + 0.02, f"{auroc_c[i]:.3f}", ha="center", fontsize=9)
    ax.text(xc[i] + wc/2, auprc_c[i] + 0.02, f"{auprc_c[i]:.3f}", ha="center", fontsize=9)

ax.set_ylabel("Validation Score")
ax.set_title("Impact of Restricting Evaluation to Clean Abstinence (Lag 0h)")
ax.set_xticks(xc)
ax.set_xticklabels(categories)
ax.set_ylim([0, 1.05])
ax.legend(loc="upper right")

fig.tight_layout()
fig.savefig(FIG_DIR / "figure4_continuation_vs_clean_onset.png")
plt.close(fig)

# -------------------------------------------------------------
# Figure 5: Per-Person AUROC Distributions (Lag 0h vs Lag 336h)
# -------------------------------------------------------------
print("Generating Figure 5: Per-Person AUROC Distributions...")
p0_csv = ROOT / "updated_0lag_per_person_auc.csv"
p336_csv = ROOT / "updated_336lag_per_person_auc.csv"

p0_df = pd.read_csv(p0_csv)["G_E+change(full)"].dropna().values
p336_df = pd.read_csv(p336_csv)["G_E+change(full)"].dropna().values

fig, ax = plt.subplots(figsize=(8, 5))
box = ax.boxplot([p0_df, p336_df], labels=["Lag 0h (Immediate)", "Lag 336h (2 Weeks Ahead)"],
                 patch_artist=True, medianprops=dict(color="black", linewidth=2))

colors = ["#4ba173", "#d9534f"]
for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.axhline(y=0.50, color="#666666", linestyle="--", linewidth=1.2, label="Chance Ranking (0.50)")
ax.set_ylabel("Individual Participant AUROC")
ax.set_title("Within-Person Discrimination Distribution Across Participants")
ax.set_ylim([0.0, 1.05])
ax.legend(loc="lower left")

fig.tight_layout()
fig.savefig(FIG_DIR / "figure5_within_person_auc_distribution.png")
plt.close(fig)

print(f"All 5 figures successfully created and saved in {FIG_DIR}")
