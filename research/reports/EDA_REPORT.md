# ANCHOR AI HEALTHCARE — PHASE 5: EXPLORATORY DATA ANALYSIS REPORT
**Document ID:** `ANCHOR-PHASE5-EDA-REPORT`  
**Execution Timestamp:** 2026-11-15T01:33:02+05:30  
**Research Pipeline:** Autonomous Machine Learning Architecture  
**Working Branch:** `mohit`  
**Figures Generated:** Saved in [`research/figures/`](file:///c:/Users/Mohit%20sharma/Desktop/7th%20sem/ai%20in%20heathcare/project/AI-HealthCare-/research/figures/)

---

## 1. Visualized Findings & Scientific Interpretation

Five publication-grade visual exhibits were generated directly from the underlying dataset and historical out-of-fold predictions:

### Exhibit 1: Participant Lapse Clustering & Heavy Drinker Skew
- **Figure Location:** [`research/figures/figure1_participant_lapse_clustering.png`](file:///c:/Users/Mohit%20sharma/Desktop/7th%20sem/ai%20in%20heathcare/project/AI-HealthCare-/research/figures/figure1_participant_lapse_clustering.png)
- **Key Empirical Pattern:**
  - $44.4\%$ of the cohort ($67/151$) remained completely abstinent throughout monitoring.
  - The top $12$ participants ($8\%$ of the cohort) account for over **$50\%$ of all positive lapse windows in the dataset**.
  - One individual alone contributed **$1,193$ positive hours**.
- **Scientific Implication:** High global accuracy or unstratified AUROC is an illusion if the model merely identifies these $12$ individuals. Grouped cross-validation (`StratifiedGroupKFold`) is strictly necessary.

### Exhibit 2: Lag Horizon Degradation Trajectory
- **Figure Location:** [`research/figures/figure2_lag_horizon_degradation.png`](file:///c:/Users/Mohit%20sharma/Desktop/7th%20sem/ai%20in%20heathcare/project/AI-HealthCare-/research/figures/figure2_lag_horizon_degradation.png)
- **Key Empirical Pattern:**
  - **Population AUROC** gently declines from $0.908$ ($0\text{h}$) to $0.830$ ($336\text{h}$).
  - **Precision-Recall AUPRC** declines steeply from $0.658$ to $0.240$.
  - **Within-Person AUROC** drops from $0.623$ ($0\text{h}$) to $0.554$ ($24\text{h}$), hits **$0.500$ (chance)** at $72\text{h}$, and collapses to **$0.454$ (sub-chance)** at $336\text{h}$.
- **Scientific Implication:** At 2 weeks out, the model retains population-level discrimination (it knows *who* is a heavy drinker), but within an individual, it cannot distinguish safe hours from high-risk hours.

### Exhibit 3: Ablation Across Model Configurations (Lag 24h)
- **Figure Location:** [`research/figures/figure3_feature_family_ablation.png`](file:///c:/Users/Mohit%20sharma/Desktop/7th%20sem/ai%20in%20heathcare/project/AI-HealthCare-/research/figures/figure3_feature_family_ablation.png)
- **Key Empirical Pattern:**
  - Static traits alone: $\text{AUROC} = 0.504$ (pure chance).
  - Past-use lapses alone: $\text{AUROC} = 0.865, \text{AUPRC} = 0.438$.
  - Anchor Compact (25 inputs): $\text{AUROC} = 0.881, \text{AUPRC} = 0.459$.
  - Full model (283 inputs): $\text{AUROC} = 0.891, \text{AUPRC} = 0.471$.
- **Scientific Implication:** Moving from 25 inputs to 283 inputs yields only a $+0.010$ AUROC gain. A lightweight 25-feature mobile check-in captures $>98\%$ of the signal.

### Exhibit 4: Continuation vs. Clean Onset (Quiet Period Analysis)
- **Figure Location:** [`research/figures/figure4_continuation_vs_clean_onset.png`](file:///c:/Users/Mohit%20sharma/Desktop/7th%20sem/ai%20in%20heathcare/project/AI-HealthCare-/research/figures/figure4_continuation_vs_clean_onset.png)
- **Key Empirical Pattern:**
  - Full stream (continuation included): $\text{AUPRC} = 0.658$ (Prevalence: $7.7\%$).
  - Clean 7-day abstinence (`quiet168`): $\text{AUPRC} = 0.088$ (Prevalence: $2.2\%$).
  - Clean + No Past-Use features: $\text{AUPRC} = 0.068$.
- **Scientific Implication:** Predicting relapse after established abstinence is an order of magnitude harder than tracking an active drinker.

### Exhibit 5: Per-Person AUROC Distribution Shift
- **Figure Location:** [`research/figures/figure5_within_person_auc_distribution.png`](file:///c:/Users/Mohit%20sharma/Desktop/7th%20sem/ai%20in%20heathcare/project/AI-HealthCare-/research/figures/figure5_within_person_auc_distribution.png)
- **Key Empirical Pattern:**
  - At Lag 0h, median individual AUROC is $0.628$, with most participants $>0.50$.
  - At Lag 336h, the entire distribution shifts downwards: median drops to $0.454$, with more than $60\%$ of participants falling below the $0.50$ chance line.
