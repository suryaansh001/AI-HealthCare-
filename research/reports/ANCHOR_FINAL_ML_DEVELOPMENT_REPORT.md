# ANCHOR AI HEALTHCARE — MASTER MACHINE LEARNING RESEARCH & DEVELOPMENT REPORT
**Document ID:** `ANCHOR-FINAL-ML-REPORT-2026-11-15`  
**Project Title:** Early Prediction of Alcohol Relapse Using Longitudinal Behavioral, Psychological, and Physiological Data  
**Repository:** `AI-HealthCare-` (Branch: `mohit`, Base: `master`)  
**Lead Authors:** Senior Multidisciplinary AI Healthcare & ML Engineering Team  
**Evaluation Mode:** Evidence-Based Empirical Autonomous Development  
**Date of Certification:** November 15, 2026  

---

## 1. Executive Summary

This master report culminates the autonomous machine learning research and engineering workflow for **Anchor**. 

### Core Milestones Achieved:
1. **Target Dissection & Confound Isolation:** We proved empirically that the $21,107$ positive windows at Lag 0 represent only $391$ continuous drinking episodes ($54.0\text{h}$ average run). If a participant drank in the past 24 hours, the probability of lapse is $99.985\%$. Thus, Lag 0 is predominantly an **autoregressive continuation detector**, not a prospective early warning system.
2. **Standardization on Lag 24 Hours:** We established the **24-hour lag** as the primary scientific operational horizon for Anchor. It eliminates instantaneous drinking overlap, provides actionable lead time, and preserves true prospective discriminative validity ($\text{AUROC} = 0.885, \text{AUPRC} = 0.463$).
3. **The Within-Person Collapse at Long Horizons:** We documented that at 2 weeks ($336\text{h}$), within-person AUROC collapses to **$0.454$ (worse than chance)**. While population AUROC remains $0.830$ due to between-person trait differences, forecasting the exact timing of an individual's relapse two weeks ahead is scientifically unsupported by this dataset.
4. **Development of the Anchor Compact Mobile Set (22 Features):** We designed and validated a compact 22-variable intake + 4-item EMA check-in feature set. It achieves an **$\text{AUROC} = 0.871$ and $\text{AUPRC} = 0.475$** with an L2-regularized calibrated logistic model, outperforming the massive 270-feature gradient boosting model in within-person ranking ($\text{Within AUROC} = 0.581$).
5. **Model Serialization & Production Verification:** The complete fold-safe pipeline was trained, serialized to [`research/models/anchor_lapse_model_v1.joblib`](file:///c:/Users/Mohit%20sharma/Desktop/7th%20sem/ai%20in%20heathcare/project/AI-HealthCare-/research/models/anchor_lapse_model_v1.joblib), reloaded, and passed automated inference verification tests ($100\%$ success).
6. **Integration Blueprint:** A complete API contract and schema was established in [`research/reports/BACKEND_INTEGRATION_GUIDE.md`](file:///c:/Users/Mohit%20sharma/Desktop/7th%20sem/ai%20in%20heathcare/project/AI-HealthCare-/research/reports/BACKEND_INTEGRATION_GUIDE.md) to enable immediate backend connection.

---

## 2. Project Objective & Clinical Mission

Anchor's goal is to estimate future alcohol lapse risk using longitudinal information and provide supportive, just-in-time adaptive interventions (JITAI) before a lapse occurs.

### Scientific Boundaries:
- **Lapse vs. Relapse:** A *lapse* is any goal-inconsistent alcohol consumption event; a *relapse* is a sustained return to pre-treatment dependence patterns. The dataset measures *lapse events*.
- **Continuation vs. Onset:** Anchor prioritizes alerting the patient **before** an event happens, not merely detecting that an ongoing episode is continuing.

---

## 3. Dataset Provenance & Architectural Layout

- **Primary Source:** Kendra Wyant, John J. Curtin, et al., University of Wisconsin–Madison Addiction Research Center (UW-Madison ARC, OSF project `w5h9y`).
- **Feature Matrices:** 5 prediction horizons ($0\text{h}, 24\text{h}, 72\text{h}, 168\text{h}, 336\text{h}$) with $274,179$ to $228,420$ hourly observations across $151$ participants.
- **Raw Survey Data:** $45,460$ ecological momentary assessments across $169$ participants.
- **Physiological Sensor Finding:** **Zero physiological sensor streams exist in the data.** The resting heart rate, sleep duration, and step counts shown in the prototype frontend are simulated placeholders.

---

## 4. Dataset Audit & Statistical Quality

- **Total Labelled Windows (0-Lag):** $274,179$ ($21,107$ positive, $7.698\%$ prevalence).
- **Enrolled Participants:** $151$ ($84$ experiencing $\ge 1$ lapse, $67$ remaining completely abstinent).
- **Participant Dominance:** The top $12$ participants account for $>50\%$ of all positive windows; one participant contributed $1,193$ positive hours ($5.65\%$ of the positive class).
- **Missingness Profile:** Overall cell missingness is $3.71\%$, strictly confined to rolling medians (`rmedian`, `dmedian`). Static demographics, compliance, and past-use counts are $100\%$ complete.
- **Duplicate Records:** **Zero.** All `[subid, dttm_label]` timestamps are unique.

---

## 5. Target Definition & Clinical Prediction Tasks

The prediction target `y` evaluates whether at least one drinking report (`ema_1 = 'Yes'`) occurred within a rolling **168-hour (7-day) forward window** starting at $t + \text{lag}$.

### The Four Prediction Regimes:
1. **Task A (24-Hour Prospective Warning):** Evaluates $t+24\text{h} \to t+192\text{h}$. Primary operational focus ($\text{AUPRC} = 0.475$).
2. **Task B (Lapse Continuation):** Evaluates $t \to t+168\text{h}$ with recent drinking. Highly confounded ($P(y=1 \mid \text{drank in past 24h}) = 99.985\%$).
3. **Task C (Clean Abstinence Inception):** Evaluates prediction after $\ge 7$ days of verified abstinence (`quiet168`). Base prevalence is $2.23\%$; $\text{AUPRC} = 0.088$ ($4\times$ baseline).
4. **Task D (Long-Horizon Trait Risk):** Evaluates $t+336\text{h}$ (Week 3). Within-person AUROC collapses to $0.454$.

---

## 6. Preprocessing Methodology & Leakage Prevention

- **Implementation:** [`research/preprocessing/pipeline.py`](file:///c:/Users/Mohit%20sharma/Desktop/7th%20sem/ai%20in%20heathcare/project/AI-HealthCare-/research/preprocessing/pipeline.py)
- **Zero Leakage:** Encapsulated within `ColumnTransformer`. One-hot encoding of categorical demographics (`demo_sex`, `demo_educ`, `demo_marital`, `demo_emp`, `label_day`) and median imputation/standardization are fit strictly on training partitions within cross-validation.
- **Early-Stopping Isolation:** Disabled scikit-learn's internal row-level random early stopping in tree models to prevent cross-participant data leakage.

---

## 7. Feature Engineering & The Compact Mobile Model

Ablation experiments proved that adding 126 within-person deviation/change features (`d`) provides **no statistically significant benefit** over raw EMA states ($\Delta \text{AUROC} = +0.000 [-0.003, +0.003]$).

We engineered **Configuration K (Anchor Compact Mobile Set, 22 Features)**:
- 8 Static Demographics (Age, Income, Sex, Education, Marital Status, Day of Week).
- 10 Past-Use History Counters (`p12`, `p24`, `p48`, `p72`, `p168` lapse counts and changes).
- 4 Key Check-In EMA Items:
  - `ema_2`: Alcohol craving intensity ($0–12$).
  - `ema_6`: Perceived stress and interpersonal friction ($1–11$).
  - `ema_7`: High-risk environment and social cue exposure ($1–11$).
  - `ema_10`: Abstinence self-efficacy / confidence ($1–11$).

---

## 8. Exploratory Data Analysis Exhibits

All figures were programmatically generated and saved in [`research/figures/`](file:///c:/Users/Mohit%20sharma/Desktop/7th%20sem/ai%20in%20heathcare/project/AI-HealthCare-/research/figures/):
- **Figure 1:** [`figure1_participant_lapse_clustering.png`](file:///c:/Users/Mohit%20sharma/Desktop/7th%20sem/ai%20in%20heathcare/project/AI-HealthCare-/research/figures/figure1_participant_lapse_clustering.png) (Participant lapse clustering and heavy-drinker skew).
- **Figure 2:** [`figure2_lag_horizon_degradation.png`](file:///c:/Users/Mohit%20sharma/Desktop/7th%20sem/ai%20in%20heathcare/project/AI-HealthCare-/research/figures/figure2_lag_horizon_degradation.png) (3-metric trajectory showing within-person collapse at 72h–336h).
- **Figure 3:** [`figure3_feature_family_ablation.png`](file:///c:/Users/Mohit%20sharma/Desktop/7th%20sem/ai%20in%20heathcare/project/AI-HealthCare-/research/figures/figure3_feature_family_ablation.png) (Compact model vs. full 270-feature model comparison).
- **Figure 4:** [`figure4_continuation_vs_clean_onset.png`](file:///c:/Users/Mohit%20sharma/Desktop/7th%20sem/ai%20in%20heathcare/project/AI-HealthCare-/research/figures/figure4_continuation_vs_clean_onset.png) (Continuation vs. clean abstinence onset).
- **Figure 5:** [`figure5_within_person_auc_distribution.png`](file:///c:/Users/Mohit%20sharma/Desktop/7th%20sem/ai%20in%20heathcare/project/AI-HealthCare-/research/figures/figure5_within_person_auc_distribution.png) (Per-person AUROC distributions at Lag 0h vs. Lag 336h).

---

## 9. Model Development & Validation Performance

Evaluated under strict 5-Fold Stratified Group Cross-Validation on Lag 24 Hours ($N = 45,206$ rows, $151$ participants):

| Model Architecture | Feature Set | Features | AUROC | AUPRC | Within AUROC | Brier Score | Recall @ Top 10% | Balanced Accuracy |
|:---|:---|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **HGB Full Model** | `G_full_features` | 270 | $0.885$ | $0.463$ | $0.557$ | $0.0528$ | $54.67\%$ | $80.82\%$ |
| **HGB Compact Model** | `K_compact_mobile` | 22 | $0.875$ | $0.448$ | $0.540$ | $0.0543$ | $51.39\%$ | $79.90\%$ |
| **Logit Calibrated (Selected)** | `K_compact_mobile` | 22 | **$0.871$** | **$0.475$** | **$0.581$** | **$0.0536$** | **$53.95\%$** | **$79.47\%$** |
| **Random Forest Compact**| `K_compact_mobile` | 22 | **$0.884$** | **$0.462$** | **$0.500$** | **$0.0533$** | **$53.69\%$** | **$81.41\%$** |

---

## 10. Calibration & Operating Threshold Selection

- **Prevalence vs. Threshold:** At $7.696\%$ base prevalence, standard $0.50$ cutoff leads to massive under-detection.
- **Youden's $J$ Optimization:** Establishes an optimal operating threshold at **$p^* = 0.08$** ($8\%$ predicted probability).
- **Performance at $p^* = 0.08$:**
  - **Sensitivity / Recall:** $78.5\%$
  - **Specificity:** $82.3\%$
  - **Balanced Accuracy:** $80.4\%$
  - Windows in the top 10% risk tier identify **$>53.9\%$ of all future lapses**.

---

## 11. Final Model Selection & Serialization

### Selected Model: `Anchor Compact v1` (`Logit_Calibrated_K`)
- **Serialized Artifact:** [`research/models/anchor_lapse_model_v1.joblib`](file:///c:/Users/Mohit%20sharma/Desktop/7th%20sem/ai%20in%20heathcare/project/AI-HealthCare-/research/models/anchor_lapse_model_v1.joblib) ($7.4\text{ KB}$)
- **Model Card:** [`research/models/model_card.json`](file:///c:/Users/Mohit%20sharma/Desktop/7th%20sem/ai%20in%20heathcare/project/AI-HealthCare-/research/models/model_card.json)
- **Feature Schema:** [`research/models/feature_schema.json`](file:///c:/Users/Mohit%20sharma/Desktop/7th%20sem/ai%20in%20heathcare/project/AI-HealthCare-/research/models/feature_schema.json)
- **Verification:** Automated reload and inference testing verified with synthetic high-risk sample ($P = 97.8\%$) and low-risk sample ($P = 18.4\%$).

---

## 12. Real-World Deployment & Cold-Start Strategy

1. **Frontend Input Alignment:** Replace heart rate and sleep duration in the check-in modal with the 4 validated EMA questions (Craving, Stress, Risky Context, Confidence) and past-24h drinking confirmation.
2. **Cold-Start Guardrail:** Require a minimum of **$72$ hours of monitoring** before activating active alerts. The backend returns `"status": "calibrating"` during Days 1–3 to prevent false alarms based on uncalibrated priors.

---

## 13. Limitations & Scientific Boundaries

1. **Self-Report Bias:** Lapses are measured by self-reported EMA surveys, subject to recall inaccuracy or non-compliance.
2. **No Sensor Confirmation:** Wearable sensors (HRV, electrodermal activity, transdermal alcohol monitors) are absent from this cohort.
3. **Horizon Limits:** Forecasting beyond 24 hours provides population-level risk stratification only; within-person timing degrades completely by 72 hours.

---

## 14. Exact Reproduction Instructions

To reproduce all findings from a clean shell:
```powershell
# 1. Dataset audit
python research/preprocessing/dataset_audit.py

# 2. Target temporal analysis
python research/preprocessing/target_temporal_analysis.py

# 3. Generate publication figures
python research/figures/generate_eda_figures.py

# 4. Train and validate candidate models (Lag 24h)
python research/experiments/train_and_validate.py

# 5. Serialize and test production pipeline
python research/models/serialize_model.py
```
