# ANCHOR AI HEALTHCARE — PHASE 3: PREPROCESSING & PIPELINE SPECIFICATION
**Document ID:** `ANCHOR-PHASE3-PREPROCESSING-PIPELINE`  
**Execution Timestamp:** 2026-11-15T01:32:00+05:30  
**Research Pipeline:** Autonomous Machine Learning Architecture  
**Working Branch:** `mohit`  

---

## 1. Overview & Architectural Principles

The Anchor preprocessing architecture is strictly engineered to enforce zero-leakage training. All learned transformations (imputation parameters, one-hot category vocabularies, scalers) are encapsulated inside `sklearn.compose.ColumnTransformer` and fit **strictly on training partitions within cross-validation folds**.

Implementation source: [`research/preprocessing/pipeline.py`](file:///c:/Users/Mohit%20sharma/Desktop/7th%20sem/ai%20in%20heathcare/project/AI-HealthCare-/research/preprocessing/pipeline.py)

---

## 2. Feature-by-Feature Transformation Rules

| Column Family | Columns / Variables | Data Type | Preprocessing Transformation | Leakage Prevention Safeguard |
|:---|:---|:---:|:---|:---|
| **Identifiers & Labels** | `subid`, `dttm_label`, `label_num`, `lapse`, `y` | Metadata | Stripped from feature matrix; `subid` passed only as `groups` to CV splitter. | Guarantees zero target or ID leakage into weights. |
| **Categorical Demographics** | `demo_sex`, `demo_educ`, `demo_marital`, `demo_emp`, `label_day` | String / Object | `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` | Categories learned strictly from train fold. Unknown test levels map to all-zeros. |
| **Continuous Demographics** | `demo_age`, `demo_income` | Float32 | Passthrough for trees; `StandardScaler()` + median imputation for linear models. | Mean and standard deviation computed strictly on train folds. |
| **Past-Use History** | 10 count variables (`p12` to `p168`) | Float32 | Native float; zero-filled when lookback is unpopulated. | Pure historical aggregation backward in time. |
| **EMA Recent (`r`)** | 9 response variables (`ema_2` to `ema_10`) | Float32 | Native scale ($1–11$); native NaN handling in HistGradientBoosting. | Only trailing survey prompts accepted. |
| **Rolling Aggregations** | 234 rolling raw and change features | Float32 | Native float array with median imputation fallback for linear baselines. | Evaluated with trailing lookback window boundaries. |

---

## 3. HistGradientBoosting Early-Stopping Isolation

A crucial defect identified in standard `HistGradientBoostingClassifier(early_stopping=True)` is that scikit-learn's default internal validation split is a random row-wise split ($10\%$ of training rows). In longitudinal participant data, a random row split leaks observations from the same participant into the internal validation set!

### The Implemented Solution:
In [`updated-generate-report-lag.py`](file:///c:/Users/Mohit%20sharma/Desktop/7th%20sem/ai%20in%20heathcare/project/AI-HealthCare-/updated-generate-report-lag.py) and our pipeline, we explicitly enforce:
```python
HistGradientBoostingClassifier(
    max_iter=150,
    learning_rate=0.06,
    max_leaf_nodes=12,
    min_samples_leaf=100,
    l2_regularization=1.0,
    early_stopping=False, # Prevents random row-level leakage
    random_state=seed
)
```
Regularization is maintained via small leaf nodes (`max_leaf_nodes=12`), large minimum samples per leaf (`min_samples_leaf=100`), and explicit $L_2$ shrinkage (`l2_regularization=1.0`), guaranteeing strict participant-level isolation.
