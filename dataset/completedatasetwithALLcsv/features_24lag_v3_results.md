# Lag 24 hours: exploratory lapse-forecasting results

## Analysis scope

This report analyzes one engineered Curtin-lab feature file. The outcome is a goal-inconsistent drinking window, not clinical relapse. Rows were sampled at every sixth hourly row, participants were kept together in five-fold stratified group cross-validation, and uncertainty intervals were bootstrapped over participants. These are exploratory results.

## 1. Basic EDA

| Quantity | Result |
|---|---:|
| Labelled windows before thinning | 270,911 |
| Participants | 151 |
| Positive windows | 20,800 |
| Positive-window rate | 7.68% |
| Date range | 2017-03-04 06:00:00+00:00 to 2019-12-04 00:00:00+00:00 |
| Rows used in modelling after stride 6 | 45,206 |
| Participants with at least one positive window | 84 |
| Participants with no positive window | 67 |
| Median rows per participant before thinning | 1986 |
| Positive windows per participant: median / mean / maximum | 24 / 137.7 / 1188 |
| Consecutive gaps exactly 1 hour | 99.57% |
| Largest consecutive gap | 261 hours |

Feature missingness was summarized by category. The category labels come from the column names; no item meanings were inferred from correlations.

| Feature category | Number of columns | Mean missingness |
|---|---:|---:|
| static | 8 | 0.00% |
| past_use | 10 | 0.00% |
| compliance | 10 | 0.00% |
| current_raw | 9 | 0.05% |
| current_change | 9 | 0.05% |
| aggregate_raw | 117 | 4.51% |
| aggregate_change | 117 | 4.51% |

Rows in each participant's first 24, 72, and 168 hours were 1.31%, 3.70%, and 8.48%, respectively.

## 2. Category-wise predictive results

AUPRC should be compared with the positive rate above, while AUROC should be compared with 0.50. The within-person metric averages AUROC only for participants with at least three positive and three negative rows after thinning.

| Feature set | Features | AUROC | AUPRC | Within-person AUROC | Eligible people | Brier |
|---|---:|---:|---:|---:|---:|---:|
| A static | 21 | 0.500 | 0.072 | 0.566 | 84 | 0.080 |
| B past use | 10 | 0.866 | 0.440 | 0.477 | 84 | 0.055 |
| C static + past use | 31 | 0.860 | 0.423 | 0.506 | 84 | 0.056 |
| D + current raw | 40 | 0.885 | 0.464 | 0.554 | 84 | 0.053 |
| E + raw state | 157 | 0.889 | 0.476 | 0.559 | 84 | 0.053 |
| F change only | 157 | 0.884 | 0.456 | 0.539 | 84 | 0.053 |
| G full + change | 283 | 0.891 | 0.466 | 0.556 | 84 | 0.053 |
| H all + compliance | 293 | 0.891 | 0.471 | 0.558 | 84 | 0.053 |

### Paired differences with 95% participant-bootstrap intervals

| Contrast | AUROC difference | AUPRC difference | Within-person difference |
|---|---:|---:|---:|
| D minus C | +0.024 [+0.014, +0.038] | +0.041 [+0.009, +0.082] | +0.048 [+0.019, +0.075] |
| E minus D | +0.004 [-0.002, +0.011] | +0.012 [-0.025, +0.048] | +0.004 [-0.019, +0.028] |
| G minus E | +0.002 [-0.002, +0.008] | -0.010 [-0.043, +0.036] | -0.003 [-0.026, +0.013] |
| F minus E | -0.005 [-0.014, +0.004] | -0.020 [-0.076, +0.043] | -0.019 [-0.050, +0.007] |

## 3. Hypotheses and plain-language results

### H1: Past drinking history is useful for prediction

**Result: supported, with an important caution.** Past-use-only AUROC was 0.866, compared with 0.50 for chance. However, the source data contain overlapping outcome windows, so past-use performance may partly reflect persistence from a recent lapse rather than genuinely early warning.

**Situation example:** If someone has recently reported drinking, many nearby hourly windows are also labelled positive. A model can score those windows correctly using recent drinking history, even if it is not predicting a completely new episode far in advance.

### H2: Current EMA state improves prediction beyond static traits and past use

**Result: see the D-minus-C interval above.** Adding current raw EMA variables changed AUROC by +0.024.

**Situation example:** A person with the same past drinking history may have a different current craving or stress response. The model uses that current response to assign different risk scores.

### H3: Personal-baseline change variables add information beyond raw state

**Result: see the G-minus-E interval above.** The estimated AUROC change was +0.002. If its interval contains zero, this file does not show clear added discrimination. This is not proof that change variables are useless; it means the added value is uncertain in this sample.

**Situation example:** Two people may both report stress=8, but one person normally reports stress=8 and the other normally reports stress=2. A change feature tries to distinguish those situations.

### H4: Longer lags reduce forecasting performance

**Result: this report estimates one lag only.** Compare the same row of each lag-specific report rather than comparing this file alone. Differences across lag files are exploratory and may also reflect different row counts and positive rates.

**Situation example:** A signal recorded now may help predict drinking tomorrow but be less useful for predicting drinking two weeks later.

## 4. Honest interpretation

The model's pooled AUROC was 0.891, while the average within-person AUROC was 0.556 among 84 eligible participants. These are different questions: pooled AUROC can benefit from stable differences between people, while within-person AUROC asks whether high-risk hours are ranked above safer hours for the same person. Results do not establish causality, clinical relapse prediction, or generalization beyond this sample.

## 5. Main limitations

- Positive hourly windows overlap, so they are not independent drinking episodes.
- Participant grouping prevents person-level leakage, but does not remove label persistence.
- The analysis uses one general model; it does not test personalized random-effects or partial-pooling models.
- Feature names identify raw, change, aggregate, current, and lag categories, but do not by themselves prove the exact time endpoint or absence of all leakage.
