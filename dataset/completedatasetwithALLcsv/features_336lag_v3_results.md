# Lag 336 hours: exploratory lapse-forecasting results

## Analysis scope

This report analyzes one engineered Curtin-lab feature file. The outcome is a goal-inconsistent drinking window, not clinical relapse. Rows were sampled at every sixth hourly row, participants were kept together in five-fold stratified group cross-validation, and uncertainty intervals were bootstrapped over participants. These are exploratory results.

## 1. Basic EDA

| Quantity | Result |
|---|---:|
| Labelled windows before thinning | 228,420 |
| Participants | 151 |
| Positive windows | 17,474 |
| Positive-window rate | 7.65% |
| Date range | 2017-03-17 05:00:00+00:00 to 2019-12-04 00:00:00+00:00 |
| Rows used in modelling after stride 6 | 38,123 |
| Participants with at least one positive window | 79 |
| Participants with no positive window | 72 |
| Median rows per participant before thinning | 1696 |
| Positive windows per participant: median / mean / maximum | 24 / 115.7 / 1071 |
| Consecutive gaps exactly 1 hour | 99.56% |
| Largest consecutive gap | 261 hours |

Feature missingness was summarized by category. The category labels come from the column names; no item meanings were inferred from correlations.

| Feature category | Number of columns | Mean missingness |
|---|---:|---:|
| static | 8 | 0.00% |
| past_use | 10 | 0.00% |
| compliance | 10 | 0.00% |
| current_raw | 9 | 0.06% |
| current_change | 9 | 0.06% |
| aggregate_raw | 117 | 3.97% |
| aggregate_change | 117 | 3.97% |

Rows in each participant's first 24, 72, and 168 hours were 1.55%, 4.36%, and 10.08%, respectively.

## 2. Category-wise predictive results

AUPRC should be compared with the positive rate above, while AUROC should be compared with 0.50. The within-person metric averages AUROC only for participants with at least three positive and three negative rows after thinning.

| Feature set | Features | AUROC | AUPRC | Within-person AUROC | Eligible people | Brier |
|---|---:|---:|---:|---:|---:|---:|
| A static | 21 | 0.550 | 0.080 | 0.518 | 79 | 0.076 |
| B past use | 10 | 0.820 | 0.303 | 0.450 | 79 | 0.061 |
| C static + past use | 31 | 0.806 | 0.275 | 0.485 | 79 | 0.063 |
| D + current raw | 40 | 0.838 | 0.310 | 0.501 | 79 | 0.061 |
| E + raw state | 157 | 0.842 | 0.288 | 0.485 | 79 | 0.061 |
| F change only | 157 | 0.823 | 0.230 | 0.452 | 79 | 0.065 |
| G full + change | 283 | 0.828 | 0.242 | 0.451 | 79 | 0.064 |
| H all + compliance | 293 | 0.834 | 0.244 | 0.453 | 79 | 0.064 |

### Paired differences with 95% participant-bootstrap intervals

| Contrast | AUROC difference | AUPRC difference | Within-person difference |
|---|---:|---:|---:|
| D minus C | +0.033 [+0.019, +0.048] | +0.035 [-0.013, +0.074] | +0.016 [-0.030, +0.059] |
| E minus D | +0.004 [-0.007, +0.016] | -0.022 [-0.075, +0.028] | -0.017 [-0.051, +0.013] |
| G minus E | -0.014 [-0.026, -0.002] | -0.047 [-0.089, -0.010] | -0.033 [-0.067, +0.005] |
| F minus E | -0.019 [-0.035, -0.002] | -0.058 [-0.109, -0.017] | -0.032 [-0.068, +0.004] |

## 3. Hypotheses and plain-language results

### H1: Past drinking history is useful for prediction

**Result: supported, with an important caution.** Past-use-only AUROC was 0.820, compared with 0.50 for chance. However, the source data contain overlapping outcome windows, so past-use performance may partly reflect persistence from a recent lapse rather than genuinely early warning.

**Situation example:** If someone has recently reported drinking, many nearby hourly windows are also labelled positive. A model can score those windows correctly using recent drinking history, even if it is not predicting a completely new episode far in advance.

### H2: Current EMA state improves prediction beyond static traits and past use

**Result: see the D-minus-C interval above.** Adding current raw EMA variables changed AUROC by +0.033.

**Situation example:** A person with the same past drinking history may have a different current craving or stress response. The model uses that current response to assign different risk scores.

### H3: Personal-baseline change variables add information beyond raw state

**Result: see the G-minus-E interval above.** The estimated AUROC change was -0.014. If its interval contains zero, this file does not show clear added discrimination. This is not proof that change variables are useless; it means the added value is uncertain in this sample.

**Situation example:** Two people may both report stress=8, but one person normally reports stress=8 and the other normally reports stress=2. A change feature tries to distinguish those situations.

### H4: Longer lags reduce forecasting performance

**Result: this report estimates one lag only.** Compare the same row of each lag-specific report rather than comparing this file alone. Differences across lag files are exploratory and may also reflect different row counts and positive rates.

**Situation example:** A signal recorded now may help predict drinking tomorrow but be less useful for predicting drinking two weeks later.

## 4. Honest interpretation

The model's pooled AUROC was 0.828, while the average within-person AUROC was 0.451 among 79 eligible participants. These are different questions: pooled AUROC can benefit from stable differences between people, while within-person AUROC asks whether high-risk hours are ranked above safer hours for the same person. Results do not establish causality, clinical relapse prediction, or generalization beyond this sample.

## 5. Main limitations

- Positive hourly windows overlap, so they are not independent drinking episodes.
- Participant grouping prevents person-level leakage, but does not remove label persistence.
- The analysis uses one general model; it does not test personalized random-effects or partial-pooling models.
- Feature names identify raw, change, aggregate, current, and lag categories, but do not by themselves prove the exact time endpoint or absence of all leakage.
