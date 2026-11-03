# Lag 0 hours: exploratory lapse-forecasting results

## Analysis scope

This report analyzes one engineered Curtin-lab feature file. The outcome is a goal-inconsistent drinking window, not clinical relapse. Rows were sampled at every sixth hourly row, participants were kept together in five-fold stratified group cross-validation, and uncertainty intervals were bootstrapped over participants. These are exploratory results.

## 1. Basic EDA

| Quantity | Result |
|---|---:|
| Labelled windows before thinning | 274,179 |
| Participants | 151 |
| Positive windows | 21,107 |
| Positive-window rate | 7.70% |
| Date range | 2017-03-03 06:00:00+00:00 to 2019-12-04 00:00:00+00:00 |
| Rows used in modelling after stride 6 | 45,749 |
| Participants with at least one positive window | 84 |
| Participants with no positive window | 67 |
| Median rows per participant before thinning | 2010 |
| Positive windows per participant: median / mean / maximum | 24 / 139.8 / 1193 |
| Consecutive gaps exactly 1 hour | 99.56% |
| Largest consecutive gap | 261 hours |

Feature missingness was summarized by category. The category labels come from the column names; no item meanings were inferred from correlations.

| Feature category | Number of columns | Mean missingness |
|---|---:|---:|
| static | 8 | 0.00% |
| past_use | 10 | 0.00% |
| compliance | 10 | 0.00% |
| current_raw | 9 | 0.05% |
| current_change | 9 | 0.05% |
| aggregate_raw | 117 | 4.53% |
| aggregate_change | 117 | 4.53% |

Rows in each participant's first 24, 72, and 168 hours were 1.30%, 3.64%, and 8.40%, respectively.

## 2. Category-wise predictive results

AUPRC should be compared with the positive rate above, while AUROC should be compared with 0.50. The within-person metric averages AUROC only for participants with at least three positive and three negative rows after thinning.

| Feature set | Features | AUROC | AUPRC | Within-person AUROC | Eligible people | Brier |
|---|---:|---:|---:|---:|---:|---:|
| A static | 21 | 0.499 | 0.072 | 0.548 | 84 | 0.077 |
| B past use | 10 | 0.888 | 0.624 | 0.553 | 84 | 0.043 |
| C static + past use | 31 | 0.886 | 0.622 | 0.581 | 84 | 0.043 |
| D + current raw | 40 | 0.908 | 0.651 | 0.636 | 84 | 0.042 |
| E + raw state | 157 | 0.909 | 0.647 | 0.628 | 84 | 0.042 |
| F change only | 157 | 0.905 | 0.654 | 0.609 | 84 | 0.041 |
| G full + change | 283 | 0.908 | 0.658 | 0.623 | 84 | 0.041 |
| H all + compliance | 293 | 0.909 | 0.657 | 0.618 | 84 | 0.041 |

### Paired differences with 95% participant-bootstrap intervals

| Contrast | AUROC difference | AUPRC difference | Within-person difference |
|---|---:|---:|---:|
| D minus C | +0.022 [+0.013, +0.035] | +0.029 [+0.012, +0.049] | +0.055 [+0.023, +0.086] |
| E minus D | +0.001 [-0.004, +0.005] | -0.004 [-0.018, +0.007] | -0.007 [-0.030, +0.018] |
| G minus E | -0.001 [-0.004, +0.003] | +0.011 [-0.003, +0.026] | -0.006 [-0.029, +0.020] |
| F minus E | -0.004 [-0.012, +0.003] | +0.007 [-0.016, +0.029] | -0.019 [-0.049, +0.007] |

## 3. Hypotheses and plain-language results

### H1: Past drinking history is useful for prediction

**Result: supported, with an important caution.** Past-use-only AUROC was 0.888, compared with 0.50 for chance. However, the source data contain overlapping outcome windows, so past-use performance may partly reflect persistence from a recent lapse rather than genuinely early warning.

**Situation example:** If someone has recently reported drinking, many nearby hourly windows are also labelled positive. A model can score those windows correctly using recent drinking history, even if it is not predicting a completely new episode far in advance.

### H2: Current EMA state improves prediction beyond static traits and past use

**Result: see the D-minus-C interval above.** Adding current raw EMA variables changed AUROC by +0.022.

**Situation example:** A person with the same past drinking history may have a different current craving or stress response. The model uses that current response to assign different risk scores.

### H3: Personal-baseline change variables add information beyond raw state

**Result: see the G-minus-E interval above.** The estimated AUROC change was -0.001. If its interval contains zero, this file does not show clear added discrimination. This is not proof that change variables are useless; it means the added value is uncertain in this sample.

**Situation example:** Two people may both report stress=8, but one person normally reports stress=8 and the other normally reports stress=2. A change feature tries to distinguish those situations.

### H4: Longer lags reduce forecasting performance

**Result: this report estimates one lag only.** Compare the same row of each lag-specific report rather than comparing this file alone. Differences across lag files are exploratory and may also reflect different row counts and positive rates.

**Situation example:** A signal recorded now may help predict drinking tomorrow but be less useful for predicting drinking two weeks later.

## 4. Honest interpretation

The model's pooled AUROC was 0.908, while the average within-person AUROC was 0.623 among 84 eligible participants. These are different questions: pooled AUROC can benefit from stable differences between people, while within-person AUROC asks whether high-risk hours are ranked above safer hours for the same person. Results do not establish causality, clinical relapse prediction, or generalization beyond this sample.

## 5. Main limitations

- Positive hourly windows overlap, so they are not independent drinking episodes.
- Participant grouping prevents person-level leakage, but does not remove label persistence.
- The analysis uses one general model; it does not test personalized random-effects or partial-pooling models.
- Feature names identify raw, change, aggregate, current, and lag categories, but do not by themselves prove the exact time endpoint or absence of all leakage.
