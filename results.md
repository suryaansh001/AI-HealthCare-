# Demographic lapse-analysis results

## Scope

This report summarizes the descriptive and statistical tests run on `dataset/features_0lag_v3.csv`.

The outcome analyzed was the binary `lapse` label:

- `yes` = lapse window
- `no` = non-lapse window

This analysis compares lapse-label rates across education, sex, employment, and marital-status categories.

## Dataset used

| Statistic | Value |
|---|---:|
| Total labelled windows | 274,179 |
| Participants | 151 |
| Lapse windows | 21,107 |
| Overall lapse-window rate | 7.70% |

Each participant contributes repeated time windows. Therefore, both row-level and participant-level summaries are reported.

## Test methods

### Row-level comparison

For each demographic variable, lapse-window rates were calculated over all rows. A Pearson chi-square test evaluated whether the demographic grouping and lapse label were associated.

These results show how the categories compare across the recorded windows.

### Participant-level comparison

Each participant was counted once and classified as:

- `ever_lapse = 1` if they had at least one lapse window
- `ever_lapse = 0` otherwise

A Pearson chi-square test was then applied to the participant-level contingency table. This gives a second comparison where each participant is counted only once.

## Results by demographic variable

### Education

#### Row-level lapse-window rates

| Education group | Participants | Windows | Lapse windows | Lapse rate |
|---|---:|---:|---:|---:|
| College or more | 81 | 146,981 | 11,772 | 8.01% |
| Some college | 55 | 100,535 | 7,088 | 7.05% |
| High school or less | 15 | 26,663 | 2,247 | 8.43% |

Row-level chi-square test:

- Chi-square = 99.356
- Degrees of freedom = 2
- p-value = `2.66 × 10^-22`

#### Participant-level ever-lapse rates

| Education group | Participants | Ever had lapse | Percentage |
|---|---:|---:|---:|
| College or more | 81 | 47 | 58.0% |
| Some college | 55 | 28 | 50.9% |
| High school or less | 15 | 9 | 60.0% |

Participant-level chi-square test:

- p-value = `0.670`

#### Interpretation

The education groups have similar lapse rates. The high-school-or-less group contains 15 participants.

### Sex

#### Row-level lapse-window rates

| Sex | Participants | Windows | Lapse windows | Lapse rate |
|---|---:|---:|---:|---:|
| Female | 74 | 133,091 | 11,876 | 8.92% |
| Male | 77 | 141,088 | 9,231 | 6.54% |

Row-level chi-square test:

- Chi-square = 545.845
- Degrees of freedom = 1
- p-value = `1.01 × 10^-120`

#### Participant-level ever-lapse rates

| Sex | Participants | Ever had lapse | Percentage |
|---|---:|---:|---:|
| Female | 74 | 48 | 64.9% |
| Male | 77 | 36 | 46.8% |

Participant-level chi-square test:

- p-value = `0.038`

#### Interpretation

Female participants and female-labelled windows have higher recorded lapse rates than male participants and male-labelled windows.

### Employment

#### Row-level lapse-window rates

| Employment group | Participants | Windows | Lapse windows | Lapse rate |
|---|---:|---:|---:|---:|
| Employed | 98 | 178,737 | 14,081 | 7.88% |
| Unemployed | 53 | 95,442 | 7,026 | 7.36% |

Row-level chi-square test:

- Chi-square = 23.288
- Degrees of freedom = 1
- p-value = `1.39 × 10^-6`

#### Participant-level ever-lapse rates

| Employment group | Participants | Ever had lapse | Percentage |
|---|---:|---:|---:|
| Employed | 98 | 52 | 53.1% |
| Unemployed | 53 | 32 | 60.4% |

Participant-level chi-square test:

- p-value = `0.489`

#### Interpretation

The employed and unemployed groups have very similar recorded lapse rates.

### Marital status

#### Row-level lapse-window rates

| Marital-status group | Participants | Windows | Lapse windows | Lapse rate |
|---|---:|---:|---:|---:|
| Never Married | 67 | 116,918 | 11,725 | 10.03% |
| Married | 32 | 59,698 | 4,130 | 6.92% |
| Other | 52 | 97,563 | 5,252 | 5.38% |

Row-level chi-square test:

- Chi-square = 1,680.408
- Degrees of freedom = 2
- p-value `< 0.001` (reported by the statistical software as 0 due to numerical rounding)

#### Participant-level ever-lapse rates

| Marital-status group | Participants | Ever had lapse | Percentage |
|---|---:|---:|---:|
| Never Married | 67 | 43 | 64.2% |
| Married | 32 | 14 | 43.8% |
| Other | 52 | 27 | 51.9% |

Participant-level chi-square test:

- p-value = `0.129`

#### Interpretation

The never-married group has the highest recorded lapse rate and the highest participant-level ever-lapse percentage.

## Overall summary

| Variable | Main observed pattern | Participant-level test |
|---|---|---|
| Education | Rates are similar across groups | Not conclusive, p = 0.670 |
| Sex | Female rate higher than male rate | Significant unadjusted association, p = 0.038 |
| Employment | Rates are very similar | Not conclusive, p = 0.489 |
| Marital status | Never-married rate is highest | Not conclusive, p = 0.129 |

Overall, female and never-married groups have the highest recorded lapse rates in these comparisons. Education and employment groups have more similar rates.

## Reading the results

- Row-level rates compare all recorded windows in each category.
- Participant-level rates count each participant once.
- The p-values show how different the category results are in each comparison.
- The results can be used to compare patterns across the groups in this dataset.
