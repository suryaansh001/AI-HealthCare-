# Raw EMA exploratory data analysis

## Scope

This report describes `ema.csv`. It is a raw repeated-EMA file, not an hourly labelled prediction file, so no AUROC, AUPRC, or category ablation was calculated from it alone.

## Basic EDA

| Quantity | Result |
|---|---:|
| EMA rows | 45,460 |
| Participants | 169 |
| Drinking reports (`ema_1 = Yes`) | 1,437 |
| Participants with at least one drinking report | 104 |
| Earliest observation | 2017-03-02 15:20:19 UTC |
| Latest observation | 2020-04-24 17:27:33 UTC |
| EMA types | 34,125 later; 11,335 morning |

## Variable checks

| Variables | Observed values and missingness |
|---|---|
| `ema_1` | `No` = 44,023; `Yes` = 1,437; no missing values |
| `ema_1_5` | `Uncertain` = 594; `Yes` = 735; `No` = 45; about 97% missing because it is a drinking follow-up field |
| `ema_2` to `ema_5` | Values from 0 to 12; about 1.6% missing |
| `ema_6` to `ema_7` | Values from 1 to 11; about 1.6% missing |
| `ema_8` to `ema_10` | Values from 1 to 11; present in morning EMAs and missing in later EMAs |
| `ema_1_6` | Mostly `America/Chicago`, with a small number of other US time zones |

## Hypotheses

### H1: Drinking reports are uncommon

**Result: supported descriptively.** Only 1,437 of 45,460 raw EMA rows report drinking. These reports come from 104 of 169 participants.

**Situation example:** Most surveys were completed during non-drinking periods, while a smaller number recorded drinking episodes.

### H2: Morning-only questions have a different missingness pattern

**Result: supported.** Items 8–10 are present in morning surveys and absent from later surveys. This looks like planned survey design rather than random missingness.

**Situation example:** A morning question about anticipated risk may be answered once in the morning and not repeated in an afternoon or evening survey.

### H3: The exact meanings of items 2–10 can be recovered from this CSV

**Result: not supported.** Ranges and correlations can describe the variables, but they cannot prove whether an item measures craving, stress, risk, valence, arousal, or self-efficacy. The questionnaire codebook or survey instrument is needed.

## Important limitations

- Raw drinking reports are self-reported.
- The file contains irregular observations and two participants with approximately 570-day spans; the cause is not documented in this file.
- A drinking report is not automatically equivalent to a distinct clinical relapse episode.
- Predictive modelling should use the engineered feature files and must verify that every feature uses only information available before prediction time.