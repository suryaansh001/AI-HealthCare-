# Anchor Target Provenance Investigation

## Executive finding

The claim that `lapse` is a rolling 168-hour future outcome derived directly from `ema_1` is **not supported by the executable label-generation code located in the original study's public GitHub materials**. The upstream script `mak_lapse_labels.qmd` constructs labels for individual study days. It marks a day positive when a retained lapse record's `lapse_start` falls in that day, whose nominal boundaries are 04:00 local time through 03:59:59 the next day. It contains no 168-hour rolling outcome, no future-window shift, and no lag parameter.

The supplied feature matrices corroborate a daily label structure: in the 0-hour matrix, all rows within each participant's Chicago-time 04:00-to-04:00 bin have one label (12,586 bins checked, zero conflicting labels); for shared `(subid, dttm_label)` keys, adjacent lag matrices have identical `lapse` values. This does **not** by itself prove that every local matrix was exported from this exact upstream script. The upstream script reads an intermediate `lapses_day.csv` that is not present locally, and its messages-specific participant filters differ from the supplied matrices.

Accordingly:

- **168-hour future target:** contradicted by the located executable label-generation logic; not the `lapse` label this script creates.
- **24-hour daily label:** verified as the upstream script's target construction.
- **`ema_1` as the exact raw event source / semantics:** not verified. The script reads a curated `lapses_day.csv`, not `ema.csv` or `ema_1`; its event-table construction is unavailable.
- **Lag 24 as a prospective forecast:** not established by target provenance. The target label remains the same on shared timestamps across the lag matrices; exact feature availability and prediction-time alignment require separate verification.

## Scope, integrity, and method

This was a read-only investigation. No branch switch, commit, training, serialization, package installation, dataset edit, source edit, or overwrite was performed. Python checks were in-memory only.

At the start of the investigation:

```text
git rev-parse --abbrev-ref HEAD
anchor-research-correction

git rev-parse HEAD
1daa744959e02b94a3914e7ae7b30415aa8a0053

git status --short
?? research/audit/ANCHOR_MASTER_REPORT_CRITICAL_REVIEW.md
```

The existing untracked critical-review report was acknowledged and preserved. Its starting SHA-256 was `7B53072BA001ED32F786055BE649CAFF31D63CB0A29BA1E5DBB81C6BE1E86771`. The requested destination was absent before this report was created.

Searches included the tracked source tree and reachable Git history for target-generation code, plus the public study materials referenced by the project. Local commands included:

```powershell
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git status --short
git --no-pager grep -n -I -E 'ema_1|lapse|shift\(|lead\(|rolling|cutoff|censor|future' HEAD -- '*.py' '*.R' '*.ipynb' '*.do' '*.sas'
```

The local Python target-analysis and dataset-audit scripts consume an existing `lapse` column and derive/count a binary `y`; neither creates `lapse`. In particular, `research/preprocessing/target_temporal_analysis.py` converts the pre-existing column at lines 25–27 and then analyzes runs/autocorrelation. It is not evidence of target construction.

## Primary executable source located

The source is the public, executable Quarto/R document:

- [`mak_lapse_labels.qmd`](https://github.com/jjcurtin/study_messages/blob/1c09e66fd05bcf9b448b677c59a64b6ba073906e/_mak/messages_archive/mak_lapse_labels.qmd), current `main` tree; Git blob SHA `02174217c3e9042ac1554aa605b08104cfa73ebb`.
- It reads `study_dates.csv` and `lapses_day.csv` (lines 94–107), creates daily labels (lines 113–119), prepares event timestamps (lines 124–142), assigns/removes labels (lines 147–181), and writes `lapses.csv` (lines 232–238).

The code is in the public repository, not in this local checkout's tracked source tree. The script's source files `study_dates.csv` and `lapses_day.csv` are not present in the local repository, so the exact row-by-row lineage from those inputs to the supplied matrices cannot be reproduced here.

### What the executable does

1. **Daily label grid.** For each row in `study_dates.csv`, it creates a sequence from `start_study + 1 day` through `end_study - 1 day`, then sets the hour to 04 and minute to 00 (`mak_lapse_labels.qmd`, lines 113–119). The code does not explicitly reset seconds, so any seconds inherited from the sequence timestamp remain.
2. **Event time.** It reads `lapse_start_date`, `lapse_start_time`, `lapse_end_date`, `lapse_end_time`, `exclude`, and `ema_1_6` from `lapses_day.csv` (lines 100–106). A missing start time is replaced with `20:00`; a missing end date is set to the day after the start date; and a missing end time is replaced with `02:00` (lines 124–130). The script parses start/end date-time values and applies the row's `ema_1_6` timezone before converting them to `America/Chicago` (lines 131–136).
3. **Positive label.** A row with `exclude == "FALSE"` and nonmissing `lapse_start` labels the participant-day `"lapse"` if the start time is inside that day (`lines 151–161`). The event is assigned by its start time only; the valid event's end time is not used to extend a positive label.
4. **Boundary.** `day_end` is `day_start + 1 day - 1 second`; both comparisons use `<=`/`>=` (`lines 147–149`, `157–160`). Thus the coded interval is inclusive from the generated nominal 04:00 start through the next day's 03:59:59, and excludes the next 04:00 boundary.
5. **Excluded event.** Rows with `exclude == "TRUE"` and nonmissing start time mark the day containing the event start and the day containing its end as `"exclude"`; those days are subsequently removed (`lines 163–181`). This code marks the endpoint days, not every intervening day of a multi-day event.
6. **Negative label.** Any remaining `NA` label becomes `"no lapse"` (`lines 179–181`). There is no explicit unknown outcome for a day with no retained event record.
7. **Output.** The script writes the resulting participant-day table to `lapses.csv` (`lines 232–238`).

The note at line 92 says missing start time should for now be `NA`, but the executable code at line 129 instead imputes missing start time to `20:00`. The executed transformation therefore takes precedence over that comment. If the start date itself is missing, the parsed start remains missing; such a row is filtered out of both the valid and excluded-event loops (lines 151–165), leaving the day eligible to default to `"no lapse"` if no other event assigns a label. Similarly, an `exclude` value other than the exact strings `"FALSE"` or `"TRUE"` is not handled by either loop.

### Participant selection and event-definition limit

The companion [`mak_study_dates.qmd`](https://github.com/jjcurtin/study_messages/blob/1c09e66fd05bcf9b448b677c59a64b6ba073906e/_mak/messages_archive/mak_study_dates.qmd) filters out participants without `followup_1` (lines 128–134) and excludes IDs 104, 269, and 204 for lapse-reporting issues (lines 136–141). The messages-specific version also removes participants for message-data reasons (lines 143–147).

The label script does **not** define what evidence qualifies an entry in `lapses_day.csv` as a drinking lapse, nor show how that intermediate table was built. It does not read raw `ema.csv`, inspect `ema_1`, or turn `ema_1 == "Yes"` into a target. `ema_1_6` is used as a timezone field when parsing event times; that use does not establish that the event itself came from `ema_1`. The notes mention manual editing of a start time (line 56) and discuss lapses reported in interviews or absent from EMA records (lines 42–54). The exact raw event rule and the extent to which labels use EMA versus interview/staff information therefore remain **NOT VERIFIED**.

## Comparison with local artifacts

### Raw EMA file

`dataset/ema.csv` contains 45,460 records from 169 participants. Its `ema_1` values are `No` (44,023) and `Yes` (1,437), with no missing values. `dataset/completedatasetwithALLcsv/ema_results.md` reports these counts and cautions that raw self-reports do not necessarily identify a distinct relapse episode (lines 23 and 50–52). The raw CSV alone does not provide an instrument/codebook sufficient to verify the meaning of the auxiliary items or the relationship between each affirmative report and a lapse event in `lapses_day.csv`.

The local report `research/audit/ANCHOR_STAGE_2_MASTER_AUDIT.md` states that `ema_1` is “Have you consumed any alcohol since the last prompt?” and describes a rolling future 168-hour target (lines 137–154). Those statements are not implemented by the local target-analysis script and conflict with the upstream executable described above. The local `preregistration.html` has “next week alcohol use” in its title, but a study title or preregistration narrative is not target-generation code and does not override the generated label logic.

### Feature matrices

The five supplied files are:

| Matrix | Rows | Positive `lapse == yes` rows | Participants |
|---|---:|---:|---:|
| `features_0lag_v3.csv` | 274,179 | 21,107 | 151 |
| `features_24lag_v3.csv` | 270,911 | 20,800 | 151 |
| `features_72lag_v3.csv` | 264,362 | 20,182 | 151 |
| `features_168lag_v3.csv` | 251,458 | 19,194 | 151 |
| `features_336lag_v3.csv` | 228,420 | 17,474 | 151 |

These are in `dataset/completedatasetwithALLcsv/`. The count differences reflect rows present in one lag matrix but not another; they do not establish a different target window. On exact `(subid, dttm_label)` intersections, the `lapse` column agrees with the next-lag matrix for all shared rows:

| Adjacent matrices | Shared rows | `lapse` disagreements |
|---|---:|---:|
| 0h / 24h | 270,911 | 0 |
| 24h / 72h | 264,362 | 0 |
| 72h / 168h | 251,458 | 0 |
| 168h / 336h | 228,420 | 0 |

For the 0-hour matrix, grouping `dttm_label` into Chicago-time bins offset at 04:00 yielded 12,586 participant-day bins and zero bins with multiple `lapse` values. This is consistent with the upstream daily labels being repeated across multiple feature rows. It is not an independent reconstruction of `lapses.csv`.

The local matrix has 151 participants versus 169 in the raw EMA file. It omits 18 raw-file participant IDs, including 104, 204, and 269; however, the local feature matrices retain IDs that the messages-specific `mak_study_dates.qmd` excludes for message availability. Thus the local cohort is not an exact match to the messages-specific analysis cohort. The overlap in the three lapse-quality exclusions and the daily label structure support a shared study lineage, but do not prove that the local matrices are a direct export of this precise messages-specific `lapses.csv`.

### Reproducible calculations used

The matrix comparison used only the existing CSVs and in-memory pandas operations equivalent to:

```python
for left, right in zip(lag_frames, lag_frames[1:]):
    shared = left.merge(right, on=["subid", "dttm_label"],
                        suffixes=("_left", "_right"))
    disagreements = (shared["lapse_left"] != shared["lapse_right"]).sum()

rows["local_time"] = pd.to_datetime(rows["dttm_label"], utc=True).dt.tz_convert(
    "America/Chicago"
)
rows["day04"] = (rows["local_time"] - pd.Timedelta(hours=4)).dt.date
daily_label_count = rows.groupby(["subid", "day04"])["lapse"].nunique()
```

Results were 0 disagreements on every adjacent shared-key merge and 0 multi-label Chicago 04:00 participant-days in the 0-hour matrix. No target was regenerated or inferred from the raw EMA file.

## Target-provenance contract

| Question | Finding |
|---|---|
| What event is positive? | **Partially verified:** a retained (`exclude == "FALSE"`) row in `lapses_day.csv` with a parsed nonmissing `lapse_start` inside the participant-day. The substantive event definition and its derivation from raw EMA are not in the located code. |
| Which timestamp is used? | The event's reconstructed `lapse_start` from `lapse_start_date` and `lapse_start_time`; missing time is imputed to 20:00. The time is interpreted using `ema_1_6` and converted to America/Chicago. |
| Is the outcome a future 168-hour window? | **No.** The located generator creates daily labels; no rolling 168-hour interval is constructed. |
| Is there a lag shift in label construction? | **No** in the located generator. The local matrices also retain identical labels on shared `(subid, dttm_label)` keys across lag variants. |
| Are interval boundaries inclusive? | Yes at both coded comparisons: nominal day start through `day_start + 1 day - 1 second`. The next 04:00 boundary is excluded. Seconds are not explicitly reset when `day_start` is created. |
| How are missing event records handled? | Missing start time is filled as 20:00, despite a contradictory comment. A missing/invalid start date can yield a missing parsed start, causing the row to bypass both loops; absent another event, its day defaults to `"no lapse"`. There is no general unknown-label state. |
| How is incomplete future follow-up handled? | No future-window follow-up check exists because the generator has no future outcome window. Missing event ascertainment on an otherwise generated day may become `"no lapse"`; its consequences cannot be quantified without `lapses_day.csv` and its source process. |
| How is participant follow-up handled? | Participants without follow-up 1 are removed before label generation. The daily grid is bounded by the supplied study start/end dates. There is no 168-hour right-censoring or endpoint purge. |
| Can this prove Lag 24 prospective warning? | **No.** A daily target with unchanged labels across lag matrices does not establish the exact feature cutoff or a prospective lead time. Those require a separate, verified feature-time contract. |

## Final assessment and unresolved evidence

The prior 168-hour/`ema_1` narrative is **not a valid description of the located executable label-generation logic**. That code labels a 24-hour study day according to the onset time of an event already present in an intermediate event table. Because local target labels remain unchanged at shared timestamps across the lag matrices, the word “lag” should not be interpreted as shifting this outcome into a future 7-day window.

The following items remain unresolved:

1. The authoritative questionnaire wording and event-level definition for `lapses_day.csv`.
2. The executable or manual process that creates `lapses_day.csv` from EMA, interviews, or other sources.
3. Whether the exact archived messages-specific generator version produced the local 151-participant matrices; the intermediate files and exact cohort do not match locally.
4. The intended mapping from each local `dttm_label`/feature row to a generated 04:00 participant-day label and to the latest feature-availability timestamp.
5. Whether missed EMA/event reports create systematic negative labels, and how the event pipeline handles incomplete observation beyond the explicit participant exclusions and default `"no lapse"` behavior.

Until those are resolved, do not describe the supplied `lapse` target as a verified rolling 168-hour future drinking outcome or describe Lag 24 as a validated prospective warning horizon.
