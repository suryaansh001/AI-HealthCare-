"""
audit_features.py
Audit a Curtin-lab lag feature file (e.g. features_0lag_v3.csv) and prepare it for modeling.

Usage:
    python3 audit_features.py features_0lag_v3.csv

Outputs (written next to the CSV):
    column_dictionary.csv   every column parsed into item / period / lag / raw-vs-change / statistic / family
    <name>.parquet          fast, compact copy of the data (if pyarrow is installed)
and prints everything you need to paste back: rows, participants, lapse rate, spacing,
missingness by feature family, early-history rows, constant columns.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PATH = Path(sys.argv[1] if len(sys.argv) > 1 else "dataset/features_0lag_v3.csv")
META = {"label_num", "subid", "dttm_label", "lapse"}

PATTERN = re.compile(
    r"^(?:ema_(?P<item>\d+)\.)?p(?P<period>\d+)\.l(?P<lag>\d+)\."
    r"(?P<kind>[rd])(?P<stat>ratecount|recent|min|max|median)"
    r"(?:_response|\.count\.(?P<target>lapse|ema))$"
)


def parse(col):
    """Split one feature name into its parts and assign it to a feature family."""
    if col in META:
        return dict(column=col, family="meta")
    if col.startswith("demo_") or col == "label_day":
        return dict(column=col, family="static")
    m = PATTERN.match(col)
    if not m:
        return dict(column=col, family="UNPARSED")
    d = m.groupdict()
    kind = "change" if d["kind"] == "d" else "raw"
    if d["stat"] == "ratecount":
        fam = "past_use" if d["target"] == "lapse" else "compliance"
    elif d["stat"] == "recent":
        fam = f"current_{kind}"
    else:
        fam = f"aggregate_{kind}"
    return dict(column=col, family=fam, item=d["item"], period_h=int(d["period"]),
                lag_h=int(d["lag"]), kind=kind, stat=d["stat"], target=d["target"])


# ---------------------------------------------------------------- 1. column dictionary
cols = pd.read_csv(PATH, nrows=0).columns.tolist()
cdict = pd.DataFrame([parse(c) for c in cols])
cdict.to_csv(PATH.with_name("column_dictionary.csv"), index=False)
print(f"\n== COLUMNS: {len(cols)} ==")
print(cdict["family"].value_counts().to_string())
if (cdict["family"] == "UNPARSED").any():
    print("\nUNPARSED columns (check these):")
    print(cdict.loc[cdict["family"] == "UNPARSED", "column"].tolist())

# ---------------------------------------------------------------- 2. load
numeric = {c: "float32" for c in cdict.loc[~cdict["family"].isin(["meta", "static"]), "column"]}
df = pd.read_csv(PATH, dtype=numeric, parse_dates=["dttm_label"]).copy()  # .copy() defragments the wide frame
df["y"] = df["lapse"].astype(str).str.strip().str.lower().isin(["yes", "1", "true"]).astype("int8")

print("\n== SIZE ==")
print("rows:", len(df), "| participants:", df["subid"].nunique())
print("lapse label values:", df["lapse"].value_counts().to_dict())
print("lapse prevalence: %.2f%%" % (100 * df["y"].mean()))
print("date range:", df["dttm_label"].min(), "->", df["dttm_label"].max())

# ---------------------------------------------------------------- 3. per participant
g = df.groupby("subid")
per = pd.DataFrame({"rows": g.size(), "lapse_rows": g["y"].sum(), "lapse_rate": g["y"].mean()})
print("\n== PER PARTICIPANT ==")
print(per.describe().round(3).to_string())
print("participants with >=1 lapse window:", int((per["lapse_rows"] > 0).sum()))
print("participants with 0 lapse windows:", int((per["lapse_rows"] == 0).sum()))

# ---------------------------------------------------------------- 4. time spacing
df = df.sort_values(["subid", "dttm_label"]).reset_index(drop=True)
gap_h = df.groupby("subid")["dttm_label"].diff().dt.total_seconds().div(3600)
print("\n== SPACING between consecutive rows (hours) ==")
print(gap_h.describe().round(2).to_string())
print("share of gaps equal to 1h: %.3f" % (gap_h.eq(1).sum() / gap_h.notna().sum()))

# ---------------------------------------------------------------- 5. early-history rows
hrs = (df["dttm_label"] - df.groupby("subid")["dttm_label"].transform("min")).dt.total_seconds() / 3600
df["hours_since_start"] = hrs.astype("float32")
print("\n== EARLY HISTORY (baseline for 'change' features is thin) ==")
for h in (24, 72, 168):
    print(f"rows within first {h:>3}h of a participant: {(hrs < h).mean():.3f}")

# ---------------------------------------------------------------- 6. missingness
print("\n== MISSINGNESS by family (mean % NA) ==")
feat_cols = cdict.loc[~cdict["family"].isin(["meta"]), "column"]
na = df[feat_cols].isna().mean()
print((100 * na.groupby(cdict.set_index("column").loc[feat_cols, "family"]).mean()).round(2).to_string())
print("columns with >20% NA:", int((na > 0.20).sum()))

num = df[feat_cols].select_dtypes("number")
const = num.columns[num.nunique(dropna=True) <= 1].tolist()
print("constant / near-constant numeric columns:", len(const), const[:10])

# ---------------------------------------------------------------- 7. save fast copy
try:
    out = PATH.with_suffix(".parquet")
    df.to_parquet(out, index=False)
    print("\nSaved:", out)
except Exception as e:  # pyarrow missing
    print("\nParquet skipped (pip install pyarrow). Reason:", e)
print("Saved:", PATH.with_name("column_dictionary.csv"))