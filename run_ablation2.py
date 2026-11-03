"""
run_ablation.py
Feature-family ablation for lapse forecasting on a Curtin-lab lag feature file.

Question: do CHANGE features (deviation from a person's own baseline) add predictive
information beyond static traits, past use, and RAW current / aggregate EMA state?

Usage (after audit_features.py has created the Parquet copy):
    python3 run_ablation.py dataset/features_0lag_v3.parquet --stride 6 --folds 5 --repeats 2

Quick test:   --stride 24 --repeats 1 --boot 100
Final run:    --stride 1  --repeats 3 --boot 500        (slow)

Design choices (all deliberate, see comments):
  * Participants never split across train/test  (StratifiedGroupKFold on subid).
  * Rows are hourly and nearly duplicate, so --stride keeps every k-th row per person.
  * Metrics are computed on out-of-fold predictions averaged over repeats.
  * Uncertainty = bootstrap over PARTICIPANTS, not rows (rows are not independent).
  * "Within-person AUROC" asks: for the same person, are high-risk hours ranked above
    safe hours?  This isolates temporal change from between-person differences.
"""
import argparse
import re
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--stride", type=int, default=6, help="keep every k-th hourly row per participant")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--repeats", type=int, default=2)
ap.add_argument("--boot", type=int, default=300, help="participant bootstrap draws (0 = skip)")
ap.add_argument("--model", choices=["hgb", "logit"], default="hgb")
ap.add_argument("--min_pos", type=int, default=3, help="min positives per person for within-person AUROC")
ap.add_argument("--min_hours", type=float, default=0,
                help="drop rows in the first N hours of each person (thin personal baseline); needs hours_since_start")
ap.add_argument("--quiet168", action="store_true",
                help="keep only windows with NO lapse in the past 168 h (emerging-risk subset)")
ap.add_argument("--no_past_use", action="store_true",
                help="remove past-use features from every set (EMA-only question)")
ap.add_argument("--no_static", action="store_true",
                help="remove static/demographic features (they can hurt grouped generalization)")
ap.add_argument("--drop_top_k", type=int, default=0,
                help="robustness: drop the k participants with the most positive windows")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="ablation_results")
args = ap.parse_args()

# ------------------------------------------------------------------ feature families
META = {"label_num", "subid", "dttm_label", "lapse", "y", "hours_since_start"}
PAT = re.compile(
    r"^(?:ema_(?P<item>\d+)\.)?p(?P<period>\d+)\.l(?P<lag>\d+)\."
    r"(?P<kind>[rd])(?P<stat>ratecount|recent|min|max|median)"
    r"(?:_response|\.count\.(?P<target>lapse|ema))$"
)


def family(c):
    if c in META:
        return "meta"
    if c.startswith("demo_") or c == "label_day":
        return "static"
    m = PAT.match(c)
    if not m:
        return "unparsed"
    d = m.groupdict()
    kind = "change" if d["kind"] == "d" else "raw"
    if d["stat"] == "ratecount":
        return "past_use" if d["target"] == "lapse" else "compliance"
    if d["stat"] == "recent":
        return f"current_{kind}"
    return f"aggregate_{kind}"


df = pd.read_parquet(args.path) if args.path.endswith(".parquet") else pd.read_csv(args.path)
if "y" not in df:
    df["y"] = df["lapse"].astype(str).str.lower().isin(["yes", "1", "true"]).astype("int8")
df = df.sort_values(["subid", "dttm_label"]).reset_index(drop=True)
if args.min_hours > 0:
    df = df[df["hours_since_start"] >= args.min_hours].reset_index(drop=True)
if args.drop_top_k > 0:
    top = df.groupby("subid")["y"].sum().nlargest(args.drop_top_k).index
    df = df[~df["subid"].isin(top)].reset_index(drop=True)
    print(f"dropped {args.drop_top_k} heaviest-lapse participants: {list(top)}")
if args.quiet168:
    q = next(c for c in df.columns if re.match(r"^p168\.l\d+\.rratecount\.count\.lapse$", c))
    df = df[df[q] <= 0].reset_index(drop=True)
    print(f"quiet168 subset: {len(df):,} rows, {df['y'].mean():.3%} positive, {df.subid.nunique()} participants")
df = df[df.groupby("subid").cumcount() % args.stride == 0].reset_index(drop=True)

fam = {c: family(c) for c in df.columns}
bad = [c for c, f in fam.items() if f == "unparsed"]
assert not bad, f"unparsed columns: {bad[:5]}"

# one-hot encode categorical static columns (no labels involved -> no leakage)
static_raw = [c for c, f in fam.items() if f == "static"]
cat = [c for c in static_raw if df[c].dtype == object or str(df[c].dtype).startswith("str")]
dummies = pd.get_dummies(df[cat], dtype="float32")
num_static = [c for c in static_raw if c not in cat]
df = pd.concat([df.drop(columns=cat), dummies], axis=1)
STATIC = [] if args.no_static else num_static + dummies.columns.tolist()


def cols(*families):
    return [c for c, f in fam.items() if f in families and c in df.columns]


PU = [] if args.no_past_use else cols("past_use")
C = STATIC + PU
SETS = {
    "A_static": STATIC,
    "B_past_use": PU,
    "C_static+past_use": C,
    "D_C+current_raw": C + cols("current_raw"),
    "E_C+raw_state": C + cols("current_raw", "aggregate_raw"),
    "F_C+change_only": C + cols("current_change", "aggregate_change"),
    "G_E+change(full)": C + cols("current_raw", "aggregate_raw", "current_change", "aggregate_change"),
    "H_all+compliance": STATIC + PU + cols("compliance", "current_raw", "aggregate_raw",
                                      "current_change", "aggregate_change"),
}
if args.no_past_use:
    SETS.pop("B_past_use")
SETS = {k: v for k, v in SETS.items() if v}   # drop sets emptied by --no_static / --no_past_use
print(f"rows={len(df):,} participants={df.subid.nunique()} positives={df.y.mean():.3%}")
for k, v in SETS.items():
    print(f"  {k:<22} {len(v):>4} features")

y = df["y"].to_numpy()
groups = df["subid"].to_numpy()


def make_model(seed):
    if args.model == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.06, max_leaf_nodes=12, min_samples_leaf=100,
            l2_regularization=1.0, random_state=seed)
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.05, max_iter=500))


# ------------------------------------------------------------------ cross-validation
oof = {k: np.zeros((args.repeats, len(df))) for k in SETS}
t0 = time.time()
for r in range(args.repeats):
    cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed + r)
    for fold, (tr, te) in enumerate(cv.split(df, y, groups)):
        for name, feats in SETS.items():
            X = df[feats].to_numpy(dtype="float32")
            m = make_model(args.seed + r).fit(X[tr], y[tr])
            oof[name][r, te] = m.predict_proba(X[te])[:, 1]
        print(f"repeat {r + 1}/{args.repeats} fold {fold + 1}/{args.folds} done ({time.time() - t0:.0f}s)")
pred = {k: v.mean(axis=0) for k, v in oof.items()}  # average over repeats

# ------------------------------------------------------------------ metrics
pidx = {p: np.where(groups == p)[0] for p in np.unique(groups)}
plist = np.array(list(pidx))


def within_person_auc(p):
    out = {}
    for pid, ix in pidx.items():
        yy = y[ix]
        if yy.sum() >= args.min_pos and (1 - yy).sum() >= args.min_pos:
            out[pid] = roc_auc_score(yy, p[ix])
    return out


def top_recall(yv, p, frac=0.10):
    k = max(1, int(len(p) * frac))
    return yv[np.argsort(-p)[:k]].sum() / max(1, yv.sum())


wp = {k: within_person_auc(p) for k, p in pred.items()}
rows = []
for k, p in pred.items():
    rows.append(dict(model=k, n_features=len(SETS[k]),
                     AUROC=roc_auc_score(y, p), AUPRC=average_precision_score(y, p),
                     within_person_AUROC=np.mean(list(wp[k].values())),
                     recall_at_top10pct=top_recall(y, p), Brier=brier_score_loss(y, np.clip(p, 0, 1))))
summ = pd.DataFrame(rows).set_index("model")
pp = pd.DataFrame({k: pd.Series(v) for k, v in wp.items()}); pp.index.name = "subid"
pp.to_csv(f"{args.out}_per_person_auc.csv")

# ------------------------------------------------------------------ participant bootstrap
CONTRASTS = [
    ("C_static+past_use", "A_static"),
    ("D_C+current_raw", "C_static+past_use"),
    ("E_C+raw_state", "D_C+current_raw"),
    ("G_E+change(full)", "E_C+raw_state"),      # KEY: do change features add beyond raw state?
    ("F_C+change_only", "E_C+raw_state"),       # change vs raw, same base
]
CONTRASTS = [(a, b) for a, b in CONTRASTS if a in SETS and b in SETS]
if args.boot > 0:
    rng = np.random.default_rng(args.seed)
    draws = {k: {"AUROC": [], "AUPRC": [], "within_person_AUROC": []} for k in SETS}
    for _ in range(args.boot):
        pick = rng.choice(plist, size=len(plist), replace=True)
        ix = np.concatenate([pidx[p] for p in pick])
        for k, p in pred.items():
            draws[k]["AUROC"].append(roc_auc_score(y[ix], p[ix]))
            draws[k]["AUPRC"].append(average_precision_score(y[ix], p[ix]))
            vals = [wp[k][p_] for p_ in pick if p_ in wp[k]]
            draws[k]["within_person_AUROC"].append(np.mean(vals) if vals else np.nan)
    for k in SETS:
        for met in ("AUROC", "AUPRC", "within_person_AUROC"):
            lo, hi = np.nanpercentile(draws[k][met], [2.5, 97.5])
            summ.loc[k, f"{met}_CI"] = f"[{lo:.3f}, {hi:.3f}]"
    crow = []
    for a, b in CONTRASTS:
        rec = dict(contrast=f"{a}  minus  {b}")
        for met in ("AUROC", "AUPRC", "within_person_AUROC"):
            d = np.array(draws[a][met]) - np.array(draws[b][met])
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            rec[met] = f"{summ.loc[a, met] - summ.loc[b, met]:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        crow.append(rec)
    contrasts = pd.DataFrame(crow)

# ------------------------------------------------------------------ report + save
pd.set_option("display.width", 220, "display.max_columns", 20)
print("\n=== PERFORMANCE (out-of-fold, participant-grouped) ===")
print(summ.round(3).to_string())
if args.boot > 0:
    print("\n=== PAIRED CONTRASTS (difference, 95% participant-bootstrap CI) ===")
    print(contrasts.to_string(in,dex=False))
    contrasts.to_csv(f"{args.out}_contrasts.csv", index=False)
summ.to_csv(f"{args.out}_summary.csv")
print("\n=== PER-PERSON AUROC DISTRIBUTION (quantiles across participants) ===")
print(pp.quantile([.1, .25, .5, .75, .9]).round(3).T.to_string())
out = df[["subid", "dttm_label", "y"]].copy()
for k, p in pred.items():
    out[k] = p
out.to_parquet(f"{args.out}_oof_predictions.parquet", index=False)
print(f"\nSaved {args.out}_summary.csv, {args.out}_contrasts.csv, {args.out}_oof_predictions.parquet")
print("Within-person AUROC averaged over", len(wp["G_E+change(full)"]), "participants with both classes")