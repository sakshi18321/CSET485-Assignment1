"""
Independent re-computation of the headline results.

Uses scikit-learn (tight solver tolerance) and plain numpy formulas instead of statsmodels / the notebook's helper
functions, then compares against results/results.json.  Writes results/verification.txt (read by build_report.py)
and exits non-zero if anything disagrees.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "results" / "results.json").read_text())
raw = pd.read_csv(ROOT / "data" / "compas-scores-two-years.csv")

CONT = ["age", "priors_count", "juv_fel_count", "juv_misd_count", "juv_other_count"]
d = pd.DataFrame({c: raw[c] for c in CONT})
d["male"] = (raw.sex == "Male").astype(int)
d["felony"] = (raw.c_charge_degree == "F").astype(int)
rg = raw.race.where(raw.race.isin(["African-American", "Caucasian", "Hispanic"]), "Other")
for r in ["African-American", "Hispanic", "Other"]:
    d["race_" + r.replace("-", "_")] = (rg == r).astype(int)
y = raw.two_year_recid.values
dec = raw.decile_score.values
race = raw.race.values

idx_tr, idx_te = train_test_split(np.arange(len(raw)), test_size=0.30, random_state=450)
sc = StandardScaler().fit(d.loc[idx_tr, CONT])


def X(idx, drop=()):
    Z = d.iloc[idx].copy()
    Z[CONT] = sc.transform(Z[CONT])
    return Z.drop(columns=list(drop))


Xtr, Xte = X(idx_tr), X(idx_te)
lin = LinearRegression().fit(Xtr, dec[idx_tr])
log = LogisticRegression(C=np.inf, tol=1e-12, max_iter=200000).fit(Xtr, y[idx_tr])
pa, pb = lin.predict(Xte), log.predict_proba(Xte)[:, 1]
yt = y[idx_te]

lines, bad = [], []


def check(label, ok, detail=""):
    lines.append(f"{label} | {'PASS' if ok else 'FAIL'}{(' (' + detail + ')') if detail else ''}")
    if not ok:
        bad.append(label)


P1 = R["part1"]
med = np.median(pa)
hiA, hiB = pa >= med, pb >= 0.5
check("Independent refit: median of Model A predictions", abs(med - P1["median_pred_A"]) < 1e-8, f"{med:.4f}")
check("Independent refit: mismatch count between the two flags", int((hiA != hiB).sum()) == P1["mismatch_n"], f"{int((hiA != hiB).sum())}")
check("Independent refit: Model B accuracy and AUC", abs(((pb >= .5) == yt).mean() - R["modelB"]["accuracy"]) < 1e-9 and abs(roc_auc_score(yt, pb) - R["modelB"]["auc"]) < 1e-4,
      f"acc {((pb >= .5) == yt).mean():.4f}, AUC {roc_auc_score(yt, pb):.4f}")

fp = ((pb >= .75) & (yt == 0)).sum()
fn = ((pb <= .25) & (yt == 1)).sum()
check("Independent refit: confident false positives / false negatives", (fp, fn) == (R["p23"]["n_fp"], R["p23"]["n_fn"]), f"{fp} / {fn}")

# fairness sweep with direct numpy formulas
A = race[idx_te] == "Caucasian"
B = race[idx_te] == "African-American"
worst = 0.0
nan_ok = True
for row in R["p24"]["sweep"]:
    t = row["t"]
    yh = (pb >= t).astype(int)

    def rate(m, num_mask):
        den = m & num_mask
        return yh[den].mean() if den.sum() else np.nan

    flagA, flagB = yh[A].mean(), yh[B].mean()
    tprA, tprB = rate(A, yt == 1), rate(B, yt == 1)
    fprA, fprB = rate(A, yt == 0), rate(B, yt == 0)
    ppvA = yt[A & (yh == 1)].mean() if (A & (yh == 1)).sum() else np.nan
    ppvB = yt[B & (yh == 1)].mean() if (B & (yh == 1)).sum() else np.nan
    dp = flagA / flagB if flagB else np.nan
    eo = min(tprA / tprB if tprB else np.nan, fprA / fprB if fprB else np.nan)
    pp = ppvA / ppvB if (ppvB and not np.isnan(ppvA)) else np.nan
    for name, v in [("DP", dp), ("EO", eo), ("PP", pp)]:
        nb = row[name]
        if (nb is None) != bool(np.isnan(v)):
            nan_ok = False
        elif nb is not None:
            worst = max(worst, abs(nb - v))
check("Independent recomputation of DP / EO / PP at all 10 thresholds", worst < 1e-4 and nan_ok, f"max abs difference {worst:.1e}; NA pattern identical")

# omitted-variable and subgroup coefficients
red = LogisticRegression(C=np.inf, tol=1e-12, max_iter=200000).fit(Xtr.drop(columns="priors_count"), y[idx_tr])
cb = dict(zip(Xtr.drop(columns="priors_count").columns, red.coef_[0]))["race_African_American"]
ref = [r for r in R["p22"]["table"] if r["predictor"] == "race_African_American"][0]["beta_reduced"]
check("Independent refit: reduced-model race coefficient", abs(cb - ref) < 1e-3, f"{cb:.4f} vs {ref:.4f}")

Xall = X(np.arange(len(raw)), drop=["male"])
m_ = raw.sex.values == "Male"
for lab, mask, key in [("men", m_, "A_coef"), ("women", ~m_, "B_coef")]:
    lr = LogisticRegression(C=np.inf, tol=1e-12, max_iter=200000).fit(Xall[mask], y[mask])
    b = dict(zip(Xall.columns, lr.coef_[0]))["priors_count"]
    ref = [r for r in R["p21"]["primary"]["table"] if r["predictor"] == "priors_count"][0][key]
    check(f"Independent refit: priors_count coefficient, {lab}", abs(b - ref) < 1e-3, f"{b:.4f} vs {ref:.4f}")

(ROOT / "results" / "verification.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
sys.exit(1 if bad else 0)
