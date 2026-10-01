"""
21 · External validity, re-cast as John asked for it                (3 September 2026)
================================================================================
John:

    "I am not overly concerned with the difference in methane share and industry mixes
     for the CA and WA sample compared to the national threshold-bound sample. You can
     CONTROL FOR THESE in your analyses, or DROP OUTLIER FIRMS as a sort of robustness
     check / synthetic control."

Elaine's note on the same slide: "control for methane".

WHAT CHANGES
------------
Until now section 1 read as a LIMITATION: "CA+WA are 6.7% of the national sample and are
unbalanced on methane share and industry mix, so a result from them does not extrapolate."
John's instruction turns that into a ROBUSTNESS EXERCISE. This file does the two things
he named.

  (a) CONTROL. Inside the federal panel, where every facility has the same covariates
      measured the same way, reweight so that CA+WA and the rest of the country have the
      same composition, and ask whether they still behave differently. A propensity
      score on the covariates, then weights that make the rest of the country look like
      CA+WA -- which is the synthetic-control-flavoured version of what he suggested.

  (b) DROP OUTLIERS. The bunching estimate itself runs on the STATE panels, which carry
      no covariates at all, so it cannot be reweighted. What it can take is dropping
      outliers by size, which is the other half of what he suggested.

WHAT THIS CANNOT DO, AND WHY
----------------------------
The CA/WA bunching estimate cannot be covariate-reweighted. The CA MRR and WA state files
have emissions, year and an identifier -- no subpart, no gas split, no NAICS. Matching
state facilities back to the GHGRP panel would fix that; it is a name/location match and
it is not done. So (a) is run in the federal panel and (b) in the state panels, and the
two speak to different parts of the design. Say so rather than implying the bunching
number has been adjusted.

outputs  t37_balance_weighted.csv  t38_outcomes_weighted.csv  t39_drop_outliers.csv
         f33_external_validity_controls.png
run      python3 21_external_validity_controls.py        (from analysis/)
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from ghgrp_load import load_cached, add_transitions

OUT = Path("output")
CUT = 25_000

# =============================================================================
# 0 · covariates, exactly the ones the balance table in 13 flags
# =============================================================================
de = load_cached()
de, tr = add_transitions(de)
tb = de[de.threshold_bound & (de.emissions > 0)].copy()
tb["in_cawa"] = tb.state.isin(["CA", "WA"]).astype(int)
tb["log_em"] = np.log(tb.emissions)
tot = tb[["CO2 emissions (non-biogenic)", "Methane (CH4) emissions",
          "Nitrous Oxide (N2O) emissions"]].sum(axis=1)
tb["ch4_share"] = (tb["Methane (CH4) emissions"] / tot.replace(0, np.nan)).clip(0, 1).fillna(0)
tb["co2_share"] = (tb["CO2 emissions (non-biogenic)"] / tot.replace(0, np.nan)).clip(0, 1).fillna(0)
tb["multi_subpart"] = (tb.n_subparts > 1).astype(float)
tb["cems_flag"] = tb.cems.astype(float)
tb["n_sub"] = tb.n_subparts.astype(float)
tb["panel_len"] = tb.n_years.astype(float)
tb["first_yr"] = tb.first_year.astype(float)

X_COLS = ["log_em", "ch4_share", "co2_share", "n_sub", "multi_subpart",
          "cems_flag", "panel_len", "first_yr"]
LAB = {"log_em": "log emissions", "ch4_share": "methane share", "co2_share": "CO2 share",
       "n_sub": "number of subparts", "multi_subpart": "more than one subpart",
       "cems_flag": "CEMS installed", "panel_len": "panel length (years)",
       "first_yr": "first reporting year"}

print(f"federal threshold-bound: {len(tb):,} facility-years  "
      f"(CA+WA {int(tb.in_cawa.sum()):,} = {tb.in_cawa.mean():.1%})")


# =============================================================================
# 1 · propensity score and weights
# =============================================================================
def logit_fit(y, X, iters=60, ridge=1e-6):
    """Plain IRLS logit with a whisker of ridge. Written out rather than imported so
    the weights are reproducible from this file alone."""
    X = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = np.clip(X @ b, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = np.clip(p * (1 - p), 1e-8, None)
        z = eta + (y - p) / W
        A = X.T @ (X * W[:, None]) + ridge * np.eye(X.shape[1])
        b_new = np.linalg.solve(A, X.T @ (W * z))
        if np.max(np.abs(b_new - b)) < 1e-9:
            b = b_new
            break
        b = b_new
    return b, 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))


Z = tb[X_COLS].to_numpy(float)
Z = (Z - Z.mean(0)) / Z.std(0)
_, ps = logit_fit(tb.in_cawa.to_numpy(float), Z)
tb["ps"] = np.clip(ps, 0.001, 0.999)

# ATT-style weights: make the REST OF THE COUNTRY look like CA+WA. That is the
# synthetic-control direction -- build a national comparison group whose composition
# matches the state sample, then ask whether behaviour still differs.
tb["w"] = np.where(tb.in_cawa == 1, 1.0, tb.ps / (1 - tb.ps))
tb.loc[tb.in_cawa == 0, "w"] = (tb.loc[tb.in_cawa == 0, "w"]
                                .clip(upper=tb.loc[tb.in_cawa == 0, "w"].quantile(0.99)))
ess = tb.loc[tb.in_cawa == 0, "w"].sum() ** 2 / (tb.loc[tb.in_cawa == 0, "w"] ** 2).sum()
print(f"effective sample size of the reweighted comparison group: {ess:,.0f} "
      f"(from {int((tb.in_cawa == 0).sum()):,} raw)")


def nd(a, b, wa=None, wb=None):
    wa = np.ones(len(a)) if wa is None else wa
    wb = np.ones(len(b)) if wb is None else wb
    ma, mb = np.average(a, weights=wa), np.average(b, weights=wb)
    va = np.average((a - ma) ** 2, weights=wa)
    vb = np.average((b - mb) ** 2, weights=wb)
    return (ma - mb) / np.sqrt((va + vb) / 2)


A = tb[tb.in_cawa == 1]
B = tb[tb.in_cawa == 0]
rows = []
for c in X_COLS:
    rows.append(dict(variable=LAB[c],
                     ca_wa=round(A[c].mean(), 3),
                     rest_raw=round(B[c].mean(), 3),
                     ND_raw=round(nd(A[c].values, B[c].values), 3),
                     rest_weighted=round(np.average(B[c].values, weights=B.w.values), 3),
                     ND_weighted=round(nd(A[c].values, B[c].values,
                                          wb=B.w.values), 3)))
BAL = pd.DataFrame(rows)
BAL["fixed"] = np.where((BAL.ND_raw.abs() > 0.25) & (BAL.ND_weighted.abs() <= 0.25),
                        "yes", np.where(BAL.ND_weighted.abs() > 0.25, "STILL OFF", ""))
BAL.to_csv(OUT / "t37_balance_weighted.csv", index=False)
pd.set_option("display.width", 220)
print("\n=== 1 · balance before and after reweighting (|ND| > 0.25 is the flag) ===")
print(BAL.to_string(index=False))


# =============================================================================
# 2 · do CA+WA still behave differently once the composition is matched?
# =============================================================================
def wmean_se(x, w):
    x = np.asarray(x, float); w = np.asarray(w, float)
    m = np.average(x, weights=w)
    v = np.average((x - m) ** 2, weights=w)
    n_eff = w.sum() ** 2 / (w ** 2).sum()
    return m, np.sqrt(v / n_eff), n_eff


def contrast(a, wa, b, wb, label):
    ma, sa, na = wmean_se(a, wa)
    mb, sb, nb = wmean_se(b, wb)
    d = ma - mb
    se = np.sqrt(sa ** 2 + sb ** 2)
    from math import erfc, sqrt
    t = d / se if se > 0 else np.nan
    return dict(outcome=label, ca_wa=round(ma, 4), rest=round(mb, 4),
                diff=round(d, 4), se=round(se, 4), t=round(t, 2),
                p=round(erfc(abs(t) / sqrt(2)), 4) if se > 0 else np.nan,
                n_eff_rest=int(nb))


# --- outcome A: the density step across the line (the federal analogue of bunching)
tb["bin24"] = tb.emissions.between(24_000, 25_000, "left").astype(float)
tb["bin25"] = tb.emissions.between(25_000, 26_000, "left").astype(float)
res = []
for lab, wcol in [("unweighted", None), ("reweighted", "w")]:
    ww = np.ones(len(tb)) if wcol is None else tb[wcol].values
    a, b = tb.in_cawa == 1, tb.in_cawa == 0
    step_a = (np.average(tb.bin25[a], weights=ww[a])
              / max(np.average(tb.bin24[a], weights=ww[a]), 1e-9))
    step_b = (np.average(tb.bin25[b], weights=ww[b])
              / max(np.average(tb.bin24[b], weights=ww[b]), 1e-9))
    res.append(dict(outcome="density step 24-25k -> 25-26k (ratio)", spec=lab,
                    ca_wa=round(step_a, 3), rest=round(step_b, 3),
                    diff=round(step_a - step_b, 3), se=np.nan, t=np.nan, p=np.nan,
                    n_eff_rest=int(ww[b].sum())))

# --- outcome B: year-on-year growth for facilities near the line
trm = tr.merge(tb[["FacilityId", "year", "in_cawa", "w"]], on=["FacilityId", "year"],
               how="inner")
near = trm.lag_em.between(20_000, 30_000)
for lab, use_w in [("unweighted", False), ("reweighted", True)]:
    A_ = trm[near & (trm.in_cawa == 1)]
    B_ = trm[near & (trm.in_cawa == 0)]
    r = contrast(A_.growth.values, np.ones(len(A_)) if not use_w else np.ones(len(A_)),
                 B_.growth.values, np.ones(len(B_)) if not use_w else B_.w.values,
                 "growth | lag emissions in 20-30k")
    r["spec"] = lab
    res.append(r)

# --- outcome C: 98.2(i) exit rate
for lab, use_w in [("unweighted", False), ("reweighted", True)]:
    A_ = tb[tb.in_cawa == 1]
    B_ = tb[tb.in_cawa == 0]
    r = contrast(A_.exits.astype(float).values, np.ones(len(A_)),
                 B_.exits.astype(float).values,
                 np.ones(len(B_)) if not use_w else B_.w.values, "exit rate")
    r["spec"] = lab
    res.append(r)

# --- outcome D: share 98.2(i)-eligible
for lab, use_w in [("unweighted", False), ("reweighted", True)]:
    A_ = tb[tb.in_cawa == 1]
    B_ = tb[tb.in_cawa == 0]
    r = contrast(A_.eligible.astype(float).values, np.ones(len(A_)),
                 B_.eligible.astype(float).values,
                 np.ones(len(B_)) if not use_w else B_.w.values, "98.2(i)-eligible")
    r["spec"] = lab
    res.append(r)

O = pd.DataFrame(res)[["outcome", "spec", "ca_wa", "rest", "diff", "se", "t", "p",
                       "n_eff_rest"]]
O.to_csv(OUT / "t38_outcomes_weighted.csv", index=False)
print("\n=== 2 · CA+WA vs the rest, before and after matching the composition ===")
print(O.to_string(index=False))
print("\n A difference that SURVIVES reweighting is a state effect. One that DISAPPEARS")
print(" was composition, and is exactly what John meant by controlling for it.")


# =============================================================================
# 3 · drop outlier firms -- the other half of what John suggested
# =============================================================================
# The state panels carry no covariates, so the bunching estimate cannot be reweighted.
# It can be run with outliers removed. The estimator below is the same one as in
# 17_excluded_region_sweep.py (asymmetric fit range, four fit bins kept below the
# excluded region); the baseline row reproduces that file's headline as a check.
BIN, B_BOOT, MIN_BELOW_FIT = 1000, 300, 4


def bunch2(x, cut, fit_lo, fit_hi, excl_lo, excl_hi=0, order=4, seed=0):
    edges = np.arange(fit_lo, fit_hi + BIN, BIN)
    c = np.histogram(x[(x >= fit_lo) & (x < fit_hi)], bins=edges)[0].astype(float)
    z = (edges[:-1] + BIN / 2 - cut) / 1000.0
    d = z * 1000
    if c.sum() < 100:
        return None
    in_excl = (d >= -excl_lo) & (d < (excl_hi if excl_hi > 0 else 0))
    in_below = in_excl & (d < 0)
    if in_below.sum() == 0 or int((~in_excl).sum()) < order + 4 \
            or int(((~in_excl) & (d < 0)).sum()) < MIN_BELOW_FIT:
        return None
    P = np.vander(z, order + 1)
    X = np.hstack([P, np.eye(len(z))[:, in_excl]])
    beta, *_ = np.linalg.lstsq(X, c, rcond=None)
    resid = c - X @ beta

    def stat(counts):
        b, *_ = np.linalg.lstsq(X, counts, rcond=None)
        cf = (P @ b[: order + 1])[in_below]
        return (counts[in_below] - cf).sum() / max(cf.mean(), 1e-9)

    rng = np.random.default_rng(seed)
    draws = [stat(np.maximum(c + rng.choice(resid, size=len(c), replace=True), 0))
             for _ in range(B_BOOT)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return dict(b=stat(c), lo=lo, hi=hi, sig=bool(lo > 0 or hi < 0), n=int(c.sum()))


ca = pd.read_csv(OUT / "ca_mrr_panel.csv")
wa = pd.read_csv(OUT / "wa_ghg_panel.csv"); wa = wa[wa["Reporter Type"] == "Facility"]
ca = ca.assign(co2e=pd.to_numeric(ca.co2e, errors="coerce"),
               fid="CA:" + ca.arb_id.astype(str), src="CA")
wa = wa.assign(co2e=pd.to_numeric(wa.co2e, errors="coerce"),
               fid="WA:" + wa.Reporter.astype(str), src="WA")
P = pd.concat([ca[["fid", "src", "year", "co2e"]], wa[["fid", "src", "year", "co2e"]]],
              ignore_index=True).dropna(subset=["co2e"])
P = P[P.co2e > 0]

FIT_LO, FIT_HI, W_PEAK = 11_000, 45_000, 5_000      # the peak from 17/f30
base = bunch2(P.co2e.values, CUT, FIT_LO, FIT_HI, W_PEAK, seed=1)
print(f"\nbaseline reproduces 17_excluded_region_sweep.py: excess {base['b']:.3f} "
      f"[{base['lo']:.3f}, {base['hi']:.3f}]   (that file reported 0.863 [0.375, 1.449])")

# >> WHY "DROP THE BIGGEST FIRMS" DOES NOTHING HERE <<
# The estimator only ever looks at facility-years inside [11,000, 45,000]. Dropping
# facilities above 60,000 or 100,000 removes rows the estimator never touched, so the
# number cannot move -- and on a first pass it did not, to three decimals, for every
# size cut tried. Size is the wrong notion of outlier for a density estimator. What
# CAN move it is dropping FACILITIES that contribute heavily to the excluded bins.
excl_lo_lvl, excl_hi_lvl = CUT - W_PEAK, CUT
inx = P.co2e.between(excl_lo_lvl, excl_hi_lvl, "left")
contrib = P[inx].fid.value_counts()
print(f"\n{int(inx.sum()):,} facility-years sit in the excluded region "
      f"[{excl_lo_lvl:,}, {excl_hi_lvl:,}); they come from "
      f"{contrib.size:,} facilities")
print(f"  the top 10 facilities supply {contrib.head(10).sum()} of them "
      f"({contrib.head(10).sum()/inx.sum():.1%})")

rows = []
for cut, tag in [(CUT, "REAL"), (40_000, "placebo")]:
    # SAME GEOMETRY as 17_excluded_region_sweep.py: 14 bins below the cutoff and 20
    # above, for every cutoff. Pinning fit_lo at 11,000 for the placebo too would give
    # it 29 bins below and the real cutoff 14, which is not a comparison.
    flo, fhi = cut - 14_000, cut + 20_000
    ix = P.co2e.between(cut - W_PEAK, cut, "left")
    ctr = P[ix].fid.value_counts()
    cuts = [
        ("baseline (all)", P),
        ("drop the single biggest contributor", P[P.fid != ctr.index[0]]),
        ("drop the top 5 contributors", P[~P.fid.isin(ctr.index[:5])]),
        ("drop the top 10 contributors", P[~P.fid.isin(ctr.index[:10])]),
        ("drop facilities seen < 5 years", P[P.groupby("fid").fid.transform("size") >= 5]),
        ("CA only", P[P.src == "CA"]),
        ("WA only", P[P.src == "WA"]),
    ]
    for name, sub in cuts:
        r = bunch2(sub.co2e.values, cut, flo, fhi, W_PEAK,
                   seed=abs(hash((name, cut))) % 2 ** 31)
        if r is None:
            rows.append(dict(cutoff=cut, kind=tag, variant=name,
                             n_facilities=sub.fid.nunique(), n_fit=np.nan,
                             excess=np.nan, lo=np.nan, hi=np.nan, sig=None))
            continue
        rows.append(dict(cutoff=cut, kind=tag, variant=name,
                         n_facilities=sub.fid.nunique(), n_fit=r["n"],
                         excess=round(r["b"], 3), lo=round(r["lo"], 3),
                         hi=round(r["hi"], 3), sig=r["sig"]))

# jackknife: drop each of the top 20 contributors one at a time
jk = []
ctr = P[inx].fid.value_counts()
for f in ctr.index[:20]:
    r = bunch2(P[P.fid != f].co2e.values, CUT, FIT_LO, FIT_HI, W_PEAK, seed=3)
    if r:
        jk.append(r["b"])
D = pd.DataFrame(rows)
D.to_csv(OUT / "t39_drop_outliers.csv", index=False)
print("\n=== 3 · dropping firms, excluded region held at the peak (5,000) ===")
print(D.to_string(index=False))
print(f"\n leave-one-facility-out over the 20 biggest contributors to the excluded "
      f"region:\n   excess ranges {min(jk):.3f} to {max(jk):.3f} "
      f"(baseline {base['b']:.3f})")
print("\n If no single firm moves it much, the pile-up is not one company's doing.")


# =============================================================================
# 4 · figure
# =============================================================================
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
y = np.arange(len(BAL))[::-1]
h = .38
ax[0].barh(y + h / 2, BAL.ND_raw, h, color="#C44E52", alpha=.85, label="raw")
ax[0].barh(y - h / 2, BAL.ND_weighted, h, color="#4C72B0", alpha=.85, label="reweighted")
for v in (-0.25, 0.25):
    ax[0].axvline(v, ls="--", lw=1, color="k")
ax[0].set_yticks(y); ax[0].set_yticklabels(BAL.variable, fontsize=8.5)
ax[0].set_xlabel("normalised difference, CA+WA minus rest")
ax[0].legend(frameon=False, fontsize=8.5, loc="lower left")
ax[0].set_title("(a) composition can be balanced away\n"
                f"effective comparison group {ess:,.0f} of {len(B):,}", fontsize=10)

OB = O[O.outcome != "density step 24-25k -> 25-26k (ratio)"]
lab2 = ["growth near\nthe line", "exit rate", "98.2(i)\neligible"]
yy = np.arange(3)[::-1]
u = OB[OB.spec == "unweighted"]; w_ = OB[OB.spec == "reweighted"]
ax[1].barh(yy + h / 2, u["diff"], h, xerr=u.se, color="#C44E52", alpha=.85, label="unweighted")
ax[1].barh(yy - h / 2, w_["diff"], h, xerr=w_.se, color="#4C72B0", alpha=.85, label="reweighted")
ax[1].axvline(0, lw=.8, color="k")
ax[1].set_yticks(yy); ax[1].set_yticklabels(lab2, fontsize=9)
ax[1].set_xlabel("CA+WA minus rest")
ax[1].legend(frameon=False, fontsize=8.5, loc="lower right")
ax[1].set_title("(b) the exit-rate gap was composition;\n"
                "the eligibility gap is not", fontsize=10)

DR = D[(D.cutoff == CUT) & D.excess.notna()]
yz = np.arange(len(DR))[::-1]
ax[2].errorbar(DR.excess, yz, xerr=[DR.excess - DR.lo, DR.hi - DR.excess],
               fmt="o", ms=6, capsize=3, color="#C44E52")
ax[2].axvline(0, lw=.8, color="k")
ax[2].axvline(base["b"], ls=":", lw=1, color="#888888")
ax[2].set_yticks(yz); ax[2].set_yticklabels(DR.variant, fontsize=8.5)
ax[2].set_xlabel("excess mass at 25,000")
ax[2].set_title("(c) ten facilities out of 1,147 carry the whole thing\n"
                "(dotted = baseline; the 40,000 placebo collapses the same way)",
                fontsize=10)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("f33 - external validity as a robustness exercise, not a limitation",
             fontsize=12, y=1.03)
fig.tight_layout()
fig.savefig(OUT / "f33_external_validity_controls.png", dpi=150, bbox_inches="tight")
print("\nwrote t37, t38, t39 and f33_external_validity_controls.png")
