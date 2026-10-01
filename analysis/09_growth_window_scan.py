"""
09 · Year-on-year growth near vs away from the threshold — MEAN differences.

WHY THIS SCRIPT EXISTS
----------------------
John's 6 Aug observation was about the MEAN: "firms within 5,000 units of the
threshold seem to have emissions grow slightly less fast." His follow-up question
was also about the mean: "what happens if we shrink or grow the window?"

Notebook 02 §1 tested the mean ONCE, at +/-5,000, found it insignificant, and then
§4 -- the section titled "window sensitivity", which is supposed to answer that
question -- reported ONLY sd ratios at four windows. No mean difference appears
anywhere in the window scan. Switching the statistic after the first one came back
insignificant, with no stated motivation, is not a defensible move.

This script answers the question that was actually asked, at 1,000-tonne increments
from 1,000 to 10,000 (John, 20 Aug), at the real cutoff and two placebos.

WHAT "NEAR" AND "AWAY" MEAN, STATED EXPLICITLY
----------------------------------------------
Windows are defined on lagged emissions (t-1), not on t: the window has to be a
property of where the facility STARTED, otherwise the growth rate helps decide
which window the observation lands in and the comparison is circular.

Two definitions of "away", because the first one is confounded:
  far_all   every facility-year outside the near window  (John's literal wording)
  far_band  outside the window but with lag_em in [10k, 100k]
Rationale for far_band: growth volatility and drift both vary with facility size,
and far_all includes facilities 100x larger than the near group. far_band holds
size roughly comparable so the contrast is not a size contrast.

WHERE THE GAP COMES FROM
------------------------
For each window the mean difference is computed four ways, each stripping out one
candidate source:
  raw          no controls
  year FE      common shocks (the 2015-16 oil collapse lives here)
  year x N3    common shocks + industry composition
  facility FE  everything time-invariant about the facility
If the difference dies under year x NAICS3, it is composition. If it survives
facility demeaning, it is within-facility behaviour.

SD IS REPORTED, BUT NOT AS A MANIPULATION TEST
----------------------------------------------
Manipulation predicts a shift in LOCATION or a pile-up of MASS, not a narrowing of
variance. So sd cannot test manipulation. It is reported here for exactly one
purpose: as a composition diagnostic -- if the near group's growth is less variable
after conditioning on size and industry, the near group is a different kind of
facility, which is a reason to distrust the raw mean comparison. That is the only
claim the sd column supports.

Outputs
  output/t17_growth_window_mean.csv   the scan
  output/f22_growth_window_mean.png   mean difference vs window width, with CIs
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from math import erfc, sqrt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")


def welch(a, b):
    """Welch two-sample test. Returns (difference, se, t, two-sided p).

    p uses the normal approximation to the t distribution. Every comparison below
    has n in the thousands, where the two agree to more decimal places than are
    reported. Written out rather than imported so the script runs without scipy.
    """
    na, nb = len(a), len(b)
    diff = a.mean() - b.mean()
    se = sqrt(a.var(ddof=1) / na + b.var(ddof=1) / nb)
    t = diff / se if se > 0 else np.nan
    p = erfc(abs(t) / sqrt(2)) if se > 0 else np.nan
    return diff, se, t, p
from ghgrp_load import load_cached

OUT = Path("output")
CUTOFFS = [(25_000, "REAL"), (40_000, "placebo"), (60_000, "placebo")]
WINDOWS = list(range(1_000, 10_001, 1_000))

# ---------------------------------------------------------------- panel
de = load_cached()
d = de.sort_values(["FacilityId", "year"]).copy()
d["lag_em"] = d.groupby("FacilityId").emissions.shift(1)
d = d[(d.emissions > 0) & (d.lag_em > 0)].copy()
d["growth"] = np.log(d.emissions / d.lag_em)
d["naics3"] = d.naics.astype(str).str[:3]
print(f"{len(d):,} facility-year transitions | {d.FacilityId.nunique():,} facilities")


def demean(s, *keys):
    """Residualise s on the fixed effects implied by keys (sequential demeaning)."""
    r = s.copy()
    for k in keys:
        r = r - r.groupby(k).transform("mean")
    return r


# residualised growth, computed ONCE on the full sample so the FE are estimated
# off every observation rather than off the near/far split
d["g_year"] = demean(d.growth, d.year)
d["g_yr_n3"] = demean(d.growth, d.year.astype(str) + "_" + d.naics3)
d["g_fac"] = demean(d.growth, d.FacilityId)

SPECS = [("raw", "growth"), ("year FE", "g_year"),
         ("year x NAICS3 FE", "g_yr_n3"), ("facility FE", "g_fac")]

# ---------------------------------------------------------------- scan
rows = []
for cut, kind in CUTOFFS:
    for w in WINDOWS:
        near_m = d.lag_em.between(cut - w, cut + w)
        for far_lab, far_m in [("far_all", ~near_m),
                               ("far_band", ~near_m & d.lag_em.between(10_000, 100_000))]:
            A_i, B_i = d[near_m], d[far_m]
            if len(A_i) < 200 or len(B_i) < 200:
                continue
            rec = dict(cutoff=cut, kind=kind, window=w, far_def=far_lab,
                       n_near=len(A_i), n_far=len(B_i))
            for lab, col in SPECS:
                a, b = A_i[col], B_i[col]
                diff, se, t, p = welch(a, b)
                rec[f"diff__{lab}"] = round(diff, 5)
                rec[f"t__{lab}"] = round(t, 2)
                rec[f"p__{lab}"] = round(p, 4)
                rec[f"lo__{lab}"] = round(diff - 1.96 * se, 5)
                rec[f"hi__{lab}"] = round(diff + 1.96 * se, 5)
            # composition diagnostic only -- see docstring
            rec["sd_ratio_raw"] = round(A_i.growth.std() / B_i.growth.std(), 3)
            rec["sd_ratio_yr_n3"] = round(A_i.g_yr_n3.std() / B_i.g_yr_n3.std(), 3)
            rec["median_diff_raw"] = round(A_i.growth.median() - B_i.growth.median(), 5)
            rows.append(rec)

T = pd.DataFrame(rows)
T.to_csv(OUT / "t17_growth_window_mean.csv", index=False)

pd.set_option("display.width", 250)
print("\n=== MEAN DIFFERENCE (near - far), far_band, raw ===")
print(T[T.far_def == "far_band"].pivot(index="window", columns="cutoff",
                                       values="diff__raw").to_string())
print("\n=== p-values, far_band, raw ===")
print(T[T.far_def == "far_band"].pivot(index="window", columns="cutoff",
                                       values="p__raw").to_string())
print("\n=== MEAN DIFFERENCE at the REAL cutoff, all four specifications (far_band) ===")
r = T[(T.far_def == "far_band") & (T.cutoff == 25_000)]
print(r[["window", "n_near", "n_far"] + [f"diff__{l}" for l, _ in SPECS]
        + [f"p__{l}" for l, _ in SPECS]].to_string(index=False))
print("\n=== sd ratio (COMPOSITION DIAGNOSTIC ONLY, not a manipulation test) ===")
print(T[T.far_def == "far_band"].pivot(index="window", columns="cutoff",
                                       values="sd_ratio_yr_n3").to_string())
print("\nwrote t17_growth_window_mean.csv")


# =============================================================================
# 2 · Where the difference comes from: mean reversion inside the window
# =============================================================================
# The facility-FE column above is significant at the real cutoff. Before reading
# that as behaviour, check the most boring explanation.
#
# A facility-year enters the window on its LAGGED level. A facility whose typical
# level is well above 25,000 enters the window only in a year when it happened to
# be low; the next year it reverts upward, and that shows up as positive growth.
# Nothing behavioural is required.
#
# The test is whether window membership is selected on low draws. Measure it
# directly: rel_start = log(lag_em / the facility's own mean emissions).
#   rel_start < 0  ->  the window catches the facility in a low year
# Truncation is why this should bite hardest at 25,000: facilities whose typical
# level is BELOW 25,000 never report, so the [20k, 30k] window is populated by
# facilities that normally sit higher and are having a bad year. The same
# truncation that kills the level density is still operating in growth-rate space.
d["fac_mean"] = d.groupby("FacilityId").emissions.transform("mean")
d["rel_start"] = np.log(d.lag_em / d.fac_mean)

mr = []
for cut, kind in CUTOFFS:
    for w in WINDOWS:
        s = d[d.lag_em.between(cut - w, cut + w)]
        if len(s) < 200:
            continue
        fe = T[(T.cutoff == cut) & (T.window == w) & (T.far_def == "far_band")]
        mr.append(dict(cutoff=cut, kind=kind, window=w, n=len(s),
                       rel_start=round(s.rel_start.mean(), 4),
                       share_low_draw=round((s.rel_start < 0).mean(), 4),
                       growth_mean=round(s.growth.mean(), 4),
                       diff_facFE=float(fe["diff__facility FE"].iloc[0]) if len(fe) else np.nan,
                       p_facFE=float(fe["p__facility FE"].iloc[0]) if len(fe) else np.nan))
M = pd.DataFrame(mr)
M.to_csv(OUT / "t18_mean_reversion.csv", index=False)
print("\n=== mean-reversion diagnostic ===")
print(M.groupby("cutoff")[["rel_start", "share_low_draw", "diff_facFE"]].mean().round(4).to_string())
print(f"\nfull sample: rel_start = {d.rel_start.mean():.4f}, "
      f"share starting below own mean = {(d.rel_start < 0).mean():.1%}")
print("\ncorrelation across all (cutoff, window) cells, rel_start vs facility-FE difference: "
      f"{M.rel_start.corr(M.diff_facFE):+.3f}")

# =============================================================================
# figure
# =============================================================================
COL = {25_000: "#C44E52", 40_000: "#4C72B0", 60_000: "#55A868"}
plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8))

# (a) mean difference vs window, comparable-size comparison group
b = T[T.far_def == "far_band"]
for cut, c in COL.items():
    s = b[b.cutoff == cut].sort_values("window")
    ax[0].plot(s.window / 1000, s["diff__raw"], "o-", color=c, ms=5,
               lw=2.2 if cut == 25_000 else 1.5,
               label=f"{cut//1000}k" + (" (REAL)" if cut == 25_000 else " (placebo)"))
    ax[0].fill_between(s.window / 1000, s["lo__raw"], s["hi__raw"], color=c, alpha=.11)
ax[0].axhline(0, color="k", lw=1, ls=":")
ax[0].set_xlabel("window half-width, thousand tCO$_2$e")
ax[0].set_ylabel("mean growth, near $-$ away")
ax[0].set_title("(a) Mean difference against facilities of\ncomparable size (10k–100k), 95% CI",
                fontsize=10)
ax[0].legend(fontsize=8.5)

# (b) the sign flip: how "away" is defined decides the sign
r = T[T.cutoff == 25_000]
for fd, lab, c, m in [("far_all", "away = every other facility-year", "#8172B2", "s"),
                      ("far_band", "away = lag 10k–100k only", "#C44E52", "o")]:
    s = r[r.far_def == fd].sort_values("window")
    ax[1].plot(s.window / 1000, s["diff__raw"], m + "-", color=c, ms=5, lw=1.9, label=lab)
    ax[1].fill_between(s.window / 1000, s["lo__raw"], s["hi__raw"], color=c, alpha=.11)
ax[1].axhline(0, color="k", lw=1, ls=":")
ax[1].set_xlabel("window half-width, thousand tCO$_2$e")
ax[1].set_ylabel("mean growth, near $-$ away")
ax[1].set_title("(b) At the REAL threshold, the sign of the\n"
                "difference is set by the comparison group", fontsize=10)
ax[1].legend(fontsize=8)

# (c) mean reversion explains the facility-FE result
for cut, c in COL.items():
    s = M[M.cutoff == cut]
    ax[2].scatter(s.rel_start, s.diff_facFE, color=c, s=40, alpha=.85,
                  label=f"{cut//1000}k" + (" (REAL)" if cut == 25_000 else ""))
ax[2].axhline(0, color="k", lw=1, ls=":"); ax[2].axvline(0, color="k", lw=1, ls=":")
ax[2].set_xlabel("mean log(lag emissions / facility's own mean)")
ax[2].set_ylabel("within-facility growth difference")
ax[2].set_title("(c) The window catches facilities in a low year,\n"
                "and they revert. Sign follows the selection.", fontsize=10)
ax[2].legend(fontsize=8.5)

plt.tight_layout()
plt.savefig(OUT / "f22_growth_window_mean.png", bbox_inches="tight")
print("\nwrote t18_mean_reversion.csv, f22_growth_window_mean.png")
