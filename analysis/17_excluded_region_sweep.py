"""
17 · Widening the excluded region until the estimate dies      (John, 3 September 2026)
================================================================================
John's note:

    "The instability of bunching for narrow thresholds is unsurprising. While a unit
     mass below the threshold would be expected in a frictionless environment,
     adjustment costs, optimization frictions, etc. often leads to a distribution
     before a threshold rather than a unit mass. You can continue to increase the
     excluded region further to see when the estimates become insignificant again."

Elaine's note on the same point: "Is that stable for the threshold as well?"

WHAT THIS IS ACTUALLY MEASURING
-------------------------------
If the response to a notch were frictionless, everyone who wanted to avoid it would sit
in one bin immediately below the line. Adjustment costs smear that mass over a region of
some width D. The excluded region has to be at least D wide or the counterfactual
polynomial is fitted through bunchers; much wider than D and the excluded region starts
eating ordinary density, the counterfactual is fitted on less and less information, and
the estimate loses significance. So the width at which significance dies is an estimate
of D -- the scale of the adjustment friction -- and can be read against the dominated-
region result, which measures the same thing a different way.

WHY THE OLD SWEEP STOPPED AT 8,000
----------------------------------
Script 13 pinned the fit range symmetric (fit_hw <= 10,000) and required
w <= fit_hw - 2,000, so at the real cutoff it never went past 8,000.

The binding constraint is real but it is only on the LEFT. The CA/WA density jumps at
10,000 -- their own state reporting threshold:

    1k bins, CA+WA:  ... 7-8k: 80   8-9k: 87   9-10k: 110 | 10-11k: 335   11-12k: 340 ...
                                                            ^ the state threshold

so a fit range reaching below ~11,000 is fitted across a discontinuity. There is no such
problem on the RIGHT. Decoupling fit_lo from fit_hi therefore buys the whole extension:

    fit_lo = 11,000 (fixed)  =>  excl_lo <= cutoff - 11,000 = 14,000 at the real cutoff

That 14,000 is a hard physical ceiling set by California and Washington's own reporting
threshold, not a modelling choice. If the estimate is still significant when the sweep
hits it, the honest answer to John's question is "we cannot see where it dies", and that
is itself the finding.

The federal always-covered sample has no truncation anywhere, so it is swept without a
ceiling as a reference for what "widening until it dies" looks like unconstrained.

outputs  t33_excluded_sweep.csv  t34_death_point.csv  f30_excluded_region_sweep.png
run      python3 17_excluded_region_sweep.py       (from analysis/)
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from ghgrp_load import load_cached

OUT = Path("output")
BIN = 1000
STATE_TRUNC = 10_000          # CA MRR and WA both start reporting here
FIT_LO_STATE = 11_000         # one bin clear of it
B = 300
MIN_BELOW_FIT = 4        # fit bins that must remain below the excluded region


# ---------------------------------------------------------------- estimator
def bunch2(x, cut, fit_lo, fit_hi, excl_lo, excl_hi=0, order=4, B=B, seed=0):
    """Chetty-Kleven polynomial counterfactual with an ASYMMETRIC fit range.

    fit_lo / fit_hi are absolute levels, not half-widths. Excess mass is measured over
    the below-cutoff excluded bins and normalised by their mean counterfactual density.
    """
    edges = np.arange(fit_lo, fit_hi + BIN, BIN)
    c = np.histogram(x[(x >= fit_lo) & (x < fit_hi)], bins=edges)[0].astype(float)
    z = (edges[:-1] + BIN / 2 - cut) / 1000.0
    if c.sum() < 100:
        return None
    d = z * 1000
    in_excl = (d >= -excl_lo) & (d < (excl_hi if excl_hi > 0 else 0))
    in_below = in_excl & (d < 0)
    n_fit_bins = int((~in_excl).sum())
    n_below_fit = int(((~in_excl) & (d < 0)).sum())
    if in_below.sum() == 0 or n_fit_bins < order + 4:
        return None
    # The counterfactual has to be FITTED on both sides of the cutoff, not extrapolated
    # into the gap from above. With fewer than MIN_BELOW_FIT bins left below the excluded
    # region the quartic is unanchored on the left and the bootstrap CI explodes
    # (at excl_lo = 14,000 the upper limit came back as 5.5e12). Refuse those cells.
    if n_below_fit < MIN_BELOW_FIT:
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
             for _ in range(B)]
    lo_ci, hi_ci = np.percentile(draws, [2.5, 97.5])
    return dict(b=stat(c), lo=lo_ci, hi=hi_ci, sig=bool(lo_ci > 0 or hi_ci < 0),
                n_fit=int(c.sum()), n_excl=int(c[in_below].sum()),
                n_fit_bins=n_fit_bins, n_below_fit=n_below_fit,
                n_excl_bins=int(in_excl.sum()))


# ---------------------------------------------------------------- samples
ca = pd.read_csv(OUT / "ca_mrr_panel.csv")
wa = pd.read_csv(OUT / "wa_ghg_panel.csv"); wa = wa[wa["Reporter Type"] == "Facility"]
S = {"CA": pd.to_numeric(ca.co2e, errors="coerce").dropna().values,
     "WA": pd.to_numeric(wa.co2e, errors="coerce").dropna().values}
S = {k: v[v > 0] for k, v in S.items()}
S["CA+WA"] = np.concatenate([S["CA"], S["WA"]])

de = load_cached()
AC = de.loc[de.always_covered & (de.emissions > 0), "emissions"].values
print(f"CA {len(S['CA']):,}  WA {len(S['WA']):,}  CA+WA {len(S['CA+WA']):,}  "
      f"federal always-covered {len(AC):,}")

print("\nwhere the state data are truncated (1k bins, CA+WA):")
h = np.histogram(S["CA+WA"], bins=np.arange(6000, 16001, 1000))[0]
for lo, n in zip(range(6000, 16000, 1000), h):
    mark = "   <-- state reporting threshold" if lo == 10_000 else ""
    print(f"  {lo//1000:>2}-{lo//1000+1:>2}k  {n:>5}{mark}")


# ---------------------------------------------------------------- the sweep
# 22,000 is dropped: with fit_lo one bin clear of the 10,000 truncation it cannot be
# swept as far as the real cutoff, so it could not be a like-for-like placebo.
CUTS = [(25_000, "REAL"), (28_000, "placebo"), (30_000, "placebo"),
        (35_000, "placebo"), (40_000, "placebo")]

# EVERY cutoff gets the SAME fit-range GEOMETRY, not the same absolute fit range:
#     fit_lo = cutoff - 14,000      fit_hi = cutoff + 20,000
# so each has exactly 14 bins below the cutoff and 20 above, and the quartic is asked to
# do the same job at each. At the real cutoff fit_lo lands on 11,000, one bin clear of
# the state truncation; every placebo sits higher still, so none crosses it. Pinning the
# same ABSOLUTE fit_lo for all of them instead would give the 40,000 placebo 29 bins
# below and the real cutoff 14, which is not a comparison.
BELOW_BINS, ABOVE_BINS = 14, 20
WMAX_COMMON = (BELOW_BINS - MIN_BELOW_FIT) * BIN                   # = 10,000
rows = []

# --- state samples: fit_lo pinned above the truncation, fit_hi free
for name in ["CA+WA", "CA", "WA"]:
    x = S[name]
    for cut, kind in CUTS:
        fit_lo = cut - BELOW_BINS * BIN
        fit_hi = cut + ABOVE_BINS * BIN
        assert fit_lo > STATE_TRUNC, (cut, fit_lo)
        for w in range(1000, WMAX_COMMON + 1, 1000):
            for shp, eh in [("asymmetric (below only)", 0), ("symmetric", w)]:
                r = bunch2(x, cut, fit_lo, fit_hi, w, eh,
                           seed=abs(hash((name, cut, w, shp))) % 2**31)
                if r is None:
                    continue
                rows.append(dict(sample=name, cutoff=cut, kind=kind, excl_shape=shp,
                                 excl_width=w, fit_lo=fit_lo, fit_hi=fit_hi,
                                 at_ceiling=(w == WMAX_COMMON), **{k: r[k] for k in
                                 ["n_fit", "n_excl", "n_fit_bins", "n_below_fit", "n_excl_bins"]},
                                 excess=round(r["b"], 3), lo=round(r["lo"], 3),
                                 hi=round(r["hi"], 3), sig=r["sig"]))

# --- federal always-covered, SAME geometry, so it is a like-for-like placebo
for w in range(1000, WMAX_COMMON + 1, 1000):
    for shp, eh in [("asymmetric (below only)", 0), ("symmetric", w)]:
        r = bunch2(AC, 25_000, 25_000 - BELOW_BINS * BIN, 25_000 + ABOVE_BINS * BIN,
                   w, eh, seed=abs(hash(("ACm", w, shp))) % 2**31)
        if r is None:
            continue
        rows.append(dict(sample="federal always-covered", cutoff=25_000,
                         kind="placebo", excl_shape=shp, excl_width=w,
                         fit_lo=25_000 - BELOW_BINS * BIN, fit_hi=25_000 + ABOVE_BINS * BIN,
                         at_ceiling=(w == WMAX_COMMON),
                         **{k: r[k] for k in ["n_fit", "n_excl", "n_fit_bins",
                                              "n_below_fit", "n_excl_bins"]},
                         excess=round(r["b"], 3), lo=round(r["lo"], 3),
                         hi=round(r["hi"], 3), sig=r["sig"]))

# --- and again with NO ceiling, because it is the one sample with no truncation at all:
#     the fit range grows with the excluded region so MIN_BELOW_FIT is always satisfied.
#     This is the only place John's exercise can be run to completion.
for w in range(1000, 20_001, 1000):
    for shp, eh in [("asymmetric (below only)", 0), ("symmetric", w)]:
        r = bunch2(AC, 25_000, 25_000 - w - MIN_BELOW_FIT * BIN - 10_000, 45_000, w, eh,
                   seed=abs(hash(("AC", w, shp))) % 2**31)
        if r is None:
            continue
        rows.append(dict(sample="federal always-covered (unconstrained)", cutoff=25_000,
                         kind="placebo (untruncated)", excl_shape=shp, excl_width=w,
                         fit_lo=25_000 - w - MIN_BELOW_FIT * BIN - 10_000, fit_hi=45_000,
                         at_ceiling=False,
                         **{k: r[k] for k in ["n_fit", "n_excl", "n_fit_bins", "n_below_fit", "n_excl_bins"]},
                         excess=round(r["b"], 3), lo=round(r["lo"], 3),
                         hi=round(r["hi"], 3), sig=r["sig"]))

T = pd.DataFrame(rows)
T.to_csv(OUT / "t33_excluded_sweep.csv", index=False)
pd.set_option("display.width", 250)

print("\n" + "=" * 78)
print("1 · CA+WA at the REAL cutoff, asymmetric excluded region")
print("=" * 78)
h = T[(T["sample"] == "CA+WA") & (T.cutoff == 25_000) &
      (T.excl_shape == "asymmetric (below only)")]
print(h[["excl_width", "n_excl_bins", "n_below_fit", "n_excl", "excess", "lo", "hi",
         "sig", "at_ceiling"]].to_string(index=False))

print("\n" + "=" * 78)
print("2 · where does significance die?  first width at which the CI covers zero")
print("=" * 78)
death = []
for (name, cut, shp), g in T.groupby(["sample", "cutoff", "excl_shape"]):
    g = g.sort_values("excl_width")
    if not g.sig.any():
        first_sig, died = np.nan, np.nan
        status = "never significant"
    else:
        first_sig = int(g.loc[g.sig, "excl_width"].min())
        after = g[(g.excl_width > first_sig) & (~g.sig)]
        if len(after):
            died = int(after.excl_width.min()); status = "dies"
        else:
            died = np.nan
            status = ("still significant at the ceiling"
                      if bool(g.iloc[-1].at_ceiling) else "still significant at the last width")
    death.append(dict(sample=name, cutoff=cut, excl_shape=shp,
                      n_widths=len(g), max_width=int(g.excl_width.max()),
                      n_sig=int(g.sig.sum()), first_sig=first_sig,
                      dies_at=died, status=status))
D = pd.DataFrame(death).sort_values(["excl_shape", "sample", "cutoff"])
D.to_csv(OUT / "t34_death_point.csv", index=False)
print(D.to_string(index=False))

print("\n" + "=" * 78)
print("3 · monotone or not?  excess mass by width, CA+WA, asymmetric")
print("=" * 78)
pv = (T[(T["sample"] == "CA+WA") & (T.excl_shape == "asymmetric (below only)")]
      .pivot(index="excl_width", columns="cutoff", values="excess"))
print(pv.round(3).to_string())
print("\n  significant cells marked:")
pvs = (T[(T["sample"] == "CA+WA") & (T.excl_shape == "asymmetric (below only)")]
       .pivot(index="excl_width", columns="cutoff", values="sig"))
print(pvs.replace({True: "  *", False: "   "}).fillna("").to_string())


# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8))
CL = {25000: "#C44E52", 28000: "#8C8C8C",
      30000: "#4C72B0", 35000: "#55A868", 40000: "#8172B2"}

# (a) the real cutoff with its band
g = T[(T["sample"] == "CA+WA") & (T.cutoff == 25_000) &
      (T.excl_shape == "asymmetric (below only)")].sort_values("excl_width")
ax[0].fill_between(g.excl_width / 1000, g.lo, g.hi, color="#C44E52", alpha=.18)
ax[0].plot(g.excl_width / 1000, g.excess, "o-", color="#C44E52", ms=5)
for _, r in g[g.sig].iterrows():
    ax[0].plot(r.excl_width / 1000, r.excess, "o", color="#C44E52", ms=11, mfc="none", mew=1.6)
ax[0].axhline(0, lw=.8, color="k")
ax[0].axvline(10, ls="--", lw=1.2, color="k")
ax[0].text(9.7, ax[0].get_ylim()[0] * .85 if ax[0].get_ylim()[0] < 0 else 0,
           "ceiling: fewer than 4 fit bins\nleft below the excluded region\n(state data stop at 10,000)",
           ha="right", va="bottom", fontsize=8)
ax[0].set_xlabel("excluded region below the cutoff (thousand tCO2e)")
ax[0].set_ylabel("excess mass")
ax[0].set_title("(a) CA+WA at 25,000   (rings = significant at 5%)", fontsize=10)

# (b) placebos
for cut in [25_000, 28_000, 30_000, 35_000, 40_000]:
    s = T[(T["sample"] == "CA+WA") & (T.cutoff == cut) &
          (T.excl_shape == "asymmetric (below only)")].sort_values("excl_width")
    if len(s):
        ax[1].plot(s.excl_width / 1000, s.excess, "o-", ms=4,
                   color=CL[cut], lw=2 if cut == 25_000 else 1,
                   label=f"{cut//1000}k" + (" (real)" if cut == 25_000 else ""))
ax[1].axhline(0, lw=.8, color="k")
ax[1].legend(frameon=False, fontsize=8, ncol=2)
ax[1].set_xlabel("excluded region below the cutoff (thousand tCO2e)")
ax[1].set_title("(b) the real cutoff against placebos\nsame sweep, same estimator", fontsize=10)

# (c) untruncated reference
g2 = T[(T["sample"] == "federal always-covered (unconstrained)") &
       (T.excl_shape == "asymmetric (below only)")].sort_values("excl_width")
ax[2].fill_between(g2.excl_width / 1000, g2.lo, g2.hi, color="#4C72B0", alpha=.18)
ax[2].plot(g2.excl_width / 1000, g2.excess, "o-", color="#4C72B0", ms=5)
for _, r in g2[g2.sig].iterrows():
    ax[2].plot(r.excl_width / 1000, r.excess, "o", color="#4C72B0", ms=11, mfc="none", mew=1.6)
ax[2].axhline(0, lw=.8, color="k")
ax[2].set_xlabel("excluded region below the cutoff (thousand tCO2e)")
ax[2].set_title("(c) federal always-covered at 25,000\nno truncation, so no ceiling", fontsize=10)

for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("f30 - widening the excluded region until the estimate dies", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "f30_excluded_region_sweep.png", dpi=150, bbox_inches="tight")
print("\nwrote t33_excluded_sweep.csv, t34_death_point.csv, f30_excluded_region_sweep.png")
