"""
22 · Minimum detectable effects                                    (2026-09-13)
================================================================================
WHY THIS FILE EXISTS
--------------------
Almost every headline in this project is a null. There is no bunching at 25,000 in the
federal panel (node 4, Marx), none in WA 2012-2022 (the one window where 25,000 means
disclosure and nothing else), the newly-arrived hypothesis fails on both limbs, and the
near/far growth-rate difference is indistinguishable from zero.

A paper built on nulls has to answer one question before any of them count:

        HOW BIG WOULD THE EFFECT HAVE HAD TO BE FOR US TO SEE IT?

Without that number a reader cannot tell "there is no response" from "there is no
power", and every null in the paper is empty. Nobody has asked for this yet. It is
still the first thing a referee or a committee will ask.

WHAT IS DONE HERE
-----------------
  Part 1  BUNCHING. Inject a known amount of bunching into the REAL data -- relocate a
          share of the facility-years sitting just above the notch into the band just
          below it -- re-run the estimator, and record how often it comes back
          significant. The minimum detectable effect is the injected excess mass at
          which that detection rate reaches 80%.

  Part 2  GROWTH RATES. Analytic, from the standard errors we already have:
          MDE = (z_0.975 + z_0.80) * SE = 2.80 * SE.

The injection is into real data, not simulated data: the counterfactual density, the
truncation, the sample sizes and the bin-to-bin noise are all the ones we actually face.

outputs  t40_mde_bunching.csv  t41_mde_growth.csv  f34_mde.png
run      python3 22_power_mde.py [sample]          (from analysis/)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, ".")
from ghgrp_load import load_cached

OUT = Path("output")
CUT = 25_000
BIN = 1000
MIN_BELOW_FIT = 4
BELOW_BINS, ABOVE_BINS = 14, 20           # same geometry as 17_excluded_region_sweep.py
W_PEAK = 5_000                            # the excluded-region width that peaked in f30
ABOVE_SRC = 7_500                         # donor region above the notch: [cut, cut+7500)
Z95, Z80 = 1.959964, 0.841621


def counts(x, cut, fit_lo, fit_hi):
    edges = np.arange(fit_lo, fit_hi + BIN, BIN)
    return np.histogram(x[(x >= fit_lo) & (x < fit_hi)], bins=edges)[0].astype(float), edges


def design(edges, cut, excl_lo, order=4):
    z = (edges[:-1] + BIN / 2 - cut) / 1000.0
    d = z * 1000
    in_excl = (d >= -excl_lo) & (d < 0)
    P = np.vander(z, order + 1)
    X = np.hstack([P, np.eye(len(z))[:, in_excl]])
    return P, X, in_excl


def excess_from_counts(c, P, X, in_excl, order=4):
    b, *_ = np.linalg.lstsq(X, c, rcond=None)
    cf = (P @ b[: order + 1])[in_excl]
    return (c[in_excl] - cf).sum() / max(cf.mean(), 1e-9)


def boot_se(x, cut, fit_lo, fit_hi, excl_lo, B=400, seed=0):
    """Residual bootstrap SE of the excess-mass estimate, at the observed data."""
    c, edges = counts(x, cut, fit_lo, fit_hi)
    P, X, in_excl = design(edges, cut, excl_lo)
    beta, *_ = np.linalg.lstsq(X, c, rcond=None)
    resid = c - X @ beta
    rng = np.random.default_rng(seed)
    draws = [excess_from_counts(np.maximum(c + rng.choice(resid, len(c), replace=True), 0),
                                P, X, in_excl) for _ in range(B)]
    return float(np.std(draws, ddof=1)), float(excess_from_counts(c, P, X, in_excl)), c.sum()


def inject(x, cut, beta, rng):
    """Relocate a share `beta` of the facility-years in [cut, cut+ABOVE_SRC) into the
    band [cut-W_PEAK, cut). This is what avoiding the notch looks like: firms that would
    have been just above it end up just below."""
    x = x.copy()
    donors = np.where((x >= cut) & (x < cut + ABOVE_SRC))[0]
    if len(donors) == 0:
        return x, 0
    k = rng.binomial(len(donors), beta)
    if k == 0:
        return x, 0
    pick = rng.choice(donors, size=k, replace=False)
    x[pick] = rng.uniform(cut - W_PEAK, cut, size=k)
    return x, k


def mde_bunching(x, name, cut=CUT, betas=(0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.14, 0.20, 0.30),
                 R=200, seed=0):
    fit_lo, fit_hi = cut - BELOW_BINS * BIN, cut + ABOVE_BINS * BIN
    se0, b0, nfit = boot_se(x, cut, fit_lo, fit_hi, W_PEAK, seed=seed)
    crit = Z95 * se0
    _, edges = counts(x, cut, fit_lo, fit_hi)
    P, X, in_excl = design(edges, cut, W_PEAK)
    rng = np.random.default_rng(seed + 1)
    n_donor = int(((x >= cut) & (x < cut + ABOVE_SRC)).sum())
    rows = []
    # TWO power definitions, because they answer different questions and, in a sample
    # whose baseline estimate is already non-zero, they are not the same question:
    #   power_vs_zero      P(|b| > 1.96 SE)            the textbook MDE
    #   power_vs_baseline  P(b > b0 + 1.96 SE)         "how much MORE bunching than
    #                                                   whatever is already here would
    #                                                   we have seen?"
    # For CA+WA the baseline is 0.863 and already clears 1.96*SE, so power_vs_zero is 1
    # at zero injection and tells the reader nothing. The headline MDE is the second one.
    for beta in betas:
        bs, h0, hb, moved = [], 0, 0, []
        for _ in range(R):
            xi, k = inject(x, cut, beta, rng)
            ci, _ = counts(xi, cut, fit_lo, fit_hi)
            bi = excess_from_counts(ci, P, X, in_excl)
            bs.append(bi); moved.append(k)
            h0 += int(abs(bi) > crit)
            hb += int(bi > b0 + crit)
        rows.append(dict(sample=name, beta=beta, n_moved=int(np.mean(moved)),
                         mean_excess=round(float(np.mean(bs)), 3),
                         power_vs_zero=round(h0 / R, 3),
                         power=round(hb / R, 3)))
    T = pd.DataFrame(rows)
    T["baseline_excess"] = round(b0, 3)
    T["se"] = round(se0, 3)
    T["n_fit"] = int(nfit)
    T["n_donor_above"] = n_donor
    # interpolate the excess mass at which power crosses 0.80
    # MDE is read off the INCREMENT: excess above the baseline
    p, e = T.power.values, (T.mean_excess.values - b0)
    mde = np.nan
    for i in range(1, len(p)):
        if p[i - 1] < 0.80 <= p[i]:
            w = (0.80 - p[i - 1]) / max(p[i] - p[i - 1], 1e-9)
            mde = e[i - 1] + w * (e[i] - e[i - 1])
            break
    T["mde_increment_at_80pct"] = round(mde, 3) if np.isfinite(mde) else np.nan
    # the number a reader actually wants: the LEVEL of excess mass this sample
    # could have detected with 80% power
    T["mde_level_at_80pct"] = round(b0 + mde, 3) if np.isfinite(mde) else np.nan
    return T


# =============================================================================
# samples
# =============================================================================
ca = pd.read_csv(OUT / "ca_mrr_panel.csv")
wa = pd.read_csv(OUT / "wa_ghg_panel.csv"); wa = wa[wa["Reporter Type"] == "Facility"]
for d in (ca, wa):
    d["co2e"] = pd.to_numeric(d.co2e, errors="coerce")
ca_v = ca.co2e.dropna().values; ca_v = ca_v[ca_v > 0]
wa_v = wa.co2e.dropna().values; wa_v = wa_v[wa_v > 0]
wa_clean = wa[wa.year <= 2022].co2e.dropna().values; wa_clean = wa_clean[wa_clean > 0]

de = load_cached()
ac = de.loc[de.always_covered & (de.emissions > 0), "emissions"].values

SAMPLES = {
    "CA + WA": np.concatenate([ca_v, wa_v]),
    "CA only": ca_v,
    "WA 2012-2022 (disclosure only)": wa_clean,
    "federal always-covered": ac,
}

pick = sys.argv[1:] or list(SAMPLES)
res = []
for k in pick:
    T = mde_bunching(SAMPLES[k], k, seed=abs(hash(k)) % 1000)
    res.append(T)
    print(f"\n--- {k} ---")
    print(T[["beta", "n_moved", "mean_excess", "power_vs_zero", "power"]].to_string(index=False))
    print(f"  n in fit range {T.n_fit.iloc[0]:,} | donors just above the notch "
          f"{T.n_donor_above.iloc[0]:,} | bootstrap SE {T.se.iloc[0]:.3f}")
    print(f"  >> MDE at 80% power: increment {T.mde_increment_at_80pct.iloc[0]} "
          f"above baseline {T.baseline_excess.iloc[0]}  ==>  this sample could have\n     detected an excess mass of {T.mde_level_at_80pct.iloc[0]} or more")

M = pd.concat(res, ignore_index=True)
old = OUT / "t40_mde_bunching.csv"
if old.exists():
    prev = pd.read_csv(old)
    M = pd.concat([prev[~prev["sample"].isin(M["sample"])], M], ignore_index=True)
M.to_csv(old, index=False)
print("\nwrote t40_mde_bunching.csv")


# =============================================================================
# PART 2 · the growth-rate nulls, analytically
# =============================================================================
# For a two-sample mean difference there is no need to simulate:
#     MDE = (z_0.975 + z_0.80) * SE = 2.80 * SE
# The standard errors are the ones already reported, so this is just arithmetic on
# tables we have. It covers the other three nulls in the paper.
K = Z95 + Z80
rows = []

T17 = pd.read_csv(OUT / "t17_growth_window_mean.csv")
g = T17[(T17.cutoff == 25_000)]
for far in g.far_def.unique():
    s = g[g.far_def == far]
    for spec, dcol, lo, hi in [("raw", "diff__raw", "lo__raw", "hi__raw"),
                               ("facility FE", "diff__facility FE",
                                "lo__facility FE", "hi__facility FE")]:
        se = ((s[hi] - s[lo]) / (2 * Z95)).median()
        rows.append(dict(null="growth, near vs away at 25,000", detail=f"{far}, {spec}",
                         estimate=round(s[dcol].median(), 4), se=round(se, 4),
                         mde=round(K * se, 4),
                         unit="log points of annual growth"))

T32 = pd.read_csv(OUT / "t32b_entry_level_split.csv")
r = T32[T32["group"].str.contains("genuine", case=False)].iloc[0]
rows.append(dict(null="newly arrived: post-entry slowdown", detail="genuine crossings",
                 estimate=round(r.slowdown, 4), se=round(r.se, 4),
                 mde=round(K * r.se, 4), unit="log points of annual growth"))

T30 = pd.read_csv(OUT / "t30_newarrival_by_bin.csv")
a = T30[(T30.group == "TB") & (T30.bin == "25-30k")].iloc[0]
b = T30[(T30.group == "AC") & (T30.bin == "25-30k")].iloc[0]
se = np.sqrt(a.se ** 2 + b.se ** 2)
rows.append(dict(null="newly arrived: share new at 25-30k, TB minus AC",
                 detail="25-30k bin", estimate=round(a.share_new - b.share_new, 4),
                 se=round(se, 4), mde=round(K * se, 4), unit="share (pp/100)"))

G = pd.DataFrame(rows)
G.to_csv(OUT / "t41_mde_growth.csv", index=False)
print("\n=== PART 2 · minimum detectable effects for the growth-rate nulls ===")
print(G.to_string(index=False))
print("\n Read: the estimate is what we found; the MDE is the smallest true effect this")
print(" design would have caught 80% of the time. A null is informative only against")
print(" effects LARGER than its MDE.")


# =============================================================================
# figure
# =============================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = pd.read_csv(OUT / "t40_mde_bunching.csv")
ORD = ["federal always-covered", "WA 2012-2022 (disclosure only)", "CA + WA", "CA only"]
A = A[A["sample"].isin(ORD)]
COL = {"federal always-covered": "#4C72B0", "WA 2012-2022 (disclosure only)": "#55A868",
       "CA + WA": "#C44E52", "CA only": "#8172B2"}

fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.7))

for k in ORD:
    s = A[A["sample"] == k].sort_values("mean_excess")
    if len(s):
        ax[0].plot(s.mean_excess, s.power * 100, "o-", ms=5, color=COL[k], label=k)
ax[0].axhline(80, ls="--", lw=1.1, color="k")
ax[0].text(0.02, 82, "80% power", fontsize=8.5, transform=ax[0].get_yaxis_transform())
ax[0].set_xlabel("excess mass present in the data")
ax[0].set_ylabel("% of replications detected")
ax[0].set_xlim(-0.6, 3.0)
ax[0].legend(frameon=False, fontsize=8)
ax[0].set_title("(a) power curves, bunching injected into the real data", fontsize=10)

M = A.drop_duplicates("sample").set_index("sample").loc[ORD]
yy = np.arange(len(ORD))[::-1]
ax[1].barh(yy, M.mde_level_at_80pct, .55, color=[COL[k] for k in ORD], alpha=.85)
for y_, v, n in zip(yy, M.mde_level_at_80pct, M.n_fit):
    ax[1].text(v + .04, y_, f"{v:.2f}   (n={int(n):,})", va="center", fontsize=8.5)
ax[1].axvline(1.219, ls=":", lw=1.4, color="#8172B2")
ax[1].text(1.24, -0.62, "CA observed 1.22", fontsize=8.5, color="#8172B2")
ax[1].set_yticks(yy)
ax[1].set_yticklabels([k.replace(" (disclosure only)", "\n(disclosure only)") for k in ORD],
                      fontsize=8.5)
ax[1].set_xlabel("smallest excess mass detectable at 80% power")
ax[1].set_xlim(0, 2.6)
ax[1].set_title("(b) WA's null clears the bar: it could have seen\n"
                "an effect the size CA shows", fontsize=10)

G = pd.read_csv(OUT / "t41_mde_growth.csv")
G = G[G.unit.str.startswith("log")]
yg = np.arange(len(G))[::-1]
ax[2].barh(yg, G.mde, .5, color="#999999", alpha=.7, label="MDE at 80% power")
ax[2].plot(G.estimate.abs(), yg, "o", ms=8, color="#C44E52", label="|estimate|")
ax[2].set_yticks(yg)
ax[2].set_yticklabels([d[:26] for d in G.detail], fontsize=8)
ax[2].set_xlabel("log points of annual growth")
ax[2].legend(frameon=False, fontsize=8.5)
ax[2].set_title("(c) the growth-rate nulls: what we found\n"
                "against what we could have found", fontsize=10)

for a_ in ax:
    a_.spines[["top", "right"]].set_visible(False)
fig.suptitle("f34 - minimum detectable effects: how big would it have had to be?",
             fontsize=12, y=1.03)
fig.tight_layout()
fig.savefig(OUT / "f34_mde.png", dpi=150, bbox_inches="tight")
print("\nwrote t40, t41 and f34_mde.png")
