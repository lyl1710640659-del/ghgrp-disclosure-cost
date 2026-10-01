"""
10 · Where the right-hand excess in the gap figure comes from.

THE OBSERVATION
---------------
In f15 the distribution of gap = log(E_t / 25,000) sits noticeably to the RIGHT of
zero for threshold-bound facilities. Read naively that says facilities move up and
away from the threshold.

THE PROBLEM WITH READING IT THAT WAY
------------------------------------
gap measures distance from a FIXED line, so it is the sum of two things:
    gap_t = log(lag_em / 25,000)  +  log(E_t / lag_em)
            ^^^ where the facility STARTED     ^^^ how far it MOVED
The sample condition is lag_em in [20,000, 30,000] -- symmetric in tonnes. But the
DENSITY inside that band is not symmetric: threshold-bound facilities are truncated
below 25,000 (they do not report), so the band holds far more observations above the
line than below it. Any right-shift in the starting term passes straight through to
gap with no movement at all.

The always-covered group is the control for this: it is not truncated, so its
starting distribution inside the band is closer to centred. If the right-hand excess
tracks the starting distribution across the two groups, it is composition.

WHAT THIS SCRIPT DOES
---------------------
  1. decomposes gap into the starting term and the movement term, by group
  2. varies the window and shows the excess tracking the starting distribution
  3. two corrections that break the link, and what survives each:
       (a) tight start   -- lag_em within +/-2% of the cutoff, so both groups start
                            essentially at the line
       (b) reweighting   -- reweight always-covered to the threshold-bound starting
                            distribution (lag_em deciles), so the two groups are
                            compared at the same starting points
     If the excess is composition, (a) and (b) remove it. If something survives, that
     residual is the part worth interpreting as behaviour.

Outputs
  output/t19_gap_origin.csv
  output/f23_gap_origin.png
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
K = 25_000
BW = 0.05
BINS = np.arange(-0.6, 0.6 + BW, BW)
I0 = int(np.round((0 - BINS[0]) / BW)) - 1          # bin immediately BELOW zero
assert abs(BINS[I0 + 1]) < 1e-9, "zero must be a bin edge"

de = load_cached()
d = de.sort_values(["FacilityId", "year"]).copy()
d["lag_em"] = d.groupby("FacilityId").emissions.shift(1)
d = d[(d.emissions > 0) & (d.lag_em > 0)].copy()
d["growth"] = np.log(d.emissions / d.lag_em)
d["start"] = np.log(d.lag_em / K)                   # where it started, in gap units
d["gap"] = np.log(d.emissions / K)                  # gap = start + growth
GROUPS = [(True, "threshold-bound"), (False, "always-covered")]


def excess(counts):
    """Excess mass in the bin just below the cutoff, vs the mean of its neighbours."""
    nb = (counts[I0 - 1] + counts[I0 + 1]) / 2
    return (counts[I0] - nb) / nb if nb > 0 else np.nan


def hist(gap, wt=None):
    g = np.asarray(gap)
    m = (g >= BINS[0]) & (g <= BINS[-1])
    w = None if wt is None else np.asarray(wt)[m]
    return np.histogram(g[m], bins=BINS, weights=w)[0].astype(float)


def boot_ci(counts, n, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    p = counts / counts.sum()
    f = lambda c: (c[I0] - (c[I0-1] + c[I0+1]) / 2) / max((c[I0-1] + c[I0+1]) / 2, 1e-9)
    return np.percentile([f(rng.multinomial(n, p).astype(float)) for _ in range(B)], [2.5, 97.5])


# =============================================================================
# 1 · decomposition, and the window scan
# =============================================================================
rows = []
for w in [1_000, 2_000, 3_000, 5_000, 7_500, 10_000]:
    sub = d[d.lag_em.between(K - w, K + w)]
    for tb, lab in GROUPS:
        s = sub[sub.threshold_bound == tb]
        if len(s) < 150:
            continue
        rows.append(dict(
            window=w, group=lab, n=len(s),
            start_below=round((s.start < 0).mean(), 4),      # started under the line
            start_mean=round(s.start.mean(), 4),
            move_mean=round(s.growth.mean(), 4),             # how far it actually moved
            move_median=round(s.growth.median(), 4),
            gap_above=round((s.gap > 0).mean(), 4),          # ended above the line
            gap_mean=round(s.gap.mean(), 4),
            excess=round(excess(hist(s.gap)), 4)))
S = pd.DataFrame(rows)
pd.set_option("display.width", 240)
print("=== 1 · gap = start + movement.  Does 'ended above' just track 'started above'? ===")
print(S.to_string(index=False))
print("\n corr(share started below, share ended above) across all cells: "
      f"{S.start_below.corr(S.gap_above):+.3f}")
print(" the movement term, for comparison — median log growth is ~0 everywhere above")

# =============================================================================
# 2 · correction (a): tight start
# =============================================================================
print("\n=== 2 · correction (a): start within +/-2% of the cutoff (24,500–25,500) ===")
tight = d[d.lag_em.between(24_500, 25_500)]
trows = []
for tb, lab in GROUPS:
    s = tight[tight.threshold_bound == tb]
    c = hist(s.gap)
    lo, hi = boot_ci(c, int(c.sum()))
    trows.append(dict(spec="tight start", group=lab, n=len(s),
                      start_below=round((s.start < 0).mean(), 4),
                      gap_above=round((s.gap > 0).mean(), 4),
                      excess=round(excess(c), 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                      significant=bool(lo > 0 or hi < 0)))
Tt = pd.DataFrame(trows)
print(Tt.to_string(index=False))

# =============================================================================
# 3 · correction (b): reweight always-covered onto the threshold-bound start
# =============================================================================
print("\n=== 3 · correction (b): reweight always-covered to the threshold-bound "
      "starting distribution (lag_em deciles, window +/-5,000) ===")
W = 5_000
sub = d[d.lag_em.between(K - W, K + W)].copy()
edges = np.quantile(sub.loc[sub.threshold_bound, "lag_em"], np.linspace(0, 1, 11))
edges[0], edges[-1] = -np.inf, np.inf
sub["dec"] = pd.cut(sub.lag_em, edges, labels=False, duplicates="drop")
tb_share = sub[sub.threshold_bound].groupby("dec").size()
tb_share = tb_share / tb_share.sum()
ac = sub[~sub.threshold_bound].copy()
ac_share = ac.groupby("dec").size()
ac_share = ac_share / ac_share.sum()
ac["wt"] = ac.dec.map(tb_share / ac_share).fillna(0.0)

rrows = []
tbs = sub[sub.threshold_bound]
c_tb = hist(tbs.gap); lo, hi = boot_ci(c_tb, int(c_tb.sum()))
rrows.append(dict(spec="+/-5,000, unweighted", group="threshold-bound", n=len(tbs),
                  start_below=round((tbs.start < 0).mean(), 4),
                  gap_above=round((tbs.gap > 0).mean(), 4),
                  excess=round(excess(c_tb), 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4)))
c_ac = hist(ac.gap); lo, hi = boot_ci(c_ac, int(c_ac.sum()))
rrows.append(dict(spec="+/-5,000, unweighted", group="always-covered", n=len(ac),
                  start_below=round((ac.start < 0).mean(), 4),
                  gap_above=round((ac.gap > 0).mean(), 4),
                  excess=round(excess(c_ac), 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4)))
c_rw = hist(ac.gap, ac.wt)
n_eff = int(ac.wt.sum() ** 2 / (ac.wt ** 2).sum())            # Kish effective n
lo, hi = boot_ci(c_rw, n_eff)
rrows.append(dict(spec="+/-5,000, REWEIGHTED", group="always-covered", n=n_eff,
                  start_below=round(np.average(ac.start < 0, weights=ac.wt), 4),
                  gap_above=round(np.average(ac.gap > 0, weights=ac.wt), 4),
                  excess=round(excess(c_rw), 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4)))
R = pd.DataFrame(rrows)
print(R.to_string(index=False))

pd.concat([S.assign(spec="window scan"), Tt, R], ignore_index=True) \
  .to_csv(OUT / "t19_gap_origin.csv", index=False)

# =============================================================================
# figure
# =============================================================================
plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8))
CB, CA = "#4C72B0", "#C44E52"

# (a) started above vs ended above — the one-to-one line
for lab, c in [("threshold-bound", CB), ("always-covered", CA)]:
    s = S[S.group == lab]
    ax[0].scatter(1 - s.start_below, s.gap_above, s=46, color=c, alpha=.9, label=lab)
lim = [.45, .70]
ax[0].plot(lim, lim, color="#777", lw=1, ls="--", label="45°: no movement at all")
ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].set_xlabel("share that STARTED above 25,000")
ax[0].set_ylabel("share that ENDED above 25,000")
ax[0].set_title("(a) Ending above the line is where the\nfacility started, not where it moved",
                fontsize=10)
ax[0].legend(fontsize=8.5)

# (b) the gap distribution, raw vs reweighted
mid = BINS[:-1] + BW / 2
ax[1].plot(mid, c_tb / c_tb.sum(), "o-", ms=4, color=CB, label=f"threshold-bound (n={len(tbs):,})")
ax[1].plot(mid, c_ac / c_ac.sum(), "o-", ms=4, color=CA, alpha=.5,
           label=f"always-covered, raw (n={len(ac):,})")
ax[1].plot(mid, c_rw / c_rw.sum(), "o-", ms=4, color=CA,
           label="always-covered, reweighted to\nthe threshold-bound start")
ax[1].axvline(0, color="k", ls="--", lw=1.2)
ax[1].axvspan(BINS[I0], BINS[I0 + 1], color="#999", alpha=.2, zorder=0)
ax[1].set_xlabel("gap = log(emissions / 25,000)"); ax[1].set_ylabel("share of facility-years")
ax[1].set_title("(b) Matching the starting distribution\nmoves the two curves together", fontsize=10)
ax[1].legend(fontsize=7.5)

# (c) decomposition: start vs movement, +/-5,000
ax[2].hist(tbs.start, bins=np.arange(-.25, .21, .02), color=CB, alpha=.55,
           density=True, label="where it started")
ax[2].hist(tbs.growth[(tbs.growth > -.6) & (tbs.growth < .6)], bins=np.arange(-.6, .62, .04),
           histtype="step", lw=2, color="#333", density=True, label="how far it moved")
ax[2].axvline(0, color="k", ls="--", lw=1.2)
ax[2].set_xlabel("log points"); ax[2].set_ylabel("density")
ax[2].set_title("(c) Threshold-bound, ±5,000 window:\nthe start is off-centre, the move is not",
                fontsize=10)
ax[2].legend(fontsize=8.5)

plt.tight_layout()
plt.savefig(OUT / "f23_gap_origin.png", bbox_inches="tight")
print("\nwrote t19_gap_origin.csv, f23_gap_origin.png")
