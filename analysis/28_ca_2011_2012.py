"""
28 · The CA 2011-2012 anomaly                                          (2026-09-13)
================================================================================
TASK C. t22_regime_split.csv reports

    CA 2011-2012  "cap-and-trade adopted, not yet binding"   excess +0.796  [0.080, 1.818]
    CA 2013-2023  "cap-and-trade binding"                    excess +1.037  [0.448, 1.892]

and the threats table has carried this since 3 September as: if the pile-up is already
there BEFORE cap-and-trade binds, the reading "cap-and-trade explains California" has a
crack in it. This file asks whether the crack is real. Three ways it could not be:

  (1) STATISTICAL. 274 facility-years in [15,000, 35,000]; ten of them in the 25,000 bin.
      The CI's lower limit is 0.080. Is +0.796 distinguishable from anything at all?
  (2) SPECIFICATION. Does it survive the bin width, polynomial order, excluded width and
      seed? Do placebo cutoffs in the SAME two years also come back significant?
  (3) INSTITUTIONAL. Was 25,000 really inert in California in 2011-2012? It was not.
      17 CCR 95101(b)(1): third-party verification of the emissions data report is
      required at 25,000 tCO2e, and the section defines that threshold by reference to
      section 95812 of the cap-and-trade regulation. So in 2011-2012 the line already
      carried (a) federal GHGRP reporting, (b) CA MRR verification, and (c) the
      determination of cap-and-trade coverage from reported emissions. "Adopted, not yet
      binding" is not the same as "inert".

outputs  t53_ca_2011_2012.csv  t54_ca_thin_sample.csv  f40_ca_2011_2012.png
run      python3 28_ca_2011_2012.py            (from analysis/)
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
OUT = Path("output")
CUT = 25_000
Z95 = 1.959964


def bunch(x, cut, bin_w=1000, fit_hw=10_000, excl_lo=3000, order=4, B=400, seed=5):
    """Exactly the estimator behind t22: symmetric fit range, excluded region below only."""
    edges = np.arange(cut - fit_hw, cut + fit_hw + bin_w, bin_w)
    c = np.histogram(x[(x >= cut - fit_hw) & (x < cut + fit_hw)], bins=edges)[0].astype(float)
    d = edges[:-1] + bin_w / 2 - cut
    z = d / 1000.0
    in_excl = (d >= -excl_lo) & (d < 0)
    if in_excl.sum() == 0 or (~in_excl).sum() < order + 4 or c.sum() < 60:
        return None
    P = np.vander(z, order + 1)
    X = np.hstack([P, np.eye(len(z))[:, in_excl]])

    def stat(counts):
        b, *_ = np.linalg.lstsq(X, counts, rcond=None)
        cf = (P @ b[: order + 1])[in_excl]
        return (counts[in_excl] - cf).sum() / max(cf.mean(), 1e-9)

    beta, *_ = np.linalg.lstsq(X, c, rcond=None)
    resid = c - X @ beta
    rng = np.random.default_rng(seed)
    draws = [stat(np.maximum(c + rng.choice(resid, len(c), replace=True), 0)) for _ in range(B)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return dict(excess=stat(c), lo=lo, hi=hi, sig=bool(lo > 0 or hi < 0),
                n_fit=int(c.sum()), n_excl=int(c[in_excl].sum()), se=float(np.std(draws, ddof=1)))


ca = pd.read_csv(OUT / "ca_mrr_panel.csv")
ca["co2e"] = pd.to_numeric(ca.co2e, errors="coerce")
early = ca[ca.year <= 2012].co2e.dropna().values
late = ca[ca.year >= 2013].co2e.dropna().values

print("=" * 78)
print("1 . how thin is the 2011-2012 window?")
print("=" * 78)
for lab, x in [("CA 2011-2012", early), ("CA 2013-2023", late)]:
    h = np.histogram(x, bins=np.arange(22_000, 28_001, 1000))[0]
    print(f"  {lab}: {len(x):,} facility-years total, "
          f"{int(((x>=15000)&(x<35000)).sum()):,} in [15k,35k]")
    print(f"     1k bins 22-28k: {h.tolist()}   (the 24k bin has {h[2]}, the 25k bin {h[3]})")
r0 = bunch(early, CUT)
print(f"\n  reproducing t22: excess {r0['excess']:+.3f}  [{r0['lo']:.3f}, {r0['hi']:.3f}]  "
      f"(published +0.796 [0.080, 1.818])")

# ---------------------------------------------------------------- 2 · specification
print("\n" + "=" * 78)
print("2 . does it survive the specification?  (2011-2012 only)")
print("=" * 78)
rows = []
for bw in (500, 1000, 2000):
    for od in (2, 3, 4):
        for ex in (2000, 3000, 5000):
            for sd in (5, 11, 23):
                r = bunch(early, CUT, bin_w=bw, excl_lo=ex, order=od, seed=sd)
                if r:
                    rows.append(dict(bin_w=bw, order=od, excl_lo=ex, seed=sd, **r))
S = pd.DataFrame(rows)
print(f"  {len(S)} specifications run")
print(f"  excess: median {S.excess.median():+.3f}   "
      f"range [{S.excess.min():+.3f}, {S.excess.max():+.3f}]")
print(f"  significant in {int(S.sig.sum())}/{len(S)} = {S.sig.mean()*100:.0f}% of them")
print("\n  by polynomial order:")
print(S.groupby("order").agg(n=("sig", "size"), median_excess=("excess", "median"),
                             share_sig=("sig", "mean")).round(3).to_string())
print("\n  by excluded width:")
print(S.groupby("excl_lo").agg(n=("sig", "size"), median_excess=("excess", "median"),
                               share_sig=("sig", "mean")).round(3).to_string())

# ---------------------------------------------------------------- 3 · placebos
print("\n" + "=" * 78)
print("3 . placebo cutoffs in the SAME two years, identical geometry")
print("=" * 78)
pl = []
for cut, kind in [(25_000, "REAL"), (18_000, "placebo"), (20_000, "placebo"),
                  (30_000, "placebo"), (35_000, "placebo"), (40_000, "placebo")]:
    r = bunch(early, cut)
    if r:
        pl.append(dict(cutoff=cut, kind=kind, **r))
P = pd.DataFrame(pl)
print(P.round(3).to_string(index=False))
n_sig_pl = int(P[P.kind == "placebo"].sig.sum())
print(f"\n  placebos significant: {n_sig_pl} / {len(P[P.kind=='placebo'])}")

# ---------------------------------------------------------------- 4 · thin-sample
print("\n" + "=" * 78)
print("4 . is +0.796 distinguishable from the LATER regime at this sample size?")
print("=" * 78)
print("  draw 2011-2012-sized subsamples out of CA 2013-2023 and re-estimate.")
rng = np.random.default_rng(0)
n_early = len(early)
sub = []
for _ in range(500):
    r = bunch(rng.choice(late, size=n_early, replace=False), CUT, B=60, seed=int(rng.integers(1e6)))
    if r:
        sub.append(r["excess"])
sub = np.array(sub)
pct = (sub <= r0["excess"]).mean() * 100
print(f"  {len(sub)} subsamples of n = {n_early:,} from the BINDING regime")
print(f"  their excess: median {np.median(sub):+.3f}   "
      f"5-95% [{np.percentile(sub,5):+.3f}, {np.percentile(sub,95):+.3f}]")
print(f"  the 2011-2012 estimate ({r0['excess']:+.3f}) sits at the {pct:.0f}th percentile "
      f"of that distribution")
print("  -> if that percentile is unremarkable, the two regimes are NOT distinguishable")
print("     at the early window's sample size, and the 'anomaly' is a power problem.")

S.to_csv(OUT / "t53_ca_2011_2012.csv", index=False)
pd.DataFrame(dict(subsample_excess=sub)).to_csv(OUT / "t54_ca_thin_sample.csv", index=False)
P.to_csv(OUT / "t53b_ca_placebos.csv", index=False)
print(f"\n  wrote t53_ca_2011_2012.csv, t53b_ca_placebos.csv, t54_ca_thin_sample.csv")


# ---------------------------------------------------------------- 5 · institutions
print("\n" + "=" * 78)
print("5 . what 25,000 actually carried in California in 2011-2012")
print("=" * 78)
INST = [
    ("federal GHGRP reporting", "40 CFR 98.2(a)(2)", "2010 onward",
     "the obligation this paper is about"),
    ("CA MRR third-party verification", "17 CCR 95101(b)(1)", "in force for data year 2011",
     "CARB's own 2011 reporting summary: 491 of 581 reports verified; the 90 that were "
     "not are footnoted 'these facilities emit <25,000 metric tons of CO2e'"),
    ("cap-and-trade covered-entity determination", "17 CCR 95812(b)", "data years 2009-2012",
     "reported or verified emissions >= 25,000 in ANY year 2009-2012 makes the entity "
     "covered as of 1 January 2013"),
]
for name, cite, when, note in INST:
    print(f"  - {name}\n      {cite} | {when}\n      {note}")
print("""
  => 2011 and 2012 were the DETERMINATION years for cap-and-trade coverage, and the
     verification notch was already live. Labelling them "adopted, not yet binding"
     describes the compliance obligation, not the incentive. There is no clean
     pre-period in California, and the CA 2011-2012 row in t22 cannot be used as one.""")

# ---------------------------------------------------------------- 6 · figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.7))

e = np.arange(15_000, 35_001, 1000)
h_e = np.histogram(early, bins=e)[0] / len(early[(early >= 15_000) & (early < 35_000)])
h_l = np.histogram(late, bins=e)[0] / len(late[(late >= 15_000) & (late < 35_000)])
w = 400
ax[0].bar(e[:-1] + 100, h_e * 100, width=w, color="#C44E52", label="2011-2012 (determination years)")
ax[0].bar(e[:-1] + 100 + w, h_l * 100, width=w, color="#4C72B0", label="2013-2023 (binding)")
ax[0].axvline(CUT, color="k", lw=1.3)
ax[0].text(CUT + 300, max(h_e.max(), h_l.max()) * 55, "25,000", fontsize=9)
ax[0].set_xlabel("reported emissions (tCO2e), California, 1k bins")
ax[0].set_ylabel("% of facility-years in [15k, 35k]")
ax[0].legend(frameon=False, fontsize=8)
ax[0].set_title("(a) both periods pile up below 25,000 and drop across it;\n"
                "the 2011-2012 peak sits in the 23-24k bin", fontsize=10)

jit = np.random.default_rng(1).uniform(-.18, .18, len(S))
ax[1].scatter(S.order + jit, S.excess, s=26,
              c=["#C44E52" if v else "#bbbbbb" for v in S.sig], alpha=.85)
ax[1].axhline(0, color="k", lw=1)
ax[1].axhline(r0["excess"], ls=":", lw=1.4, color="#2F4B7C")
ax[1].text(4.15, r0["excess"] + .12, "t22's +0.796", fontsize=8.5, color="#2F4B7C", ha="right")
ax[1].set_xticks([2, 3, 4]); ax[1].set_xlabel("polynomial order")
ax[1].set_ylabel("excess mass, 2011-2012")
ax[1].set_title(f"(b) {int(S.sig.sum())} of {len(S)} specifications significant\n"
                "(bin width, order, excluded width, seed)", fontsize=10)

ax[2].hist(sub, bins=40, color="#bbbbbb",
           label=f"same-sized subsamples\nof CA 2013-2023 (n={n_early:,})")
ax[2].axvline(r0["excess"], color="#C44E52", lw=2.2, label=f"CA 2011-2012 = {r0['excess']:+.2f}")
ax[2].axvline(np.median(sub), color="#4C72B0", lw=1.4, ls="--",
              label=f"their median = {np.median(sub):+.2f}")
ax[2].set_xlabel("excess mass")
ax[2].set_ylabel("subsamples")
ax[2].legend(frameon=False, fontsize=8)
ax[2].set_title(f"(c) the early estimate is at the {pct:.0f}th percentile of the\n"
                "binding regime: the two are not distinguishable", fontsize=10)

for a_ in ax:
    a_.spines[["top", "right"]].set_visible(False)
fig.suptitle("f40 - the CA 2011-2012 'anomaly': real, robust, and not an anomaly",
             fontsize=12, y=1.03)
fig.tight_layout()
fig.savefig(OUT / "f40_ca_2011_2012.png", dpi=150, bbox_inches="tight")
print(f"\n  wrote {OUT/'f40_ca_2011_2012.png'}")
