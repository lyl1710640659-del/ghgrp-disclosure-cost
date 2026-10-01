"""
15 · Three diagnostics on f14, the figure the truncation argument rests on.

WHY
---
f14 is a POOLED CROSS-SECTIONAL density of log emissions over all facility-years
2010-2023. It is not, and was never meant to be, the year-on-year growth analysis --
that is f22 / f12 / f16. f14 answers the 6 August note "bunching just above the
reporting threshold exists across sectors"; the growth-rate notes are answered
elsewhere. Keeping those two straight matters, because the two lines of work have
been confused before.

Three things this script checks, all of which change how f14 should be presented.

1. BIN ALIGNMENT. f14 uses bins = arange(7, 17, 0.25). log(25,000) = 10.1266 lands
   almost exactly in the MIDDLE of the bin [10.00, 10.25] = [22,026, 28,283] tonnes.
   The bin at the threshold therefore contains facilities on BOTH sides of it -- it
   blurs the one place the figure is asking the reader to look. Panel (a) redoes it
   with bins aligned so the threshold falls on a bin EDGE.

2. WHAT THE PLACEBO GROUP ACTUALLY IS. always-covered is 47.4% subpart D (power,
   modal size ~1.36m tonnes) and 47.5% subpart HH (municipal landfills, modal size
   ~53k tonnes) -- two nearly equal halves with modes two orders of magnitude apart.
   Its density near 25,000 is therefore essentially the landfill distribution.
   Panel (b) shows the mixture.

3. THE ROBUSTNESS THAT HAS NEVER BEEN RUN. ghgrp_load.py flags HH as "only partially
   always-covered -- read the rule text", and that classification has never been
   verified against 40 CFR 98 Table A-3. Since HH is half the placebo group, the
   argument needs to survive dropping it. Panel (c) compares the full and strict
   definitions across the cutoff.

Outputs
  output/f27_f14_diagnostics.png
  output/t28_f14_diagnostics.csv
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
K = np.log(25_000)
de = load_cached()

# bins aligned so the threshold is a bin EDGE, not a bin centre
BW = 0.25
B = np.arange(K - 3.25, K + 6.25, BW)
MID = B[:-1] + BW / 2
J = int(np.searchsorted(B, K))          # first bin whose LEFT edge is the threshold

GROUPS = [("threshold-bound", de.threshold_bound, "#4C72B0"),
          ("always-covered", de.always_covered, "#C44E52"),
          ("always-covered STRICT (no HH)", de.always_covered_strict, "#8172B2")]

rows = []
H = {}
for lab, m, _ in GROUPS:
    s = np.log(de.loc[m & (de.emissions > 0), "emissions"])
    h, _ = np.histogram(s, bins=B, density=True)
    H[lab] = h
    rows.append(dict(group=lab, n=int(m.sum()),
                     share_below_cutoff=round((s < K).mean(), 4),
                     bin_below=round(h[J - 1], 4), bin_above=round(h[J], 4),
                     ratio_across=round(h[J] / max(h[J - 1], 1e-9), 3),
                     bin_two_below=round(h[J - 2], 4),
                     falloff_below=round(1 - h[J - 2] / max(h[J - 1], 1e-9), 3)))
T = pd.DataFrame(rows)
T.to_csv(OUT / "t28_f14_diagnostics.csv", index=False)
pd.set_option("display.width", 200)
print("=== density across the cutoff, bins ALIGNED to the threshold ===")
print(T.to_string(index=False))
print("\n  ratio_across  = density just above / just below the cutoff")
print("  falloff_below = how much the density drops going one bin FURTHER below")
print("  The fall-off is the truncation fingerprint, not the ratio: both groups rise")
print("  through this region because it is on the way up to the mode.")

# ---------------------------------------------------------------- figure
plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.9))

# (a) f14's left panel, redone with aligned bins, zoomed on the threshold region
for lab, m, c in GROUPS[:2]:
    ax[0].step(MID, H[lab], where="mid", color=c, lw=1.8, label=lab)
    ax[0].fill_between(MID, H[lab], step="mid", color=c, alpha=.25)
ax[0].axvline(K, color="k", ls="--", lw=1.5)
ax[0].axvspan(10.00, 10.25, color="#999", alpha=.25, zorder=0)
ax[0].annotate("f14's original bin [22,026, 28,283]\nstraddles the threshold",
               (10.125, 0.44), xytext=(10.9, 0.47), fontsize=8, color="#444",
               arrowprops=dict(arrowstyle="->", color="#444", lw=1))
ax[0].set_xlim(K - 2.2, K + 3.2)
ax[0].set_xlabel("log emissions"); ax[0].set_ylabel("density")
ax[0].set_title("(a) Same comparison, bins aligned so the cutoff\nis a bin EDGE",
                fontsize=10)
ax[0].legend(fontsize=8.5)

# (b) what always-covered is made of
ac = de[de.always_covered & (de.emissions > 0)].copy()
ac["L"] = np.log(ac.emissions)
for sub, nm, c in [("HH", "HH municipal landfills", "#55A868"),
                   ("D", "D power generation", "#C44E52")]:
    mm = ac.subpart_set.map(lambda x: sub in x)
    h, _ = np.histogram(ac.loc[mm, "L"], bins=B, density=True)
    ax[1].step(MID, h, where="mid", lw=1.8, color=c,
               label=f"{nm}  ({mm.mean():.0%} of the group)")
    ax[1].fill_between(MID, h, step="mid", color=c, alpha=.20)
ax[1].step(MID, H["always-covered"], where="mid", color="#333", lw=2, ls="--",
           label="always-covered, all")
ax[1].axvline(K, color="k", ls="--", lw=1.5)
ax[1].set_xlabel("log emissions"); ax[1].set_ylabel("density")
ax[1].set_title("(b) The placebo group is two populations.\n"
                "Near the cutoff it is essentially landfills.", fontsize=10)
ax[1].legend(fontsize=7.5)

# (c) does the argument survive dropping HH?
lo, hi = K - 1.2, K + 1.6
for lab, m, c in [GROUPS[1], GROUPS[2]]:
    ax[2].step(MID, H[lab], where="mid", color=c, lw=2, label=lab)
ax[2].axvline(K, color="k", ls="--", lw=1.5)
ax[2].set_xlim(lo, hi)
sub = (MID >= lo) & (MID <= hi)
ax[2].set_ylim(0, max(H["always-covered"][sub].max(), H[GROUPS[2][0]][sub].max()) * 1.25)
r_full = T.loc[T.group == "always-covered", "ratio_across"].iloc[0]
r_str = T.loc[T.group.str.contains("STRICT"), "ratio_across"].iloc[0]
ax[2].set_xlabel("log emissions"); ax[2].set_ylabel("density")
ax[2].set_title(f"(c) Dropping HH: the placebo gets FLATTER at the line\n"
                f"ratio across the cutoff {r_full:.2f} → {r_str:.2f}", fontsize=10)
ax[2].legend(fontsize=8.5)

plt.tight_layout()
plt.savefig(OUT / "f27_f14_diagnostics.png", bbox_inches="tight")
print("\nwrote t28_f14_diagnostics.csv, f27_f14_diagnostics.png")
