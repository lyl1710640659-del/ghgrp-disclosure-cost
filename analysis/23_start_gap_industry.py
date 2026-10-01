"""
23 · Is the start-point gap industry, or is it economic?            (John's item 3)
================================================================================
John, 3 September 2026:

    "Whether the gap in start points is for industry or economic reasons is an
     empirical question. This could be explored descriptively by looking at INDUSTRY
     COMPOSITION and firm/sector characteristics (revenues, employment,
     expenditures). If there are major differences in industry composition and/or
     firm/sector characteristics, you have a likely answer."

This is the one item of the nine that had not been touched. Its first half -- industry
composition -- needs nothing we do not already have: NAICS is in the panel.

WHAT THE START-POINT GAP IS
---------------------------
Script 10 split the gap figure into two terms:

    gap_t = log(lag_em / 25,000)   +   log(E_t / lag_em)
            ^^^ where it STARTED       ^^^ how far it MOVED

and found the right-hand excess is entirely the first term: inside a symmetric
[20,000, 30,000] band, threshold-bound facilities sit higher relative to the line than
always-covered ones, because they are truncated below it. This file asks WHY the two
groups start in different places, and splits that into

    BETWEEN industries   the two groups are in different sectors, and sectors differ
                         in typical size relative to 25,000   -> "industry"
    WITHIN industries    even in the same sector, the two groups sit differently
                         -> "economic", and the part revenue/employment would speak to

Two methods, deliberately: a DFL reweighting (non-parametric, reweight always-covered
to the threshold-bound industry mix) and an Oaxaca-Blinder decomposition. If they
disagree, say so rather than picking the friendlier one.

THE CHECK THAT COMES FIRST
--------------------------
always_covered is defined by SUBPART, and subparts map onto industries -- a
Table A-3 facility is a power plant, a refinery, a cement works, a landfill. If the two
groups barely overlap in NAICS, no decomposition is identified and the honest answer is
"industry and group are the same variable here". Section 1 checks that before anything
else is computed.

outputs  t42_start_gap_support.csv  t43_start_gap_decomp.csv  t44_start_gap_by_naics.csv
         f35_start_gap_industry.png
run      python3 23_start_gap_industry.py        (from analysis/)
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
LO, HI = 20_000, 30_000          # the symmetric band script 10 uses
MIN_CELL = 30                    # a NAICS3 cell needs this many in BOTH groups to count

de = load_cached()
de, tr = add_transitions(de)
t = tr[(tr.lag_em >= LO) & (tr.lag_em <= HI) & (tr.emissions > 0)].copy()
t["grp"] = np.where(t.threshold_bound, "TB", "AC")
t["start"] = np.log(t.lag_em / CUT)          # where it started, relative to the line
t["gap_above"] = (t.emissions >= CUT).astype(float)
t["naics3"] = pd.to_numeric(t.naics3, errors="coerce")
t = t.dropna(subset=["naics3", "start"])
t["naics3"] = t.naics3.astype(int)

A = t[t.grp == "TB"]
B = t[t.grp == "AC"]
print(f"window [{LO:,}, {HI:,}] on the base year: TB {len(A):,}   AC {len(B):,}")
print(f"raw start-point gap  mean(start|TB) - mean(start|AC) = "
      f"{A.start.mean() - B.start.mean():+.4f}")
print(f"raw gap_above        {A.gap_above.mean():.4f} vs {B.gap_above.mean():.4f} "
      f"= {A.gap_above.mean() - B.gap_above.mean():+.4f}")


# =============================================================================
# 1 · common support -- does the decomposition even exist?
# =============================================================================
ct = pd.crosstab(t.naics3, t.grp)
for c in ("TB", "AC"):
    if c not in ct:
        ct[c] = 0
ct["both"] = (ct.TB >= MIN_CELL) & (ct.AC >= MIN_CELL)
ct["tb_share"] = ct.TB / ct.TB.sum()
ct["ac_share"] = ct.AC / ct.AC.sum()
sup = pd.DataFrame({
    "measure": ["NAICS3 cells in the window",
                f"cells with >= {MIN_CELL} in BOTH groups",
                "TB observations inside those cells",
                "AC observations inside those cells"],
    "value": [len(ct), int(ct.both.sum()),
              f"{ct.loc[ct.both, 'TB'].sum() / ct.TB.sum():.1%}",
              f"{ct.loc[ct.both, 'AC'].sum() / ct.AC.sum():.1%}"]})
sup.to_csv(OUT / "t42_start_gap_support.csv", index=False)
print("\n=== 1 · common support in NAICS3 ===")
print(sup.to_string(index=False))
top = ct.sort_values("TB", ascending=False).head(10)[["TB", "AC", "tb_share", "ac_share", "both"]]
print("\n top 10 NAICS3 by TB count (shares are within-group):")
print(top.assign(tb_share=(top.tb_share * 100).round(1),
                 ac_share=(top.ac_share * 100).round(1)).to_string())


# =============================================================================
# 2 · the four cells that DO overlap -- the only place the question is identified
# =============================================================================
# With 97.7% of always-covered observations but only 9.7% of threshold-bound ones inside
# cells where both groups are present, reweighting one group to the other's industry mix
# would mean extrapolating always-covered behaviour into industries where it has between
# zero and sixteen observations. That is not a decomposition, it is an extrapolation.
# What CAN be done is a comparison inside the cells that genuinely overlap.
both_cells = ct.index[ct.both].tolist()
print(f"\n=== 2 · inside the {len(both_cells)} overlapping NAICS3 cells: {both_cells} ===")
rows = []
for n in both_cells:
    a, b = A[A.naics3 == n], B[B.naics3 == n]
    d_start = a.start.mean() - b.start.mean()
    se_start = np.sqrt(a.start.var(ddof=1) / len(a) + b.start.var(ddof=1) / len(b))
    d_above = a.gap_above.mean() - b.gap_above.mean()
    se_above = np.sqrt(a.gap_above.var(ddof=1) / len(a) + b.gap_above.var(ddof=1) / len(b))
    rows.append(dict(naics3=n, n_TB=len(a), n_AC=len(b),
                     start_TB=round(a.start.mean(), 4), start_AC=round(b.start.mean(), 4),
                     d_start=round(d_start, 4), t_start=round(d_start / se_start, 2),
                     above_TB=round(a.gap_above.mean(), 4),
                     above_AC=round(b.gap_above.mean(), 4),
                     d_above=round(d_above, 4), t_above=round(d_above / se_above, 2)))
W = pd.DataFrame(rows)
print(W.to_string(index=False))

# pooled within-cell difference, cells weighted by TB count
w = W.n_TB / W.n_TB.sum()
pooled_start = float((W.d_start * w).sum())
pooled_above = float((W.d_above * w).sum())
print(f"\n pooled WITHIN-industry gap (TB-weighted over the {len(both_cells)} cells):")
print(f"   start     {pooled_start:+.4f}   (raw, ignoring industry: "
      f"{A.start.mean() - B.start.mean():+.4f})")
print(f"   gap_above {pooled_above:+.4f}   (raw: {A.gap_above.mean() - B.gap_above.mean():+.4f})")
share = 1 - abs(pooled_start) / abs(A.start.mean() - B.start.mean())
print(f"   => holding industry fixed removes {share:.0%} of the start-point gap")


# =============================================================================
# 3 · reframe: WITHIN threshold-bound, how much of the start position is industry?
# =============================================================================
# The TB-vs-AC comparison cannot separate industry from economics. The question can
# still be asked inside the threshold-bound sample, which is the one the design is
# about: of the variation in where a facility sits relative to the line, how much is
# between industries and how much is within?
g = A.groupby("naics3").start
nj, mj = g.size(), g.mean()
grand = A.start.mean()
ss_between = float((nj * (mj - grand) ** 2).sum())
ss_total = float(((A.start - grand) ** 2).sum())
r2_naics3 = ss_between / ss_total

A6 = A.dropna(subset=["naics"]).copy()
A6["naics6"] = pd.to_numeric(A6.naics, errors="coerce")
A6 = A6.dropna(subset=["naics6"])
g6 = A6.groupby("naics6").start
n6, m6 = g6.size(), g6.mean()
gr6 = A6.start.mean()
r2_naics6 = float((n6 * (m6 - gr6) ** 2).sum()) / float(((A6.start - gr6) ** 2).sum())

print("\n=== 3 · within threshold-bound: how much of the START POSITION is industry? ===")
print(f"   share of the variance in `start` between NAICS3 : {r2_naics3:.1%}  "
      f"({A.naics3.nunique()} industries, {len(A):,} obs)")
print(f"   share between NAICS6                            : {r2_naics6:.1%}  "
      f"({A6.naics6.nunique()} industries, {len(A6):,} obs)")
print("   the remainder is WITHIN industry -- the part firm revenue / employment")
print("   would have to explain.")

D = pd.DataFrame([
    dict(quantity="raw start-point gap, TB - AC", value=round(A.start.mean() - B.start.mean(), 4)),
    dict(quantity="pooled within-industry gap (4 overlapping cells)", value=round(pooled_start, 4)),
    dict(quantity="share of the gap removed by holding industry fixed", value=round(share, 3)),
    dict(quantity="raw gap_above difference", value=round(A.gap_above.mean() - B.gap_above.mean(), 4)),
    dict(quantity="pooled within-industry gap_above", value=round(pooled_above, 4)),
    dict(quantity="variance of start between NAICS3, within TB", value=round(r2_naics3, 4)),
    dict(quantity="variance of start between NAICS6, within TB", value=round(r2_naics6, 4)),
])
D.to_csv(OUT / "t43_start_gap_decomp.csv", index=False)
W.to_csv(OUT / "t44_start_gap_by_naics.csv", index=False)
print("\nwrote t42, t43, t44")


# =============================================================================
# 4 · figure
# =============================================================================
NAME = {486: "pipelines", 211: "oil & gas extr.", 311: "food mfg", 325: "chemicals",
        221: "utilities", 331: "primary metals", 327: "nonmetallic mineral",
        322: "paper", 336: "transport equip", 611: "educational svcs",
        212: "mining (ex. oil/gas)", 324: "petroleum & coal", 562: "waste mgmt"}

fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.7))

# (a) common support
sh = ct.sort_values("TB", ascending=False).head(12)
y = np.arange(len(sh))[::-1]
h = .38
ax[0].barh(y + h / 2, sh.tb_share * 100, h, color="#C44E52", alpha=.85,
           label="threshold-bound")
ax[0].barh(y - h / 2, sh.ac_share * 100, h, color="#4C72B0", alpha=.85,
           label="always-covered")
ax[0].set_yticks(y)
ax[0].set_yticklabels([f"{int(i)} {NAME.get(int(i), '')}" for i in sh.index], fontsize=8)
ax[0].set_xlabel("% of the group's observations in the window")
ax[0].legend(frameon=False, fontsize=8.5)
ax[0].set_title("(a) the two groups barely share an industry\n"
                "only 4 of 43 cells have both; they hold 9.7% of TB, 97.7% of AC",
                fontsize=9.5)

# (b) within-cell gaps
yy = np.arange(len(W))[::-1]
se = W.d_start / W.t_start.replace(0, np.nan)
ax[1].errorbar(W.d_start, yy, xerr=1.96 * se.abs(), fmt="o", ms=7, capsize=3,
               color="#C44E52")
ax[1].axvline(0, lw=.8, color="k")
ax[1].axvline(A.start.mean() - B.start.mean(), ls=":", lw=1.4, color="#888888")
ax[1].text(A.start.mean() - B.start.mean() + .002, yy.min() - .35,
           "raw gap,\nindustry ignored", fontsize=8, color="#666666", va="bottom")
ax[1].set_yticks(yy)
ax[1].set_yticklabels([f"{int(n)} {NAME.get(int(n), '')}\n(TB {a:,} / AC {b:,})"
                       for n, a, b in zip(W.naics3, W.n_TB, W.n_AC)], fontsize=8)
ax[1].set_xlabel("start-point gap, TB minus AC, within industry")
ax[1].set_title("(b) holding industry fixed removes only 23% of it", fontsize=9.5)

# (c) variance decomposition inside threshold-bound
lv = ["NAICS3\n(41 industries)", "NAICS6\n(236 industries)"]
btw = [r2_naics3 * 100, r2_naics6 * 100]
ax[2].bar(lv, btw, .5, color="#4C72B0", alpha=.85, label="between industries")
ax[2].bar(lv, [100 - v for v in btw], .5, bottom=btw, color="#DDDDDD",
          label="within industry")
for i, v in enumerate(btw):
    if v >= 6:
        ax[2].text(i, v / 2, f"{v:.1f}%", ha="center", va="center", fontsize=10,
                   color="white", fontweight="bold")
    else:
        ax[2].text(i, v + 1.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=10,
                   color="#4C72B0", fontweight="bold")
    ax[2].text(i, v + (100 - v) / 2, f"{100-v:.1f}%\nwhat revenue /\nemployment\nwould have to\nexplain",
               ha="center", va="center", fontsize=8.5, color="#444444")
ax[2].set_ylabel("% of the variance in `start`, threshold-bound only")
ax[2].set_ylim(0, 100)
ax[2].legend(frameon=False, fontsize=8.5, loc="upper right")
ax[2].set_title("(c) industry explains 2-8% of where a facility\nsits relative to the line",
                fontsize=9.5)

for a_ in ax:
    a_.spines[["top", "right"]].set_visible(False)
fig.suptitle("f35 - John's item 3: the start-point gap is not industry", fontsize=12, y=1.03)
fig.tight_layout()
fig.savefig(OUT / "f35_start_gap_industry.png", dpi=150, bbox_inches="tight")
print("wrote f35_start_gap_industry.png")
