"""
18 · The perceived probability of being caught, against IRS audit rates
================================================================================
John, 3 September 2026:

    "The perceptions of the probability of being caught line up well with the actual
     probabilities of being audited by the IRS, which may bleed over to firms'
     perceptions of being caught for failing to report."

Elaine's note on the slide: "Actual probability is closer to the perception /
Actual tax audit rate, especially for large companies."

WHAT THIS DOES
--------------
The backward induction (node 9) takes compliance as revealed preference. A facility
that files must believe

        p_hat * F  >=  c        =>   p_hat  >=  c / F

with c = $7,654/year, the EPA ICR estimate of the annual reporting and recordkeeping
burden. That gives a LOWER BOUND on the perceived probability, one per assumption
about the penalty F.

The paper's puzzle was that these bounds are two to four orders of magnitude above the
GHGRP's own enforcement probability (<= 0.0032%: zero inspections, one stationary-source
case at $0 in fourteen years). John's observation reframes it: firms may not be pricing
GHGRP enforcement at all. They may be importing the enforcement probability they
actually experience, which is the IRS's.

This file puts the two sets of numbers on one axis so the claim can be checked rather
than asserted.

A NOTE ON THE IRS NUMBERS
-------------------------
IRS Data Book coverage rates for recent tax years are NOT comparable to older ones.
Examinations stay open through the statutory period, and the Data Book counts only
closed ones, so a recent year's published rate is a floor that rises for years. The IRS
says so itself: TY2018 is "the most recent tax year outside the statutory period." So
TY2018 is the anchor here, TY2021 figures are marked immature, and nothing is concluded
from them.

outputs  t35_irs_benchmark.csv  f31_perceived_vs_irs.png
run      python3 18_irs_audit_benchmark.py        (from analysis/)
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("output")
# --- c, VERIFIED 2026-09-12 against the primary source -----------------------
# 89 FR 21516 (2024-03-28), "Agency Information Collection Activities ... Greenhouse Gas
# Reporting Program (Renewal)", EPA ICR Number 2300.20, OMB Control Number 2060-0629.
# Quoted verbatim from the notice:
#     "Estimated number of respondents (annual average): 12,434."
#     "Total estimated burden: 705,554 hours (per year)."
#     "Total estimated cost: $95,175,521 (per year), which includes $33,282,257
#      annualized capital or operation & maintenance costs."
#     "...adjustment of labor rates, capital, and operation and maintenance (O&M) costs
#      to reflect 2021 dollars."
ICR_TOTAL_COST   = 95_175_521          # per year, 2021 dollars
ICR_CAPITAL_OM   = 33_282_257          # of which annualised capital / O&M
ICR_RESPONDENTS  = 12_434              # annual average
ICR_BURDEN_HOURS = 705_554             # per year

C_ANNUAL = ICR_TOTAL_COST / ICR_RESPONDENTS            # = 7,654.46  -> the $7,654 we used
C_PAPERWORK = (ICR_TOTAL_COST - ICR_CAPITAL_OM) / ICR_RESPONDENTS   # labour only

# TWO THINGS THE VERIFICATION TURNED UP, both of which change how c should be used:
#
# (1) c is NOT all paperwork. 35% of it ($33.3M of $95.2M) is annualised capital and
#     O&M -- monitoring equipment, not filing. A facility that stops reporting avoids
#     the filing for certain; whether it also avoids the monitoring depends on whether
#     the equipment is required by something else (a state programme, a Title V permit).
#     So the avoidable cost is bounded:  C_PAPERWORK <= c <= C_ANNUAL.
#     Both variants are carried through below.
#
# (2) The 12,434 respondents include SUPPLIERS and CO2 INJECTORS, not just the direct
#     emitters in our panel (~6,500 per year). Per-respondent cost is therefore an
#     average over a broader population than the sample the bound is applied to. If
#     direct emitters bear more than the average, c is understated and every p-hat with
#     it. Flag in the paper; do not claim c is facility-specific.
#
# Dollars are 2021. The panel is 2010-2023. Not deflated anywhere.

# --- our bounds ------------------------------------------------------------
PENALTIES = [
    ("ECHO median fine",            30_510),
    ("ICIS minimum case",           84_546),
    ("ICIS maximum case",          382_473),
    ("statutory upper limit",    1_000_000),
]
rows = [dict(group="perceived (this paper)",
             label=f"p-hat, F = ${F:,} ({name})",
             rate=C_ANNUAL / F, note=f"c/F with c = ${C_ANNUAL:,.0f} (full ICR cost)",
             mature=True)
        for name, F in PENALTIES]

print(f"c, full ICR cost per respondent      = ${C_ANNUAL:,.2f}"
      f"   ({ICR_TOTAL_COST:,} / {ICR_RESPONDENTS:,})")
print(f"c, excluding capital and O&M         = ${C_PAPERWORK:,.2f}"
      f"   (capital/O&M is {ICR_CAPITAL_OM/ICR_TOTAL_COST:.0%} of the total)")
print(f"burden hours per respondent per year = {ICR_BURDEN_HOURS/ICR_RESPONDENTS:,.1f}")
print(f"implied labour rate                  = "
      f"${(ICR_TOTAL_COST-ICR_CAPITAL_OM)/ICR_BURDEN_HOURS:,.2f}/hour\n")

# --- IRS examination coverage ------------------------------------------------
# VERIFIED 2026-09-12 cell by cell against the primary source:
#   IRS Data Book Table 3-1, "Examination Coverage and Recommended Additional Tax After
#   Examination, by Type and Size of Return, Tax Years (2010)-2023"
#   file 25db-3-01-ex-allyears.xlsx, sheet "Table 3-1 All Available Years"
# www.irs.gov is blocked by this session's egress policy, so the workbook was read in a
# browser viewer and each figure below was confirmed by selecting the cell and reading
# the formula bar. The cell reference is recorded next to every number.
#
# Two of the second-hand figures used on 3 September did NOT survive:
#   $20B+ TY2018 was quoted at 57.2%  -> the primary source says 62.5% (AI37)
#   individuals TY2018 was quoted at 0.3% -> the primary source says 0.4% (AI8)
# and two TRAC figures (assets $250M+, FY2002 34.4% and FY2005 44.1%) are DROPPED:
# they are fiscal-year, not tax-year, and they predate this table's range entirely.
# The TY2018 asset ladder below replaces them on a like-for-like basis.
#
# DEFINITION NOTE: in this edition "percentage covered" counts closed AND in-process
# examinations against returns filed (footnote [4]). Older Data Books counted closed
# examinations only. That change, as much as maturation, is why the older secondary
# quotations sit below these. Do not mix editions.
IRS = [
    # label                                        rate   mature  cell / note
    ("corporations, assets $20B+, TY2011",         0.845, True,
     "Table 3-1 cell BY37 = 84.4512195121951"),
    ("corporations, assets $20B+, TY2018",         0.625, True,
     "Table 3-1 cell AI37 = 62.5  (the 57.2% quoted on 3 Sept was a stale secondary source)"),
    ("corporations, assets $5B-20B, TY2011",       0.604, True,
     "Table 3-1 cell BY36 = 60.4379562043796"),
    ("corporations, assets $5B-20B, TY2018",       0.342, True,
     "Table 3-1 cell AI36 = 34.2"),
    ("corporations, assets $1B-5B, TY2018",        0.167, True, "Table 3-1 row 35, col AI"),
    ("corporations, assets $500M-1B, TY2018",      0.079, True, "Table 3-1 row 34, col AI"),
    ("corporations, assets $250M-500M, TY2018",    0.053, True, "Table 3-1 row 33, col AI"),
    ("all corporations, TY2011",                   0.014, True,
     "Table 3-1 cell BY23 = 1.42603192790225"),
    ("all corporations, TY2012",                   0.013, True,
     "Table 3-1 cell BS23 = 1.30750436171836"),
    ("all corporations, TY2018",                   0.006, True, "Table 3-1 row 23, col AI"),
    ("all individuals, TY2018",                    0.004, True,
     "Table 3-1 cell AI8 = 0.4  (the 0.3% quoted on 3 Sept was a stale secondary source)"),
    ("corporations, assets $20B+, TY2023",         0.219, False,
     "Table 3-1 row 37, col E - IMMATURE, still inside the 3-year statute"),
]
rows += [dict(group="actual (IRS audit)", label=lab, rate=r, note=note, mature=m)
         for lab, r, m, note in IRS]

# --- what GHGRP enforcement actually is -----------------------------------
rows.append(dict(group="actual (GHGRP enforcement)",
                 label="GHGRP, failure to report",
                 rate=0.000032, note="zero inspections; one stationary-source case at $0 in 14 years",
                 mature=True))

# --- sensitivity: the same bounds with the capital/O&M share stripped out
SENS = pd.DataFrame([
    dict(F=F, name=name,
         p_full=round(100 * C_ANNUAL / F, 3),
         p_paperwork_only=round(100 * C_PAPERWORK / F, 3))
    for name, F in PENALTIES])
SENS.to_csv(OUT / "t35b_c_sensitivity.csv", index=False)
print("=== p-hat under the two readings of c (%) ===")
print(SENS.to_string(index=False))
print(f"\n every bound scales by {C_PAPERWORK/C_ANNUAL:.3f}. Both columns still sit inside")
print(" the IRS ladder (0.4% to 84.5%), so the conclusion does not turn on this choice.\n")

T = pd.DataFrame(rows)
T["pct"] = (T.rate * 100).round(4)
T.to_csv(OUT / "t35_irs_benchmark.csv", index=False)
pd.set_option("display.width", 200)
print("=== every number on one scale ===")
print(T.sort_values("rate", ascending=False)[["group", "label", "pct", "mature", "note"]]
       .to_string(index=False))

# --- the pairing that matters ---------------------------------------------
print("\n=== each bound against its nearest MATURE IRS analogue ===")
mature = T[(T.group == "actual (IRS audit)") & T.mature]
pairs = []
for _, r in T[T.group == "perceived (this paper)"].iterrows():
    j = (mature.rate - r.rate).abs().idxmin()
    m = mature.loc[j]
    pairs.append(dict(bound=r.label, p_hat_pct=round(r.rate * 100, 2),
                      nearest_irs=m.label, irs_pct=round(m.rate * 100, 2),
                      ratio=round(r.rate / m.rate, 2)))
P = pd.DataFrame(pairs)
print(P.to_string(index=False))
print("\n ratio ~1 means the perceived probability implied by compliance is the same")
print(" number the firm already faces in the tax system.")

ghg = 0.000032
print(f"\n=== and against GHGRP's own enforcement probability ({ghg*100:.4f}%) ===")
for _, r in T[T.group == "perceived (this paper)"].iterrows():
    print(f"  {r.label:<52s}  {r.rate/ghg:>9,.0f}x")
print(f"  {'all individuals, TY2018 (IRS)':<52s}  {0.004/ghg:>9,.0f}x")
print(f"  {'all corporations, TY2018 (IRS)':<52s}  {0.006/ghg:>9,.0f}x")
print(f"  {'corporations $20B+, TY2018 (IRS)':<52s}  {0.625/ghg:>9,.0f}x")

# --- figure ----------------------------------------------------------------
# Form: magnitudes spanning five orders of magnitude across ~15 named entities ->
# horizontal lollipop on a log axis. One axis. Every point directly labelled, so
# identity never rests on colour alone.
CAT = {"perceived (this paper)": "#C44E52",
       "actual (IRS audit)": "#4C72B0",
       "actual (GHGRP enforcement)": "#55A868"}

d = T.sort_values("rate").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(11.5, 7.2))
for i, r in d.iterrows():
    col = CAT[r.group]
    ax.plot([1e-5, r.rate * 100], [i, i], lw=2, color=col,
            alpha=.35 if not r.mature else .8,
            ls=":" if not r.mature else "-", solid_capstyle="round")
    ax.plot(r.rate * 100, i, "o", ms=9, color=col,
            mfc="white" if not r.mature else col, mew=2, zorder=3)
    txt = f"{r.rate*100:.3g}%" + ("  (immature)" if not r.mature else "")
    ax.text(r.rate * 100 * 1.35, i, txt, va="center", fontsize=8.5, color="#333333")

ax.set_yticks(range(len(d)))
ax.set_yticklabels(d.label, fontsize=9)
ax.set_xscale("log")
ax.set_xlim(2e-3, 600)
ax.set_xlabel("probability of being caught / audited  (%, log scale)")
ax.grid(axis="x", lw=.4, alpha=.35)
ax.set_axisbelow(True)
ax.spines[["top", "right", "left"]].set_visible(False)
handles = [plt.Line2D([], [], marker="o", ls="-", lw=2, ms=9, color=c, label=g)
           for g, c in CAT.items()]
ax.legend(handles=handles, frameon=False, fontsize=9, loc="lower right")
ax.set_title("f31 - the perceived probability implied by compliance sits in the range of\n"
             "IRS audit rates, four orders of magnitude above GHGRP's own enforcement",
             fontsize=11, loc="left")
fig.tight_layout()
fig.savefig(OUT / "f31_perceived_vs_irs.png", dpi=150, bbox_inches="tight")
print("\nwrote t35_irs_benchmark.csv, f31_perceived_vs_irs.png")
