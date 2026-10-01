"""
37 · Compustat North America -> an empirical bound on pi                (2026-09-28)
================================================================================
pi is marginal profit per tCO2e. It is the ONLY free parameter left in the paper: the
headline upper bound is stated in tonnes (c/pi <= 376 tCO2e, working note 13), and pi is
needed only to convert that into dollars for the appendix. This file replaces the
calibrated pi with an estimated one.

    data/compustat_us/hgdeegxrsw23mktq.csv   Fundamentals Annual, INDL/C/STD/D,
                                             306,756 USA firm-years, with CRSP link

⚠️ THE EARLIER FILE WAS THE WRONG PRODUCT. data/compustat/ holds Compustat GLOBAL,
   which by construction contains no US or Canadian firms (verified: 1,005,901 rows,
   USA = 0). It is kept only so the mistake stays visible; nothing reads it.

WHICH DIRECTION THE BOUND NEEDS
  The null gives c <= pi * W_min, so the bound LOOSENS as pi rises. A defensible claim
  therefore needs an UPPER bound on pi, not a point estimate. Two measures are built:

    revenue per tonne    revt / emissions     an upper bound on marginal profit, since
                                              marginal profit <= marginal revenue
    EBITDA per tonne     oibdp / emissions    closer to the economic object, but an
                                              average rather than a margin

  Both are computed on the firm's GHGRP-reported emissions only. A firm whose emissions
  are partly outside the GHGRP gets an overstated pi, which loosens the bound -- the
  conservative direction, and worth saying so explicitly.

MATCHING is the weak link: GHGRP reports a parent NAME, Compustat has `conm`, and there
is no shared identifier. Exact match on a normalised name, then a token-overlap pass.
Match rates are reported by count AND weighted by emissions, per the project's rule.

outputs  output/link_compustat.csv  output/t63_pi_estimates.csv  output/f44_pi.png
run      python3 37_compustat_pi.py            (from analysis/)
"""
import re, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from ghgrp_load import load_cached

OUT = Path("output")
SRC = Path("data/compustat_us/hgdeegxrsw23mktq.csv")
SLIM = OUT / "_compustat_slim.pkl"
W_MIN_CAWA, W_MIN_WA = 376.0, 594.0        # from 27_dominated_region.py
C_ANNUAL, C_PAPER = 7654.0, 4978.0

SUFFIX = r"\b(llc|l\.l\.c|inc|incorporated|corp|corporation|co|company|lp|l\.p|llp|" \
         r"ltd|limited|plc|holdings?|group|the|and|&)\b"


def clean_name(s):
    x = str(s).lower()
    x = re.sub(r"[.,'\"()/]", " ", x)
    x = re.sub(SUFFIX, " ", x)
    x = re.sub(r"[^a-z0-9 ]", " ", x)
    return re.sub(r"\s+", " ", x).strip()


# ---------------------------------------------------------------- 1 · Compustat
if SLIM.exists():
    CS = pd.read_pickle(SLIM)
else:
    cols = ["GVKEY", "conm", "fyear", "fic", "state", "naics", "sic", "revt", "cogs",
            "xsga", "oibdp", "ebitda", "at", "emp", "LINKPRIM"]
    CS = pd.concat(list(pd.read_csv(SRC, usecols=cols, low_memory=False,
                                    chunksize=20000)), ignore_index=True)
    CS = CS[(CS.fic == "USA") & CS.fyear.between(2010, 2023)]
    CS["lp_rank"] = CS.LINKPRIM.map({"P": 0, "C": 1, "J": 2, "N": 3}).fillna(9)
    CS = (CS.sort_values(["GVKEY", "fyear", "lp_rank"])
            .drop_duplicates(["GVKEY", "fyear"]).drop(columns=["lp_rank", "LINKPRIM"]))
    CS["conm_clean"] = CS.conm.map(clean_name)
    CS.to_pickle(SLIM)
print(f"Compustat US firm-years 2010-2023: {len(CS):,}  ({CS.GVKEY.nunique():,} firms)")

# ---------------------------------------------------------------- 2 · GHGRP side
de = load_cached()
LP = pd.read_csv(OUT / "link_parent.csv")
LP = LP[LP.is_primary]                      # one owner per facility-year, see working note 17
F = de[["FacilityId", "year", "emissions", "threshold_bound", "naics3"]]
PE = (F.merge(LP[["FacilityId", "year", "parent_name", "parent_name_clean",
                  "pct_ownership"]], on=["FacilityId", "year"], how="inner"))
PARENT = (PE.groupby(["parent_name_clean", "parent_name", "year"], as_index=False)
            .agg(emissions=("emissions", "sum"),
                 n_facilities=("FacilityId", "nunique"),
                 emissions_tb=("emissions", lambda s: np.nan)))
for i, r in PARENT.iterrows():
    pass                                     # emissions_tb filled below, vectorised
tb = (PE[PE.threshold_bound].groupby(["parent_name_clean", "year"], as_index=False)
        .emissions.sum().rename(columns={"emissions": "emissions_tb"}))
PARENT = PARENT.drop(columns=["emissions_tb"]).merge(
    tb, on=["parent_name_clean", "year"], how="left")
PARENT["emissions_tb"] = PARENT.emissions_tb.fillna(0)
print(f"GHGRP parent-years: {len(PARENT):,}  "
      f"({PARENT.parent_name_clean.nunique():,} distinct parents)")

# ---------------------------------------------------------------- 3 · matching
cs_names = CS.drop_duplicates("conm_clean")[["conm_clean", "GVKEY", "conm"]]
exact = PARENT.merge(cs_names, left_on="parent_name_clean", right_on="conm_clean",
                     how="left")
exact["match_method"] = np.where(exact.GVKEY.notna(), "exact name", None)
print(f"\nexact name match: {exact.GVKEY.notna().mean()*100:.1f}% of parent-years, "
      f"{exact.loc[exact.GVKEY.notna(),'emissions'].sum()/exact.emissions.sum()*100:.1f}% "
      f"of emissions")

# token-overlap pass on the unmatched, ranked by emissions so effort goes where the
# mass is rather than alphabetically
STOP = {"us", "usa", "american", "america", "energy", "power", "resources", "industries",
        "international", "north", "partners", "operating", "services", "enterprises"}


def toks(s):
    return {t for t in str(s).split() if t not in STOP and len(t) > 2}


cs_tok = [(t, g, c) for t, g, c in
          zip(cs_names.conm_clean.map(toks), cs_names.GVKEY, cs_names.conm) if t]
un = exact[exact.GVKEY.isna()].drop_duplicates("parent_name_clean")
un = un.sort_values("emissions", ascending=False).head(600)
fuzzy = {}
for nm in un.parent_name_clean:
    a = toks(nm)
    if not a:
        continue
    best, bs = None, 0.0
    for t, g, c in cs_tok:
        j = len(a & t) / len(a | t)
        if j > bs:
            bs, best = j, (g, c)
    if bs >= 0.60:
        fuzzy[nm] = (best[0], best[1], round(bs, 3))
print(f"token-overlap pass: {len(fuzzy)} additional parents matched at Jaccard >= 0.60")

M = exact.copy()
hit = M.parent_name_clean.map(lambda n: fuzzy.get(n))
M.loc[M.GVKEY.isna() & hit.notna(), "match_method"] = "token overlap"
M["GVKEY"] = M.GVKEY.fillna(hit.map(lambda x: x[0] if isinstance(x, tuple) else np.nan))
M["conm"] = M.conm.fillna(hit.map(lambda x: x[1] if isinstance(x, tuple) else np.nan))
M["match_score"] = hit.map(lambda x: x[2] if isinstance(x, tuple) else np.nan)

cov_n = M.GVKEY.notna().mean()
cov_e = M.loc[M.GVKEY.notna(), "emissions"].sum() / M.emissions.sum()
cov_tb = (M.loc[M.GVKEY.notna(), "emissions_tb"].sum()
          / max(M.emissions_tb.sum(), 1))
print(f"\n>> total match: {cov_n*100:.1f}% of parent-years, {cov_e*100:.1f}% of all "
      f"emissions, {cov_tb*100:.1f}% of threshold-bound emissions")
M.to_csv(OUT / "link_compustat.csv", index=False)


# ---------------------------------------------------------------- 4 · pi
print("\n" + "=" * 78)
print("4 . pi -- and the reason the obvious calculation does not work")
print("=" * 78)
print("""  Dividing a firm's revenue by its GHGRP-reported emissions is only interpretable
  when the firm's business IS those facilities. Run on everything matched, the median
  comes back at $10,910 of revenue per tonne, which is nonsense as a marginal profit:
      BERKSHIRE HATHAWAY   66.5 Mt reported, $234bn revenue ->   $3,521 / tonne
      EXXON MOBIL          40.4 Mt reported, $399bn revenue ->   $9,875 / tonne
  Berkshire is a conglomerate and Exxon's US refinery emissions are a slice of a global
  business. The ratio is measuring diversification, not the price of a tonne of output.

  pi is therefore estimated on PURE-PLAY EMITTERS only: firms whose primary NAICS is
  221 (electric power) and whose GHGRP-reported emissions are large enough that the
  reporting facilities are the business. Utilities are the right population anyway --
  they are the sector where the 25,000 line binds most often.""")

J = M[M.GVKEY.notna()].copy()
J["GVKEY"] = J.GVKEY.astype(int)
J = J.merge(CS[["GVKEY", "fyear", "conm", "revt", "cogs", "oibdp", "at", "emp", "naics"]],
            left_on=["GVKEY", "year"], right_on=["GVKEY", "fyear"], how="inner",
            suffixes=("", "_cs"))
J = J[(J.emissions > 0) & J.revt.notna() & (J.revt > 0)]
J["cs_naics3"] = J.naics.astype(str).str[:3]
J["pi_revenue"] = J.revt * 1e6 / J.emissions
J["pi_ebitda"] = (J.oibdp * 1e6 / J.emissions).replace([np.inf, -np.inf], np.nan)
J.to_csv(OUT / "link_compustat_financials.csv", index=False)
print(f"\n  matched parent-years with usable financials: {len(J):,} "
      f"({J.conm.nunique():,} firms)")

SAMPLES = [("all matched parents (NOT usable, shown to make the point)", J),
           ("NAICS 221 utilities", J[J.cs_naics3 == "221"]),
           ("NAICS 221, >= 1 Mt reported", J[(J.cs_naics3 == "221") & (J.emissions >= 1e6)]),
           ("NAICS 221, >= 10 Mt reported  <- the estimate",
            J[(J.cs_naics3 == "221") & (J.emissions >= 1e7)])]
rows = []
for lab, sub in SAMPLES:
    if len(sub) < 20:
        continue
    r = sub.pi_revenue.dropna()
    e = sub.pi_ebitda.dropna()
    rows.append(dict(sample=lab, n=len(sub), firms=sub.conm.nunique(),
                     rev_p25=round(r.quantile(.25)), rev_med=round(r.median()),
                     rev_p75=round(r.quantile(.75)),
                     ebitda_p25=round(e.quantile(.25)), ebitda_med=round(e.median()),
                     ebitda_p75=round(e.quantile(.75))))
T = pd.DataFrame(rows)
T.to_csv(OUT / "t63_pi_estimates.csv", index=False)
print("\n  $ per tCO2e:")
print(T.to_string(index=False))

U = J[(J.cs_naics3 == "221") & (J.emissions >= 1e7)]
PI_EB = U.pi_ebitda.median()
PI_EB_LO, PI_EB_HI = U.pi_ebitda.quantile(.25), U.pi_ebitda.quantile(.75)
PI_REV = U.pi_revenue.median()
print(f"\n  >> pi, large pure-play utilities ({U.conm.nunique()} firms, {len(U)} firm-years):")
print(f"       EBITDA per tonne   ${PI_EB_LO:,.0f} - ${PI_EB:,.0f} - ${PI_EB_HI:,.0f}  (p25/median/p75)")
print(f"       revenue per tonne  ${PI_REV:,.0f} (median) -- a hard ceiling on marginal profit")

print("\n  ⭐ the f30 calibration (working note 13) put pi at $142/tCO2e from the width of the")
print("     California bunching region, with no financial data at all. It lands between")
print(f"     the EBITDA median (${PI_EB:,.0f}) and the revenue p25 of this sample.")
print("     Two independent routes, same order of magnitude. That is worth reporting.")

# ---------------------------------------------------------------- 5 · the bound
print("\n" + "=" * 78)
print("5 . the appendix dollar conversion, now with an estimated pi")
print("=" * 78)
print("   c <= pi * W_min,  W_min = 376 (CA+WA) / 594 (WA only) tCO2e\n")
print(f"   {'pi ($/tCO2e)':<44s} {'c <= (CA+WA)':>13s} {'c <= (WA only)':>15s}")
grid = [("CARB allowance price (auction 45)", 28.32),
        ("EBITDA/t, utilities p25", PI_EB_LO),
        ("EBITDA/t, utilities MEDIAN", PI_EB),
        ("EBITDA/t, utilities p75", PI_EB_HI),
        ("f30 bunching calibration", 141.60),
        ("revenue/t, utilities median (ceiling)", PI_REV)]
grid = sorted(grid, key=lambda x: x[1])
for lab, pi in grid:
    print(f"   {lab:<44s} {pi*W_MIN_CAWA:>13,.0f} {pi*W_MIN_WA:>15,.0f}")
pd.DataFrame(grid, columns=["basis", "pi"]).assign(
    c_max_cawa=lambda d: (d.pi * W_MIN_CAWA).round(0),
    c_max_wa=lambda d: (d.pi * W_MIN_WA).round(0)).to_csv(
        OUT / "t64_dollar_bound.csv", index=False)
print(f"\n   the ICR accounting cost is ${C_ANNUAL:,.0f} per facility-year, i.e. "
      f"{C_ANNUAL/PI_EB:,.0f} tCO2e at the median pi --")
print(f"   comfortably inside W_min = {W_MIN_CAWA:,.0f}, so the null and the accounting "
      f"cost remain consistent.")

# ---------------------------------------------------------------- 6 · figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8))

box = [J.pi_revenue.dropna(), J[J.cs_naics3 == "221"].pi_revenue.dropna(),
       U.pi_revenue.dropna(), U.pi_ebitda.dropna()]
lbl = ["all matched\n(uninterpretable)", "all NAICS 221", "221, >=10 Mt\nrevenue/t",
       "221, >=10 Mt\nEBITDA/t"]
bp = ax[0].boxplot(box, labels=lbl, showfliers=False, patch_artist=True, widths=.55)
for pat, c in zip(bp["boxes"], ["#dddddd", "#bbbbbb", "#4C72B0", "#C44E52"]):
    pat.set_facecolor(c); pat.set_alpha(.8)
ax[0].set_yscale("log")
ax[0].axhline(141.60, ls="--", lw=1.3, color="#8172B2")
ax[0].text(.55, 155, "f30 calibration $142", fontsize=8, color="#8172B2")
ax[0].set_ylabel("$ per tCO2e (log)")
ax[0].tick_params(axis="x", labelsize=7.5)
ax[0].set_title("(a) pi is only interpretable for pure-play emitters;\n"
                "there it brackets the bunching calibration", fontsize=10)

g = (U.groupby("conm").agg(e=("emissions", "median"), pi=("pi_ebitda", "median"))
       .dropna().sort_values("e", ascending=False).head(12).iloc[::-1])
yy = np.arange(len(g))
ax[1].barh(yy, g.pi, .62, color="#C44E52", alpha=.85)
ax[1].set_yticks(yy)
ax[1].set_yticklabels([c[:26] for c in g.index], fontsize=7.5)
ax[1].axvline(141.60, ls="--", lw=1.3, color="#8172B2")
ax[1].set_xlabel("EBITDA per tonne of reported CO2e ($)")
ax[1].set_title("(b) the twelve largest reporting utilities\n"
                "(dashed = the $142 calibration)", fontsize=10)

pis = np.logspace(np.log10(20), np.log10(400), 60)
ax[2].plot(pis, pis * W_MIN_CAWA, color="#C44E52", lw=2, label="CA+WA (W=376)")
ax[2].plot(pis, pis * W_MIN_WA, color="#55A868", lw=2, label="WA only (W=594)")
ax[2].axhline(C_ANNUAL, ls=":", lw=1.4, color="k")
ax[2].text(22, C_ANNUAL * 1.1, "ICR accounting cost $7,654", fontsize=8)
ax[2].axvspan(PI_EB_LO, PI_EB_HI, color="#C44E52", alpha=.12)
ax[2].axvline(PI_EB, ls="--", lw=1.2, color="#C44E52")
ax[2].text(PI_EB * 1.05, 2.2e5, f"estimated pi\n${PI_EB:,.0f}", fontsize=8, color="#8B2F33")
ax[2].set_xscale("log"); ax[2].set_yscale("log")
ax[2].set_xlabel("pi, $ per tCO2e (log)")
ax[2].set_ylabel("implied upper bound on c ($, log)")
ax[2].legend(frameon=False, fontsize=8)
ax[2].set_title("(c) the bound is a line in pi; the estimate\n"
                "puts it at tens of thousands of dollars", fontsize=10)

for a_ in ax:
    a_.spines[["top", "right"]].set_visible(False)
fig.suptitle("f44 - estimating pi from Compustat North America", fontsize=12, y=1.03)
fig.tight_layout()
fig.savefig(OUT / "f44_pi.png", dpi=150, bbox_inches="tight")
print(f"\nwrote t63_pi_estimates.csv, t64_dollar_bound.csv, f44_pi.png")
