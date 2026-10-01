"""
36 · QCEW: state x NAICS3 industry aggregates                           (2026-09-28)
================================================================================
QCEW was demoted on 2026-09-13. John's third point turned out to be answerable with the
NAICS codes already in the panel (working note 09): industry explains only 2.0% (NAICS3) to
8.2% (NAICS6) of the variance in where facilities start, so an INDUSTRY-LEVEL aggregate
cannot explain variation that lives WITHIN industries. QCEW is therefore not the answer
to John ③. It is still worth having for two narrower jobs:

  1. controls and descriptive context -- industry size, pay, establishment scale, by
     state and year, to sit alongside the facility panel;
  2. a DENOMINATOR. QCEW counts every establishment covered by unemployment insurance;
     GHGRP counts only those above 25,000 tCO2e. The ratio is the share of an industry
     that is visible to the disclosure regime at all, which is the "selective
     invisibility" framing in a number rather than a phrase.

WHAT IS EXTRACTED
  own_code = 5 (private) and agglvl_code = 55, which is state x 3-digit NAICS. Verified
  against the data rather than assumed: at agglvl 55 every industry_code is three
  characters and area_fips ends in 000.
  County level (agglvl 75) is deliberately NOT used: the GHGRP panel carries county
  NAMES, not FIPS, and matching names to FIPS needs a gazetteer this project does not
  have. State is enough for a control.

  ⚠️ `disclosure_code` = "N" marks a cell suppressed for confidentiality. Those rows
  carry zeros, not missing values, so they are set to NaN here rather than read as zero.

outputs  output/qcew_state_naics3.csv   output/t62_ghgrp_vs_qcew.csv
         output/f43_qcew_coverage.png
run      python3 36_qcew_industry.py            (from analysis/)
"""
import subprocess, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from ghgrp_load import load_cached

OUT = Path("output")
QDIR = Path("data/qcew")
CACHE = OUT / "_qcew_cache"
CACHE.mkdir(parents=True, exist_ok=True)
YEARS = range(2010, 2024)

# FIPS <-> USPS. Fixed codes; the GHGRP panel carries the two-letter abbreviation and
# QCEW carries the numeric FIPS, so one of the two has to be spelled out somewhere.
FIPS = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT",
    "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL",
    "18": "IN", "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD",
    "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT", "31": "NE",
    "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA", "54": "WV",
    "55": "WI", "56": "WY", "72": "PR", "78": "VI",
}

AWK = (r'''NR>1{o=$2;al=$4;gsub(/"/,"",o);gsub(/"/,"",al);'''
       r'''if(o=="5"&&al=="55"){a=$1;ic=$3;y=$6;dc=$8;'''
       r'''gsub(/"/,"",a);gsub(/"/,"",ic);gsub(/"/,"",y);gsub(/"/,"",dc);'''
       r'''print a","ic","y","dc","$9","$10","$11","$14","$15}}''')

for y in YEARS:
    dst = CACHE / f"qcew_{y}.csv"
    # the cache is valid only if the header actually made it in: the first version of
    # this script wrote the header through Python's buffer and let awk append on the
    # same handle, so awk's output landed first and the header was lost.
    if dst.exists() and dst.open().readline().startswith("area_fips"):
        continue
    src = QDIR / f"{y}.annual.singlefile.csv"
    with open(dst, "w") as fh:
        fh.write("area_fips,naics3,year,disclosure_code,estabs,emplvl,"
                 "total_wages,avg_wkly_wage,avg_annual_pay\n")
        fh.flush()                      # before handing the handle to awk
        subprocess.run(["awk", "-F,", AWK, str(src)], stdout=fh, check=True)
    print(f"  {y} extracted", flush=True)

Q = pd.concat([pd.read_csv(CACHE / f"qcew_{y}.csv", dtype={"area_fips": str,
                                                           "naics3": str})
               for y in YEARS], ignore_index=True)
Q["state_fips"] = Q.area_fips.str[:2]
Q["state"] = Q.state_fips.map(FIPS)
Q = Q[Q.state.notna()]
# suppressed cells carry zeros, not blanks -- do not read them as real zeros
sup = Q.disclosure_code.astype(str).str.upper().eq("N")
for c in ("estabs", "emplvl", "total_wages", "avg_wkly_wage", "avg_annual_pay"):
    Q[c] = pd.to_numeric(Q[c], errors="coerce")
    Q.loc[sup, c] = np.nan
Q["suppressed"] = sup
Q["avg_estab_size"] = Q.emplvl / Q.estabs.replace(0, np.nan)
Q = Q[["state", "state_fips", "naics3", "year", "estabs", "emplvl", "total_wages",
       "avg_annual_pay", "avg_estab_size", "suppressed"]]
Q.to_csv(OUT / "qcew_state_naics3.csv", index=False)

print(f"\nqcew_state_naics3.csv: {len(Q):,} rows, {Q.state.nunique()} states, "
      f"{Q.naics3.nunique()} NAICS3, {Q.year.min()}-{Q.year.max()}")
print(f"  suppressed cells: {Q.suppressed.mean()*100:.1f}%")


# ---------------------------------------------------------------- 2 · controls
de = load_cached()
P = de[["FacilityId", "year", "state", "naics3", "emissions", "threshold_bound",
        "always_covered"]].copy()
P["naics3"] = P.naics3.astype(str).str.zfill(3)
Q["naics3"] = Q.naics3.astype(str).str.zfill(3)

LK = P.merge(Q.drop(columns=["state_fips", "suppressed"]),
             on=["state", "naics3", "year"], how="left")
LK = LK.rename(columns={"estabs": "qcew_estabs", "emplvl": "qcew_emplvl",
                        "total_wages": "qcew_total_wages",
                        "avg_annual_pay": "qcew_avg_annual_pay",
                        "avg_estab_size": "qcew_avg_estab_size"})
LK.to_csv(OUT / "link_qcew.csv", index=False)
cov_n = LK.qcew_emplvl.notna().mean()
cov_e = LK.loc[LK.qcew_emplvl.notna(), "emissions"].sum() / max(LK.emissions.sum(), 1)
print("\n" + "=" * 74)
print("2 . QCEW as a control: coverage of the facility panel")
print("=" * 74)
print(f"  facility-years matched to a state x NAICS3 x year cell: "
      f"{cov_n*100:.1f}% by count, {cov_e*100:.1f}% by emissions")
miss = LK[LK.qcew_emplvl.isna()].naics3.value_counts().head(6)
print("\n  NAICS3 with no private-sector QCEW cell (top 6):")
print(miss.to_string())
print("  ⚠️ QCEW own_code=5 is PRIVATE only. Federal and municipal facilities -- and")
print("     the GHGRP has plenty (US GOVERNMENT is a top-10 parent) -- have no cell.")


# ---------------------------------------------------------------- 3 · denominator
print("\n" + "=" * 74)
print("3 . the denominator: what share of an industry is visible to the GHGRP?")
print("=" * 74)
print("  ⚠️ This is an INDICATOR, not an exact rate. A GHGRP facility is not a QCEW")
print("     establishment: one facility can span several establishments, and QCEW")
print("     counts every UI-covered establishment including very small ones. Read the")
print("     ordering across industries, not the level.")

g = (P[P.year == 2022].groupby("naics3")
     .agg(ghgrp_facilities=("FacilityId", "nunique"),
          ghgrp_emissions=("emissions", "sum")).reset_index())
q = (Q[(Q.year == 2022) & ~Q.suppressed].groupby("naics3")
     .agg(qcew_estabs=("estabs", "sum"), qcew_emplvl=("emplvl", "sum"),
          qcew_pay=("avg_annual_pay", "median")).reset_index())
D = g.merge(q, on="naics3", how="left")
D["visible_pct"] = (D.ghgrp_facilities / D.qcew_estabs * 100).round(3)
D["emissions_Mt"] = (D.ghgrp_emissions / 1e6).round(1)
D = D.sort_values("ghgrp_emissions", ascending=False)
D.to_csv(OUT / "t62_ghgrp_vs_qcew.csv", index=False)
show = D.head(14)[["naics3", "ghgrp_facilities", "qcew_estabs", "visible_pct",
                   "emissions_Mt", "qcew_pay"]]
print("\n  2022, the 14 NAICS3 with the most GHGRP emissions:")
print(show.to_string(index=False))

# ---------------------------------------------------------------- 4 · figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8))

d = D.dropna(subset=["qcew_estabs"]).head(14).iloc[::-1]
yy = np.arange(len(d))
ax[0].barh(yy, d.visible_pct, .62, color="#4C72B0", alpha=.85)
ax[0].set_yticks(yy); ax[0].set_yticklabels("NAICS " + d.naics3, fontsize=8)
ax[0].set_xscale("log")
ax[0].set_xlabel("GHGRP facilities as % of QCEW establishments (log)")
ax[0].set_title("(a) how much of each industry the 25,000 line\n"
                "makes visible -- three orders of magnitude apart", fontsize=10)

tb = LK[LK.threshold_bound & LK.qcew_avg_annual_pay.notna()]
ac = LK[LK.always_covered & LK.qcew_avg_annual_pay.notna()]
bins = np.linspace(3e4, 1.6e5, 40)
ax[1].hist(tb.qcew_avg_annual_pay, bins=bins, density=True, alpha=.6,
           color="#C44E52", label=f"threshold-bound (n={len(tb):,})")
ax[1].hist(ac.qcew_avg_annual_pay, bins=bins, density=True, alpha=.55,
           color="#4C72B0", label=f"always-covered (n={len(ac):,})")
ax[1].set_xlabel("QCEW average annual pay in the facility's state x NAICS3 ($)")
ax[1].set_ylabel("density")
ax[1].legend(frameon=False, fontsize=8)
ax[1].set_title("(b) a NEGATIVE result: the two groups' industry pay\n"
                "distributions overlap, so QCEW does not separate them", fontsize=10)

# (c) is the visible share just a story about establishment size? The guess was that
# industries whose typical establishment is large would have more plants over 25,000.
# It comes out BACKWARDS (slope about -0.5): pipelines have few employees per
# establishment and enormous emissions, schools have many employees and almost none.
# So QCEW EMPLOYMENT is not a usable proxy for facility scale in this setting -- what
# puts a plant over the line is emissions intensity, not headcount. Reported as the
# negative result it is.
E = (Q[(Q.year == 2022) & ~Q.suppressed].groupby("naics3")
     .apply(lambda d: d.emplvl.sum() / max(d.estabs.sum(), 1)).rename("estab_size")
     .reset_index())
C = D.merge(E, on="naics3", how="left").dropna(subset=["estab_size", "visible_pct"])
C = C[(C.visible_pct > 0) & (C.ghgrp_facilities >= 20)]
sz = 20 + 280 * (C.ghgrp_emissions / C.ghgrp_emissions.max()) ** .45
ax[2].scatter(C.estab_size, C.visible_pct, s=sz, alpha=.55, color="#C44E52",
              edgecolor="white", linewidth=.8)
for _, r in C.nlargest(7, "ghgrp_emissions").iterrows():
    ax[2].annotate(r.naics3, (r.estab_size, r.visible_pct), fontsize=8,
                   xytext=(4, 4), textcoords="offset points")
lx, ly = np.log(C.estab_size), np.log(C.visible_pct)
b1, b0 = np.polyfit(lx, ly, 1)
xs_ = np.linspace(lx.min(), lx.max(), 50)
ax[2].plot(np.exp(xs_), np.exp(b0 + b1 * xs_), "--", lw=1.2, color="#555")
rho = np.corrcoef(lx, ly)[0, 1]
ax[2].set_xscale("log"); ax[2].set_yscale("log")
ax[2].set_xlabel("QCEW average establishment size (employees, 2022)")
ax[2].set_ylabel("GHGRP facilities as % of establishments")
ax[2].set_title(f"(c) another NEGATIVE result: headcount does not predict\n"
                f"visibility (slope {b1:.2f}, r = {rho:.2f}). Emissions intensity does.",
                fontsize=10)

for a_ in ax:
    a_.spines[["top", "right"]].set_visible(False)
fig.suptitle("f43 - QCEW industry context for the GHGRP panel", fontsize=12, y=1.03)
fig.tight_layout()
fig.savefig(OUT / "f43_qcew_coverage.png", dpi=150, bbox_inches="tight")
print(f"\nwrote {OUT/'link_qcew.csv'}, {OUT/'t62_ghgrp_vs_qcew.csv'}, "
      f"{OUT/'f43_qcew_coverage.png'}")
