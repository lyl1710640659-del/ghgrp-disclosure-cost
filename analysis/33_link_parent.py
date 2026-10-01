"""
33 · GHGRP Reported Parent Companies -> link_parent.csv                 (2026-09-28)
================================================================================
Source: EPA, "Reported Parent Companies", ghgp_data_parent_company.xlsb
        https://www.epa.gov/ghgreporting/data-sets  (October 2024 release)

ONE file, fourteen sheets, one per reporting year 2010-2023. Header on the first row.
The key is free: `GHGRP FACILITY ID` IS our FacilityId, so there is no matching problem
here -- unlike EIA (script 31) this link is exact.

⚠️ FORMAT. The file is .xlsb (Excel Binary), not .xlsx. Preview/Numbers/most online
   viewers render it blank, which is why it first looked like an empty download.
   Python reads it through `pyxlsb`.

MULTIPLE PARENTS. A joint venture reports several parents whose ownership shares sum to
100. The decision, recorded here rather than buried in a script: KEEP EVERY PARENT ROW,
and additionally flag the largest stake as `is_primary`. Analyses that need one owner per
facility filter on is_primary; analyses that want to apportion (emissions-weighted
ownership, say) use pct_ownership. Nothing is thrown away at this stage.

outputs  output/link_parent.csv          FacilityId x year x parent (long)
         output/t60_parent_coverage.csv  coverage against the emissions panel
run      python3 33_link_parent.py            (from analysis/)
"""
import re, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from ghgrp_load import load_cached

OUT = Path("output")
SRC = Path("data/ghgrp_parent/ghgp_data_parent_company.xlsb")

SUFFIX = r"\b(llc|l\.l\.c|inc|incorporated|corp|corporation|co|company|lp|l\.p|llp|" \
         r"ltd|limited|plc|holdings?|group|the|and|&)\b"


def clean_name(s):
    """Conservative normalisation for later matching to Compustat `conm`.
    Lowercase, strip punctuation and the usual corporate suffixes, squeeze spaces."""
    x = str(s).lower()
    x = re.sub(r"[.,'\"()/]", " ", x)
    x = re.sub(SUFFIX, " ", x)
    x = re.sub(r"[^a-z0-9 ]", " ", x)
    return re.sub(r"\s+", " ", x).strip()


xl = pd.ExcelFile(SRC, engine="pyxlsb")
print(f"sheets: {xl.sheet_names}")
parts = []
for s in xl.sheet_names:
    d = pd.read_excel(SRC, engine="pyxlsb", sheet_name=s)
    d.columns = [str(c).strip() for c in d.columns]
    parts.append(d)
    print(f"  {s}: {len(d):,} rows", flush=True)

R = pd.concat(parts, ignore_index=True)
R = R.rename(columns={
    "GHGRP FACILITY ID": "FacilityId", "REPORTING YEAR": "year",
    "PARENT COMPANY NAME": "parent_name",
    "PARENT CO. PERCENT OWNERSHIP": "pct_ownership",
    "PARENT CO. STATE": "parent_state", "FRS ID (FACILITY)": "FRSId_parentfile",
    "FACILITY NAICS CODE": "naics_parentfile"})
for c in ("FacilityId", "year", "pct_ownership"):
    R[c] = pd.to_numeric(R[c], errors="coerce")
R = R[R.FacilityId.notna() & R.year.notna()].copy()
R["FacilityId"] = R.FacilityId.astype(int)
R["year"] = R.year.astype(int)
R["parent_name"] = R.parent_name.astype(str).str.strip()
R["parent_name_clean"] = R.parent_name.map(clean_name)

# is_primary = largest stake within facility-year; ties broken by name for determinism
R = R.sort_values(["FacilityId", "year", "pct_ownership", "parent_name"],
                  ascending=[True, True, False, True])
R["is_primary"] = ~R.duplicated(["FacilityId", "year"])

keep = ["FacilityId", "year", "parent_name", "parent_name_clean", "pct_ownership",
        "parent_state", "is_primary", "FRSId_parentfile", "naics_parentfile"]
L = R[keep]
L.to_csv(OUT / "link_parent.csv", index=False)

print("\n" + "=" * 70)
print(f"link_parent.csv: {len(L):,} rows  |  "
      f"{L.FacilityId.nunique():,} facilities  |  years {L.year.min()}-{L.year.max()}")
n_par = R.groupby(["FacilityId", "year"]).size()
print(f"\nparents per facility-year: 1 in {(n_par==1).mean()*100:.1f}% of cases, "
      f"max {n_par.max()}")
print(f"facility-years with >1 parent (joint ventures): {(n_par>1).sum():,}")
own = R.groupby(["FacilityId", "year"]).pct_ownership.sum()
print(f"ownership sums to 100 +/- 1 in {((own-100).abs()<=1).mean()*100:.1f}% of "
      f"facility-years")

# ---------------------------------------------------------------- coverage
de = load_cached()
key = de[["FacilityId", "year", "emissions", "threshold_bound"]].copy()
m = key.merge(L[L.is_primary][["FacilityId", "year", "parent_name"]],
              on=["FacilityId", "year"], how="left")
m["has_parent"] = m.parent_name.notna()
rows = []
for lab, sub in [("all facility-years", m),
                 ("threshold-bound only", m[m.threshold_bound])]:
    rows.append(dict(sample=lab, n=len(sub),
                     matched=int(sub.has_parent.sum()),
                     pct_by_count=round(sub.has_parent.mean() * 100, 1),
                     pct_by_emissions=round(
                         sub.loc[sub.has_parent, "emissions"].sum()
                         / max(sub.emissions.sum(), 1) * 100, 1)))
C = pd.DataFrame(rows)
C.to_csv(OUT / "t60_parent_coverage.csv", index=False)
print("\ncoverage against the emissions panel (the discipline: report emissions-weighted too):")
print(C.to_string(index=False))

print("\nmissing by year (facility-years in the panel with no parent record):")
byy = m.groupby("year").has_parent.mean().mul(100).round(1)
print(byy.to_string())

print("\ntop 12 parents by facility count (primary only, 2023):")
t = L[(L.is_primary) & (L.year == 2023)].parent_name.value_counts().head(12)
print(t.to_string())
print(f"\nwrote {OUT/'link_parent.csv'}, {OUT/'t60_parent_coverage.csv'}")


# ---------------------------------------------------------------- 4 · caveats
print("\n" + "=" * 70)
print("caveats worth carrying into the write-up")
print("=" * 70)
n_par = R.groupby(["FacilityId", "year"]).size()
big = n_par[n_par > 10]
print(f"  facility-years with >10 reported parents: {len(big):,} "
      f"(max {n_par.max()})")
print("  these are oil and gas units with many small working-interest owners --")
print("  the parent list includes named individuals. Real data, not a parsing bug.")
prim = R[R.is_primary]
print(f"\n  the PRIMARY parent's stake: median {prim.pct_ownership.median():.0f}%, "
      f"{(prim.pct_ownership >= 50).mean()*100:.1f}% hold a majority, "
      f"{(prim.pct_ownership < 20).mean()*100:.1f}% hold under 20%")
print("  => for a facility with many owners, `is_primary` can be a small stake.")
print("     Any firm-level analysis should either restrict to pct_ownership >= 50")
print("     or apportion, and should say which.")
dup = R.groupby(["FacilityId", "year"]).parent_name.agg(["size", "nunique"])
print(f"\n  facility-years where the same parent name appears twice: "
      f"{int((dup['size'] > dup['nunique']).sum()):,} -- EPA duplicates, harmless "
      f"for is_primary but de-duplicate before summing ownership.")
