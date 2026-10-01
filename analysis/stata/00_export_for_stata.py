"""Export the panels Stata needs. Run from analysis/ :  python3 stata/00_export_for_stata.py"""
from pathlib import Path
import pandas as pd, numpy as np

OUT = Path("output"); DTA = Path("stata")
DTA.mkdir(exist_ok=True)

ca = pd.read_csv(OUT / "ca_mrr_panel.csv")
wa = pd.read_csv(OUT / "wa_ghg_panel.csv")
wa = wa[wa["Reporter Type"] == "Facility"]
ca = pd.DataFrame({"co2e": pd.to_numeric(ca.co2e, errors="coerce"),
                   "year": ca.year, "src": "CA"})
wa = pd.DataFrame({"co2e": pd.to_numeric(wa.co2e, errors="coerce"),
                   "year": wa.year, "src": "WA"})
x = pd.concat([ca, wa], ignore_index=True).dropna(subset=["co2e"])
x = x[x.co2e > 0]
x["src"] = x.src.astype("category")
# 25,000 tCO2e triggers an ALLOWANCE-SURRENDER obligation, not just reporting, in:
#   CA from data year 2013  (17 CCR 95812, Cap-and-Trade covered entity)
#   WA from data year 2023  (Climate Commitment Act cap-and-invest, ch. 173-446 WAC)
# The WA leg was missing before 2026-09-12; WA 2023-24 was coded 0.
x["capandtrade"] = (((x.src == "CA") & (x.year >= 2013)) |
                    ((x.src == "WA") & (x.year >= 2023))).astype(int)

# finer regime label: what does 25,000 actually mean in this state-year?
def _regime(r):
    if r.src == "CA":
        return "CA_pre" if r.year <= 2012 else "CA_cat"      # cap-and-trade binding 2013+
    return "WA_clean" if r.year <= 2022 else "WA_cca"        # CCA binding 2023+
x["regime"] = x.apply(_regime, axis=1)
x.to_stata(DTA / "state_panel.dta", write_index=False, version=118)
print(f"state_panel.dta  {len(x):,} rows  (CA {(x.src=='CA').sum():,}, WA {(x.src=='WA').sum():,})")

# federal panel, for the dominated-region check and as a reference density
import sys; sys.path.insert(0, ".")
from ghgrp_load import load_cached
de = load_cached()
f = de.loc[de.threshold_bound & (de.emissions > 0),
           ["FacilityId", "year", "emissions", "naics3", "state"]].copy()
f = f.rename(columns={"emissions": "co2e"})
f["naics3"] = pd.to_numeric(f.naics3, errors="coerce")
f.to_stata(DTA / "federal_tb.dta", write_index=False, version=118)
print(f"federal_tb.dta   {len(f):,} rows")
