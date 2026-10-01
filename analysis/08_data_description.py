"""
Descriptive statistics and sample rows for the data added in the 20 August round.

Produces the tables shown in the update page:
  t13_new_data_summary.csv   one row per source
  t14_state_vs_federal.csv   emissions distribution, CA / WA / federal
  t15_state_head.csv         first rows of the two state panels
  t16_echo_summary.csv       enforcement history of the matched facilities
"""
import warnings
warnings.filterwarnings("ignore")
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from ghgrp_load import load_cached

OUT = "output"
de = load_cached()

ca = pd.read_csv(f"{OUT}/ca_mrr_panel.csv")
wa = pd.read_csv(f"{OUT}/wa_ghg_panel.csv")
wa = wa[wa["Reporter Type"] == "Facility"]
echo = pd.read_csv(f"{OUT}/echo_matched.csv", dtype=str, low_memory=False)
for c in ["FAC_INSPECTION_COUNT", "FAC_FORMAL_ACTION_COUNT",
          "FAC_PENALTY_COUNT", "FAC_TOTAL_PENALTIES"]:
    echo[c] = pd.to_numeric(echo[c], errors="coerce").fillna(0)

# ------------------------------------------------------------------ 1 · coverage
rows = [
    dict(source="GHGRP federal panel (reference)", period="2010–2023",
         observations=f"{len(de):,} facility-years", units=f"{de.FacilityId.nunique():,} facilities",
         threshold="25,000 tCO2e"),
    dict(source="California ARB, Mandatory Reporting Regulation", period="2011–2023",
         observations=f"{len(ca):,} facility-years", units=f"{ca.arb_id.nunique():,} facilities",
         threshold="10,000 tCO2e"),
    dict(source="Washington Dept. of Ecology, mandatory GHG reporting", period="2012–2024",
         observations=f"{len(wa):,} facility-years", units=f"{wa.Reporter.nunique():,} facilities",
         threshold="10,000 tCO2e"),
    dict(source="EPA ECHO Exporter (facility compliance history)", period="through 2025",
         observations=f"{len(echo):,} facilities matched", units="84.0% of the panel",
         threshold="—"),
]
S = pd.DataFrame(rows)
S.to_csv(f"{OUT}/t13_new_data_summary.csv", index=False)
print("=== 1 · coverage ===")
print(S.to_string(index=False))

# ------------------------------------------------------------------ 2 · distributions
def describe(x, label):
    x = pd.to_numeric(x, errors="coerce").dropna()
    q = x.quantile([.10, .25, .50, .75, .90])
    return dict(sample=label, n=f"{len(x):,}", mean=f"{x.mean():,.0f}",
                p10=f"{q[.10]:,.0f}", p25=f"{q[.25]:,.0f}", median=f"{q[.50]:,.0f}",
                p75=f"{q[.75]:,.0f}", p90=f"{q[.90]:,.0f}",
                share_10k_25k=f"{x.between(10_000, 25_000).mean():.1%}")


D = pd.DataFrame([
    describe(de.emissions, "GHGRP federal — reports only above 25,000"),
    describe(ca.co2e, "California — reports from 10,000"),
    describe(wa.co2e, "Washington — reports from 10,000"),
])
D.to_csv(f"{OUT}/t14_state_vs_federal.csv", index=False)
print("\n=== 2 · reported emissions, tCO2e ===")
print(D.to_string(index=False))

n_ca = int(ca.co2e.between(10_000, 25_000).sum())
n_wa = int(wa.co2e.between(10_000, 25_000).sum())
print(f"\nfacility-years in the 10,000–25,000 band: California {n_ca:,}, Washington {n_wa:,}")
print("The federal panel contains this band only for facilities already reporting, "
      "which is the truncation the state data removes.")

# ------------------------------------------------------------------ 3 · sample rows
h_ca = (ca.sort_values(["year", "co2e"], ascending=[False, False])
          .loc[:, ["arb_id", "name", "year", "co2e"]].head(5)
          .rename(columns={"arb_id": "facility id", "co2e": "emissions (tCO2e)"}))
h_wa = (wa.sort_values(["year", "co2e"], ascending=[False, False])
          .loc[:, ["Reporter", "Sector", "year", "co2e"]].head(5)
          .rename(columns={"Reporter": "facility", "co2e": "emissions (tCO2e)"}))
h_ca.to_csv(f"{OUT}/t15_state_head.csv", index=False)
print("\n=== 3 · first rows, California ===")
print(h_ca.to_string(index=False))
print("\n=== first rows, Washington ===")
print(h_wa.to_string(index=False))

# ------------------------------------------------------------------ 4 · ECHO
E = pd.DataFrame([
    dict(measure="ever inspected", value=f"{(echo.FAC_INSPECTION_COUNT > 0).mean():.1%}"),
    dict(measure="ever a formal enforcement action", value=f"{(echo.FAC_FORMAL_ACTION_COUNT > 0).mean():.1%}"),
    dict(measure="ever penalised", value=f"{(echo.FAC_PENALTY_COUNT > 0).mean():.1%}"),
    dict(measure="median cumulative penalty among those penalised",
         value=f"${echo.loc[echo.FAC_TOTAL_PENALTIES > 0, 'FAC_TOTAL_PENALTIES'].median():,.0f}"),
    dict(measure="Title V major sources",
         value=f"{echo.CAA_PERMIT_TYPES.fillna('').str.contains('Major Emissions').mean():.1%}"),
])
E.to_csv(f"{OUT}/t16_echo_summary.csv", index=False)
print("\n=== 4 · ECHO, matched facilities ===")
print(E.to_string(index=False))
print("\nwrote t13–t16")
