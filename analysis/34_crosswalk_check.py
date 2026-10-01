"""
34 · Cross-checking the geographic GHGRP<->EIA match against EPA's own crosswalk
================================================================================
                                                                       (2026-09-28)
Script 31 matched GHGRP facilities to EIA plants by coordinates and name because no
official key was in hand. EPA publishes one: the Power Plant Crosswalk, GHGRP Facility
ID -> ORIS code (up to five per facility).

    data/ghgrp_parent/ghgrp_oris_power_plant_crosswalk_12_13_21.xlsx

WHAT THE CROSSWALK IS AND IS NOT (from its own README, worth keeping)
  - An ORIS code is issued by EIA *or* by EPA's Clean Air Markets Division. Most
    coincide with the EIA-860 Plant Code, but a CAMD-only code need not appear in
    EIA-860 at all.
  - A facility may carry several ORIS codes: "individual combustion units at a given
    GHGRP facility may be associated with different ORIS codes".
  - It covers non-power facilities too, e.g. "a refinery that includes a cogeneration
    unit that reports to EIA".
  - ⚠️ It is a snapshot: "GHGRP data was reported to EPA by facilities as of 08/7/2021",
    so facilities that first appear in 2022-2023 are absent by construction. This makes
    it a partial, not a universal, key -- which is why the geographic match stays as the
    fallback rather than being thrown away.

This file (1) measures agreement between the two, (2) builds link_eia_v2.csv preferring
the crosswalk, and (3) re-runs the threat-#1 comparison on it.

outputs  output/link_eia_v2.csv  output/t61_crosswalk_agreement.csv
run      python3 34_crosswalk_check.py           (from analysis/)
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
OUT = Path("output")
XW = Path("data/ghgrp_parent/ghgrp_oris_power_plant_crosswalk_12_13_21.xlsx")

# ---------------------------------------------------------------- 1 · load
raw = pd.read_excel(XW, sheet_name="ORIS Crosswalk")
oris_cols = [c for c in raw.columns if str(c).upper().startswith("ORIS CODE")]
X = raw.melt(id_vars=["GHGRP Facility ID", "FACILITY NAME", "GHGRP - State",
                      "GHGRP - Power Plant Sector"],
             value_vars=oris_cols, value_name="oris").dropna(subset=["oris"])
X = X.rename(columns={"GHGRP Facility ID": "FacilityId",
                      "GHGRP - Power Plant Sector": "power_sector"})
X["FacilityId"] = pd.to_numeric(X.FacilityId, errors="coerce")
X["oris"] = pd.to_numeric(X.oris, errors="coerce")
X = X[X.FacilityId.notna() & X.oris.notna()]
X["FacilityId"] = X.FacilityId.astype(int)
X["oris"] = X.oris.astype(int)
X = X.drop_duplicates(["FacilityId", "oris"])
print(f"crosswalk: {len(raw):,} facilities, {len(X):,} facility-ORIS pairs, "
      f"{X.FacilityId.nunique():,} unique facilities")
print(f"  facilities with more than one ORIS: "
      f"{int((X.groupby('FacilityId').size() > 1).sum())}")

P = pd.read_csv(OUT / "eia860_plants.csv")
eia_codes = set(P.plant_code)
X["in_eia860"] = X.oris.isin(eia_codes)
print(f"  ORIS codes that exist in EIA-860 2010-2023: "
      f"{X.in_eia860.mean()*100:.1f}%  ({int((~X.in_eia860).sum())} do not -- "
      f"CAMD-only codes, or plants retired before 2010)")

L = pd.read_csv(OUT / "link_eia.csv")

# ---------------------------------------------------------------- 2 · agreement
print("\n" + "=" * 78)
print("2 . do the geographic match and EPA's crosswalk agree?")
print("=" * 78)
xw_sets = X[X.in_eia860].groupby("FacilityId").oris.apply(set).to_dict()
G = L[L.match_confidence.isin(["high", "medium", "low"])].copy()
G["plant_code"] = G.plant_code.astype("Int64")
G["xw"] = G.FacilityId.map(xw_sets)
both = G[G.xw.notna() & G.plant_code.notna()].copy()
both["agree"] = [pc in s for pc, s in zip(both.plant_code, both.xw)]
rows = []
for conf in ["high", "medium", "low"]:
    sub = both[both.match_confidence == conf]
    if len(sub):
        rows.append(dict(confidence=conf, n_comparable=len(sub),
                         agree=int(sub.agree.sum()),
                         pct_agree=round(sub.agree.mean() * 100, 1)))
A = pd.DataFrame(rows)
print("\n  among facilities the crosswalk covers AND the geo match found a plant:")
print(A.to_string(index=False))
A.to_csv(OUT / "t61_crosswalk_agreement.csv", index=False)

hm = both[both.match_confidence.isin(["high", "medium"])]
print(f"\n  >> high+medium agreement: {hm.agree.mean()*100:.1f}% "
      f"({int(hm.agree.sum())}/{len(hm)})")
print(f"     low agreement:          "
      f"{both[both.match_confidence=='low'].agree.mean()*100:.1f}% "
      f"-- confirms the decision to exclude 'low'")

print("\n  where high/medium disagrees, how far apart are the two answers?")
bad = hm[~hm.agree].merge(P[["plant_code", "plant_name", "lat", "lon"]],
                          on="plant_code", how="left", suffixes=("", "_geo"))
print(f"    {len(bad)} disagreements; median distance of the geo pick "
      f"{bad.dist_km.median():.2f} km, median name score {bad.name_score.median():.2f}")

# coverage of each key
print("\n  coverage (of all 8,778 GHGRP facilities):")
print(f"    crosswalk with an EIA-860 code : {len(xw_sets):,}")
print(f"    geo high+medium                : "
      f"{int(L.match_confidence.isin(['high','medium']).sum()):,}")
only_xw = set(xw_sets) - set(L[L.match_confidence.isin(["high", "medium"])].FacilityId)
only_geo = set(L[L.match_confidence.isin(["high", "medium"])].FacilityId) - set(xw_sets)
print(f"    crosswalk only (geo missed)    : {len(only_xw):,}")
print(f"    geo only (crosswalk missed)    : {len(only_geo):,}")

# ---------------------------------------------------------------- 3 · link v2
print("\n" + "=" * 78)
print("3 . link_eia_v2: crosswalk first, geography as the fallback")
print("=" * 78)
xw_rows = (X[X.in_eia860].groupby("FacilityId").oris.first().reset_index()
             .rename(columns={"oris": "plant_code"}))
xw_rows["match_method"] = "EPA crosswalk"
xw_rows["match_confidence"] = "crosswalk"
geo = L[L.match_confidence.isin(["high", "medium"])][
    ["FacilityId", "plant_code", "dist_km", "name_score", "match_confidence"]].copy()
geo["match_method"] = "geo+name"
geo = geo[~geo.FacilityId.isin(xw_rows.FacilityId)]
V2 = pd.concat([xw_rows, geo], ignore_index=True)
V2 = V2.merge(P[["plant_code", "plant_name", "state"]].drop_duplicates("plant_code"),
              on="plant_code", how="left")
V2.to_csv(OUT / "link_eia_v2.csv", index=False)
print(V2.match_method.value_counts().to_string())
print(f"\n  total linked facilities: {V2.FacilityId.nunique():,} "
      f"(was {int(L.match_confidence.isin(['high','medium']).sum()):,} on geography alone)")
print(f"  wrote {OUT/'link_eia_v2.csv'}, {OUT/'t61_crosswalk_agreement.csv'}")
