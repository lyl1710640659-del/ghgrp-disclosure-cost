"""
31 · Linking GHGRP facilities to EIA-860 plant codes                    (2026-09-28)
================================================================================
EIA's `Plant Code` (ORISPL) and GHGRP's `FacilityId` are different numbering systems.
EPA publishes a crosswalk (Power Plant Crosswalk, on epa.gov/ghgreporting/data-sets)
and that is the right bridge; this file builds the GEOGRAPHIC fallback, which also
serves as the check on the crosswalk once it is downloaded.

METHOD
  nearest EIA plant within MAX_KM of the GHGRP facility's coordinates, scored on name
  similarity as well. Both sides are full populations -- 8,778 GHGRP facilities and
  ~17,000 EIA plants -- so proximity alone is not enough: petrochemical complexes often
  sit next door to the cogeneration plant that serves them, and they are different
  facilities with different obligations.

  confidence   rule
  high         <= 1 km and name >= 0.35, or <= 5 km and name >= 0.60
  medium       <= 5 km, name >= 0.15
  low          <= 5 km, anything else  -- do not use without eyeballing
  none         no EIA plant within 5 km

⚠️ A match here means "this GHGRP facility is (or contains) this EIA power plant".
   EIA-860 covers POWER PLANTS ONLY, so most GHGRP facilities correctly have no match.
   The unmatched are not failures; section 3 reports what fraction of the population
   that actually matters -- the facilities that stop reporting -- is adjudicable.

outputs  output/link_eia.csv
run      python3 31_link_ghgrp_eia.py            (from analysis/)
"""
import re, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from ghgrp_load import load_cached

OUT = Path("output")
MAX_KM = 5.0
STOP = {"llc", "inc", "lp", "co", "company", "corp", "corporation", "ltd", "plant",
        "power", "station", "energy", "generating", "facility", "the", "and", "of",
        "l", "p", "lc", "partners", "holdings", "us", "usa", "operations"}


def toks(s):
    w = re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split()
    return {t for t in w if t not in STOP and len(t) > 1}


def name_score(a, b):
    ta, tb = toks(a), toks(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------- inputs
de = load_cached()
F = (de.sort_values(["FacilityId", "year"])
       .drop_duplicates("FacilityId", keep="last")
       [["FacilityId", "facility_name", "state", "lat", "lon", "naics"]].copy())
F = F[F.lat.notna() & F.lon.notna()]

P = pd.read_csv(OUT / "eia860_plants.csv")
P = P[P.lat.notna() & P.lon.notna()].drop_duplicates("plant_code")
print(f"GHGRP facilities with coordinates: {len(F):,}")
print(f"EIA plants with coordinates:       {len(P):,}")

plat, plon = P.lat.values, P.lon.values
pcode, pname, pstate = P.plant_code.values, P.plant_name.values, P.state.values

rows = []
CH = 400
for i in range(0, len(F), CH):
    blk = F.iloc[i:i + CH]
    d = haversine_km(blk.lat.values[:, None], blk.lon.values[:, None],
                     plat[None, :], plon[None, :])
    j = np.argmin(d, axis=1)
    dm = d[np.arange(len(blk)), j]
    for k, (_, r) in enumerate(blk.iterrows()):
        rows.append(dict(FacilityId=r.FacilityId, facility_name=r.facility_name,
                         ghgrp_state=r.state, naics=r.naics,
                         plant_code=int(pcode[j[k]]), plant_name=pname[j[k]],
                         eia_state=pstate[j[k]], dist_km=float(dm[k]),
                         name_score=name_score(r.facility_name, pname[j[k]])))
    print(f"  {min(i+CH, len(F)):>5}/{len(F)}", end="\r", flush=True)

L = pd.DataFrame(rows)
L["same_state"] = L.ghgrp_state.astype(str).str.upper() == L.eia_state.astype(str).str.upper()


def conf(r):
    if r.dist_km > MAX_KM:
        return "none"
    if (r.dist_km <= 1.0 and r.name_score >= 0.35) or r.name_score >= 0.60:
        return "high"
    if r.name_score >= 0.15:
        return "medium"
    return "low"


L["match_method"] = "geo+name"
L["match_confidence"] = L.apply(conf, axis=1)
L.loc[L.match_confidence == "none", ["plant_code", "plant_name", "eia_state"]] = np.nan
L.to_csv(OUT / "link_eia.csv", index=False)

print("\n\n=== match confidence, all GHGRP facilities ===")
print(L.match_confidence.value_counts().to_string())
print(f"\nof the matched (<= {MAX_KM} km), same state: "
      f"{L[L.match_confidence!='none'].same_state.mean()*100:.1f}%")
print("\nmatch rate by GHGRP NAICS3 (top 10 by facility count):")
L["n3"] = L.naics.astype(str).str[:3]
g = (L.assign(matched=L.match_confidence.isin(["high", "medium"]))
       .groupby("n3").agg(n=("FacilityId", "size"), matched=("matched", "mean"))
       .sort_values("n", ascending=False).head(10))
g["matched"] = (g.matched * 100).round(1)
print(g.to_string())
print(f"\nwrote {OUT/'link_eia.csv'}")
