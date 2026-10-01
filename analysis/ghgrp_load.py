"""Shared GHGRP loader. Imported by both notebooks so the sample is identical across them.

    from ghgrp_load import load_ghgrp, add_eligibility
    de = load_ghgrp()          # stacks data/ghgp_data_2010.xlsx ... 2023.xlsx

WHY THE PER-YEAR FILES AND NOT `ghgp_data_by_year_2023.xlsx`
------------------------------------------------------------
The multi-year workbook is convenient but worse in three ways:

  1. It starts in **2011**. The per-year files give you **2010**.
  2. Its subpart column is `Latest Reported Industry Type (subparts)` -- one time-invariant
     value stamped on every year. Facilities change subpart. Using the latest value for all
     years mismeasures `always_covered`, which is the placebo group the whole design leans on.
     The per-year files carry `Industry Type (subparts)` **as reported that year**.
  3. It has total emissions only. The per-year files add emissions by gas, emissions by
     subpart, and -- crucially -- `Does the facility employ continuous emissions monitoring?`
     (the CEMS flag), which is the manipulation-vs-real-abatement robustness check.

Sheet naming changed midway: 2010-2017 use "Direct Emitters", 2018+ use "Direct Point
Emitters". Handled below.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
DATA = _HERE / "data"

# ---------------------------------------------------------------------------
# Source categories that must report REGARDLESS of the 25,000 tCO2e threshold.
#
# CORRECTED 2026-09-12. The previous list was D/F/G/H/HH -- five letters, chosen by hand
# and never checked against the rule. 40 CFR 98 Table A-3 has nineteen entries that touch
# this panel. The omission mattered: 248 facilities classified as threshold_bound first
# appeared BELOW 25,000 tCO2e, which a genuinely threshold-bound facility cannot do
# (see 16_new_arrivals.py section 4b). Subparts FF, and a mis-set of others, were the cause.
#
# Table A-3 (40 CFR 98), mapped to subpart letters using the "Industry Type" lookup sheet
# of the EPA workbooks themselves:
#
#   RY2010+   D  electricity generation (part 75 reporters)
#             E  adipic acid            F  aluminum         G  ammonia
#             H  cement                 O  HCFC-22 production / HFC-23 destruction
#             S  lime                   V  nitric acid      X  petrochemical
#             Y  petroleum refineries   Z  phosphoric acid  BB silicon carbide
#             CC soda ash               EE titanium dioxide
#   RY2011+   RR geologic sequestration of CO2              UU injection of CO2
#
# Three Table A-3 categories carry their OWN category-specific threshold rather than
# reporting unconditionally. They are still observed below 25,000 tCO2e of TOTAL
# emissions -- which is what the design needs from the placebo group -- so they are in
# ALWAYS_COVERED, but out of ALWAYS_COVERED_STRICT:
#
#   HH  municipal landfills            CH4 >= 25,000 tCO2e
#   DD  SF6 from electrical equipment  fluorinated GHG >= 25,000 tCO2e
#   FF  underground coal mines         >= 36,500,000 actual cubic feet CH4/yr
#
# Table A-3 also lists manure management (subpart JJ) and, from RY2025, VV/WW/XX/YY.
# Neither appears in this 2010-2023 Direct-Emitters panel.
#
# NOT in Table A-3, so bound by the 25,000 facility-wide threshold (Table A-4):
#   C I K L N P Q R T U W(all) AA GG II SS TT
#
# >> STILL OPEN: 217 subpart-C-only facilities first appear below 25,000 tCO2e. C is
#    Table A-4, so Table A-3 does not explain them. Partial-year reporting, a different
#    applicability basis, and revisions are the candidates. Do not claim the below-25,000
#    region is fully explained until this is resolved.
# ---------------------------------------------------------------------------
TABLE_A3_UNCONDITIONAL = ["D", "E", "F", "G", "H", "O", "S", "V", "X", "Y", "Z",
                          "BB", "CC", "EE", "RR", "UU"]
TABLE_A3_CONDITIONAL = ["HH", "DD", "FF"]

ALWAYS_COVERED = TABLE_A3_UNCONDITIONAL + TABLE_A3_CONDITIONAL
ALWAYS_COVERED_STRICT = TABLE_A3_UNCONDITIONAL

# the pre-2026-09-12 list, kept so every result can be diffed against what it replaced
ALWAYS_COVERED_LEGACY = ["D", "F", "G", "H", "HH"]

EM_COL = "Total reported direct emissions"
SUBPART_COL = "Industry Type (subparts)"
SECTOR_COL = "Industry Type (sectors)"
CEMS_COL = "Does the facility employ continuous emissions monitoring?"

GAS_COLS = [
    "CO2 emissions (non-biogenic)", "Methane (CH4) emissions",
    "Nitrous Oxide (N2O) emissions", "HFC emissions", "PFC emissions",
    "SF6 emissions", "NF3 emissions", "Biogenic CO2 emissions (metric tons)",
]

ID_COLS = ["Facility Id", "FRS Id", "Facility Name", "City", "State", "Zip Code",
           "Address", "County", "Latitude", "Longitude", "Primary NAICS Code"]

RENAME = {
    "Facility Id": "FacilityId", "FRS Id": "FRSId", "Facility Name": "facility_name",
    "City": "city", "State": "state", "Zip Code": "zip", "County": "county",
    "Latitude": "lat", "Longitude": "lon", "Primary NAICS Code": "naics",
    SUBPART_COL: "subparts", SECTOR_COL: "sectors", EM_COL: "emissions", CEMS_COL: "cems",
}


def _direct_sheet(xl):
    """2010-2017: 'Direct Emitters'.  2018+: 'Direct Point Emitters'."""
    cands = [s for s in xl.sheet_names if s.strip().startswith("Direct")]
    if not cands:
        raise ValueError(f"no Direct* sheet among {xl.sheet_names}")
    return cands[0]


def _header_row(path, sheet, probe="Facility Id", nrows=12):
    raw = pd.read_excel(path, sheet_name=sheet, header=None, nrows=nrows)
    hit = raw.apply(lambda r: r.astype(str).str.contains(probe, case=False, na=False).any(), axis=1)
    if not hit.any():
        raise ValueError(f"no header row containing {probe!r} in {path.name}:{sheet}")
    return int(raw.index[hit][0])


def load_year(path, year=None, keep_gases=True):
    path = Path(path)
    if year is None:
        m = re.search(r"(20\d\d)", path.stem)
        year = int(m.group(1))
    xl = pd.ExcelFile(path)
    sheet = _direct_sheet(xl)
    df = pd.read_excel(path, sheet_name=sheet, header=_header_row(path, sheet))
    df.columns = [str(c).strip() for c in df.columns]

    cols = [c for c in ID_COLS + [SUBPART_COL, SECTOR_COL, EM_COL, CEMS_COL] if c in df.columns]
    if keep_gases:
        cols += [c for c in GAS_COLS if c in df.columns]
    df = df[cols].rename(columns=RENAME)
    df["year"] = year
    df["source_file"] = path.name
    return df


def load_ghgrp(data_dir=DATA, years=None, verbose=True):
    data_dir = Path(data_dir)
    files = sorted(f for f in data_dir.glob("ghgp_data_*.xlsx") if "by_year" not in f.name)
    if not files:
        raise FileNotFoundError(f"no per-year ghgp_data_YYYY.xlsx in {data_dir.resolve()}")

    frames = []
    for f in files:
        y = int(re.search(r"(20\d\d)", f.stem).group(1))
        if years and y not in years:
            continue
        d = load_year(f)
        frames.append(d)
        if verbose:
            print(f"  {y}  {len(d):>6,} facilities   ({f.name})")
    de = pd.concat(frames, ignore_index=True)

    # ---- types ----
    de["FacilityId"] = pd.to_numeric(de.FacilityId, errors="coerce").astype("Int64")
    de["emissions"] = pd.to_numeric(de.emissions, errors="coerce")
    de["naics"] = pd.to_numeric(de.naics, errors="coerce").astype("Int64")
    de["naics3"] = de.naics.astype(str).str[:3]
    de = de.dropna(subset=["FacilityId", "emissions"])

    # ---- CEMS flag: stored as Yes/No text ----
    if "cems" in de.columns:
        de["cems"] = (de.cems.astype(str).str.strip().str.lower()
                        .map({"yes": True, "y": True, "no": False, "n": False}))

    # ---- subparts: reported PER YEAR, so always_covered is time-varying ----
    def _sset(s):
        if pd.isna(s):
            return frozenset()
        return frozenset(p.strip().upper() for p in str(s).split(",") if p.strip())

    de["subpart_set"] = de.subparts.map(_sset)
    de["always_covered"] = de.subpart_set.map(lambda s: bool(s & set(ALWAYS_COVERED)))
    de["always_covered_strict"] = de.subpart_set.map(lambda s: bool(s & set(ALWAYS_COVERED_STRICT)))
    de["always_covered_legacy"] = de.subpart_set.map(lambda s: bool(s & set(ALWAYS_COVERED_LEGACY)))
    de["threshold_bound"] = ~de.always_covered
    de["n_subparts"] = de.subpart_set.map(len)

    de["log_em"] = np.log(de.emissions.where(de.emissions > 0))

    # ---- one row per facility-year ----
    dup = de.duplicated(["FacilityId", "year"], keep=False)
    if dup.any():
        if verbose:
            print(f"  ! {dup.sum():,} duplicate facility-years -> keeping max emissions row")
        de = (de.sort_values(["FacilityId", "year", "emissions"])
                .drop_duplicates(["FacilityId", "year"], keep="last"))

    de = de.sort_values(["FacilityId", "year"]).reset_index(drop=True)

    # ---- panel structure ----
    g = de.groupby("FacilityId").year
    de["first_year"] = g.transform("min")
    de["last_year"] = g.transform("max")
    de["n_years"] = g.transform("nunique")
    y0, y1 = de.year.min(), de.year.max()
    # exit = last year observed and not right-censored at the end of the panel
    de["exits"] = (de.year == de.last_year) & (de.last_year < y1)
    # entry = first year observed and not left-censored at program start
    de["enters"] = (de.year == de.first_year) & (de.first_year > y0)
    de["gap_after"] = de.groupby("FacilityId").year.shift(-1) - de.year > 1

    if verbose:
        print(f"\n{len(de):,} facility-years | {de.FacilityId.nunique():,} facilities | {y0}-{y1}")
        print(f"always-covered share: {de.always_covered.mean():.1%} "
              f"(strict, Table A-3 unconditional only: {de.always_covered_strict.mean():.1%})")
        if "cems" in de.columns:
            print(f"CEMS facilities: {de.cems.mean():.1%}")
    return de


def add_eligibility(de, verbose=True):
    """40 CFR 98.2(i): a reporter may cease reporting after 5 consecutive years below
    25,000 tCO2e OR 3 consecutive years below 15,000, and must resume on reaching 25,000.

    Rolling windows are computed on the facility's own reporting sequence. Note this treats
    consecutive *observations* as consecutive years; facilities with gaps are flagged so you
    can decide how to handle them.
    """
    de = de.sort_values(["FacilityId", "year"]).copy()

    u25 = (de.emissions < 25000).astype(float)
    u15 = (de.emissions < 15000).astype(float)
    r25 = u25.groupby(de.FacilityId).rolling(5, min_periods=5).sum().reset_index(level=0, drop=True)
    r15 = u15.groupby(de.FacilityId).rolling(3, min_periods=3).sum().reset_index(level=0, drop=True)

    de["elig_5yr_25k"] = r25.eq(5).fillna(False)
    de["elig_3yr_15k"] = r15.eq(3).fillna(False)
    de["eligible"] = de.elig_5yr_25k | de.elig_3yr_15k

    if verbose:
        n = de.groupby("FacilityId").eligible.any().sum()
        print(f"eligible facility-years: {de.eligible.sum():,} ({de.eligible.mean():.1%})")
        print(f"facilities ever eligible: {n:,} ({n / de.FacilityId.nunique():.1%})")
        print(f"exit rate | eligible: {de.loc[de.eligible,'exits'].mean():.3f}   "
              f"| not eligible: {de.loc[~de.eligible,'exits'].mean():.3f}")
    return de


# --------------------------------------------------------------------------- cache
CACHE = _HERE / "output" / "_panel_cache.pkl"


def load_cached(rebuild=False, verbose=False):
    """Panel + 98.2(i) eligibility, cached to a pickle.

    Stacking fourteen workbooks takes ~30s, which is fine once but tiresome when five
    notebooks each need the same frame. Delete output/_panel_cache.pkl (or pass
    rebuild=True) after changing anything upstream.
    """
    if CACHE.exists() and not rebuild:
        return pd.read_pickle(CACHE)
    de = add_eligibility(load_ghgrp(verbose=verbose), verbose=verbose)
    CACHE.parent.mkdir(exist_ok=True)
    de.to_pickle(CACHE)
    return de


def add_transitions(de):
    """Consecutive-year transitions with log growth. Used in notebooks 02 and 03."""
    de = de.sort_values(["FacilityId", "year"]).copy()
    g = de.groupby("FacilityId")
    de["lag_em"] = g.emissions.shift(1)
    de["lag_yr"] = g.year.shift(1)
    t = de[(de.year - de.lag_yr == 1) & (de.lag_em > 0) & (de.emissions > 0)].copy()
    t["growth"] = np.log(t.emissions) - np.log(t.lag_em)
    return de, t
