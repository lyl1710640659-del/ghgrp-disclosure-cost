"""
30 · EIA-860: who actually shut down                                    (2026-09-28)
================================================================================
THREAT #1, the only one left that can knock a claim over.

The paper's LOWER bound says firms pay something to get out of reporting: they run the
40 CFR 98.2(i) clock (five years under 25,000, or three under 15,000) and then stop
reporting. The objection is obvious and, until now, unanswerable with GHGRP data alone:

    a facility that stops reporting because it SHUT DOWN is not evidence of anything.

EIA-860 Schedule 3 carries a generator-level `Retirement Year`. For any GHGRP facility
that is also an EIA power plant, it separates the two cases:

    GHGRP report stops, EIA says every generator retired      -> physical closure
    GHGRP report stops, EIA generators still operable         -> DISCLOSURE EXIT

This file builds the EIA side only. Linking to GHGRP FacilityId is script 31.

FILE FORMAT CHANGES ACROSS YEARS (all handled here)
  2010  GeneratorsY2010.xls    sheet 'Ret_IP'              header row 0, UPPER_SNAKE
  2011  GeneratorY2011.xlsx    sheet 'retired & canceled'  header row 1, UPPER_SNAKE
  2012  GeneratorY2012.xlsx    sheet 'Retired and Canceled' header row 1, Title Case
  2013+ 3_1_Generator_Y<y>.xlsx sheet 'Retired and Canceled' header row 1, Title Case
  The operable sheet is 'Exist'/'operable'/'Operable' over the same years.

outputs  output/eia860_generators.csv      generator x 860-vintage, long
         output/eia860_plants.csv          plant code -> name, state, lat, lon
         output/eia860_plant_status.csv    plant-level retirement summary
run      python3 30_eia860_retirement.py            (from analysis/)
"""
import re, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
OUT = Path("output")
D860 = Path("data/eia/860")
YEARS = range(2010, 2024)


def norm(c):
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


WANT = {                      # canonical -> matcher on the normalised header
    "plant_code":      lambda k: k in ("plantcode", "plantid"),
    "plant_name":      lambda k: k == "plantname",
    "utility_name":    lambda k: k == "utilityname",
    "state":           lambda k: k == "state",
    "county":          lambda k: k == "county",
    "generator_id":    lambda k: k == "generatorid",
    "prime_mover":     lambda k: k in ("primemover", "primemovercode"),
    "status":          lambda k: k == "status",
    "nameplate_mw":    lambda k: k in ("nameplate", "nameplatecapacitymw"),
    "operating_year":  lambda k: k == "operatingyear",
    "retirement_year": lambda k: k in ("retirementyear", "plannedretirementyear"),
    "energy_source":   lambda k: k in ("energysource1", "energysource"),
}


def find_sheet(xl, kind):
    """kind = 'ret' or 'op'. Sheet names change wording every couple of years."""
    for s in xl.sheet_names:
        k = norm(s)
        if kind == "ret" and ("retired" in k or k == "retip"):
            return s
        if kind == "op" and (k in ("operable", "exist") or k.startswith("operable")):
            return s
    return None


def read_sheet(path, sheet):
    """Header is on row 0 or row 1 depending on the year; pick whichever yields a
    recognisable Plant Code column."""
    for sk in (0, 1, 2):
        d = pd.read_excel(path, sheet_name=sheet, skiprows=sk)
        keys = {norm(c): c for c in d.columns}
        if any(WANT["plant_code"](k) for k in keys):
            out = {}
            for canon, test in WANT.items():
                hit = [orig for k, orig in keys.items() if test(k)]
                if hit:
                    out[canon] = d[hit[0]]
            return pd.DataFrame(out)
    raise RuntimeError(f"no usable header in {path}::{sheet}")


def gen_file(y):
    d = D860 / f"eia860{y}"
    cands = [f for f in d.iterdir()
             if "enerator" in f.name and f.suffix in (".xls", ".xlsx")
             and not f.name.startswith("~")]
    if not cands:
        raise FileNotFoundError(f"no generator file for {y}")
    return sorted(cands)[0]


def plant_file(y):
    d = D860 / f"eia860{y}"
    cands = [f for f in d.iterdir()
             if re.match(r"^(2___)?Plant", f.name) and f.suffix in (".xls", ".xlsx")
             and not f.name.startswith("~")]
    return sorted(cands)[0] if cands else None


# ---------------------------------------------------------------- 1 · generators
# Reading 14 years x 2 sheets of xlsx through openpyxl does not finish inside the
# device shell's three-minute cap, so each year is cached to its own CSV and the run
# stops when the budget is spent. Call the script again to continue.
import time
CACHE = OUT / "_eia860_cache"
CACHE.mkdir(exist_ok=True)
BUDGET = float(sys.argv[sys.argv.index("--budget") + 1]) if "--budget" in sys.argv else 150.0
t0 = time.time()

todo = [y for y in YEARS if not (CACHE / f"gen_{y}.csv").exists()]
print(f"{len(YEARS)-len(todo)}/{len(YEARS)} years cached, {len(todo)} to go")
for y in todo:
    if time.time() - t0 > BUDGET:
        print(f"budget reached; {len([z for z in todo if not (CACHE/f'gen_{z}.csv').exists()])} left")
        break
    ts = time.time()
    f = gen_file(y)
    xl = pd.ExcelFile(f)
    parts = []
    for kind in ("ret", "op"):
        sh = find_sheet(xl, kind)
        if sh is None:
            print(f"  {y}: no '{kind}' sheet in {f.name} ({xl.sheet_names})")
            continue
        d = read_sheet(f, sh)
        d["sheet"] = "retired" if kind == "ret" else "operable"
        d["vintage"] = y
        parts.append(d)
    pd.concat(parts, ignore_index=True).to_csv(CACHE / f"gen_{y}.csv", index=False)
    print(f"  {y}  {f.name:32s} {time.time()-ts:5.0f}s", flush=True)

done = sorted(CACHE.glob("gen_*.csv"))
if len(done) < len(list(YEARS)):
    print(f"\n{len(done)}/{len(list(YEARS))} years cached. Run again to continue.")
    sys.exit(0)

G = pd.concat([pd.read_csv(f) for f in done], ignore_index=True)
for c in ("plant_code", "nameplate_mw", "operating_year", "retirement_year"):
    if c in G:
        G[c] = pd.to_numeric(G[c], errors="coerce")
G = G[G.plant_code.notna()].copy()
G["plant_code"] = G.plant_code.astype(int)
G["generator_id"] = G.generator_id.astype(str).str.strip()
G.to_csv(OUT / "eia860_generators.csv", index=False)
print(f"\ngenerators: {len(G):,} rows, {G.plant_code.nunique():,} plants, "
      f"vintages {G.vintage.min()}-{G.vintage.max()}")
print(G.groupby(["sheet"]).size().to_string())
print("\nretirement_year present on the retired sheet: "
      f"{G[G.sheet=='retired'].retirement_year.notna().mean()*100:.1f}%")


# ---------------------------------------------------------------- 2 · plants
# The Plant schedule carries latitude/longitude, which script 31 needs to match EIA
# plant codes to GHGRP facilities when the EPA crosswalk does not cover one.
PWANT = {
    "plant_code": lambda k: k in ("plantcode", "plantid"),
    "plant_name": lambda k: k == "plantname",
    "state":      lambda k: k == "state",
    "county":     lambda k: k == "county",
    "lat":        lambda k: k == "latitude",
    "lon":        lambda k: k == "longitude",
    "zip":        lambda k: k in ("zip", "zip5", "zipcode"),
    "naics":      lambda k: k in ("naicscode", "primarypurposenaicscode",
                                  "primarypurpose", "primarypurposenaics"),
}

prows = []
for y in YEARS:
    pf = plant_file(y)
    if pf is None:
        print(f"  plants {y}: none found"); continue
    got = None
    for sk in (0, 1, 2):
        d = pd.read_excel(pf, skiprows=sk)
        keys = {norm(c): c for c in d.columns}
        if any(PWANT["plant_code"](k) for k in keys):
            got = pd.DataFrame({c: d[[o for k, o in keys.items() if t(k)][0]]
                                for c, t in PWANT.items()
                                if any(t(k) for k in keys)})
            break
    if got is None:
        print(f"  plants {y}: no header in {pf.name}"); continue
    got["vintage"] = y
    prows.append(got)
    print(f"  plants {y}  {pf.name}", flush=True)

P = pd.concat(prows, ignore_index=True)
P["plant_code"] = pd.to_numeric(P.plant_code, errors="coerce")
P = P[P.plant_code.notna()]
P["plant_code"] = P.plant_code.astype(int)
for c in ("lat", "lon"):
    if c in P:
        P[c] = pd.to_numeric(P[c], errors="coerce")
# one row per plant: the most recent vintage that has coordinates
P = P.sort_values(["plant_code", "vintage"])
PL = (P[P.lat.notna() & P.lon.notna()].groupby("plant_code").tail(1)
      if "lat" in P else P.groupby("plant_code").tail(1))
miss = P.groupby("plant_code").tail(1)
PL = pd.concat([PL, miss[~miss.plant_code.isin(PL.plant_code)]], ignore_index=True)
PL.to_csv(OUT / "eia860_plants.csv", index=False)
print(f"\nplants: {len(PL):,} unique plant codes, "
      f"{PL.lat.notna().mean()*100:.1f}% with coordinates")


# ---------------------------------------------------------------- 3 · plant x year
# operable_mw[p, y] comes from the OPERABLE sheet of the vintage-y file, so it is a
# genuine annual panel of "did this plant still have generating units that year".
# Status codes on that sheet are OP / SB / OS / OA -- operating, standby, out of
# service, out of service but expected to return. All four mean the plant exists.
op = G[G.sheet == "operable"]
panel = (op.groupby(["plant_code", "vintage"])
           .agg(operable_gens=("generator_id", "size"),
                operable_mw=("nameplate_mw", "sum"),
                op_status_OP=("status", lambda s: int((s == "OP").sum())))
           .reset_index().rename(columns={"vintage": "year"}))

# retirements: status RE only. IP (indefinitely postponed) and CN (cancelled) are
# units that were never built -- they carry no retirement year and are not closures.
ret = G[(G.sheet == "retired") & (G.status == "RE") & G.retirement_year.notna()]
ret = ret.drop_duplicates(["plant_code", "generator_id", "retirement_year"])
ret_by_plant = (ret.groupby("plant_code")
                  .agg(n_retired_gens=("generator_id", "nunique"),
                       retired_mw=("nameplate_mw", "sum"),
                       first_retire_year=("retirement_year", "min"),
                       last_retire_year=("retirement_year", "max"))
                  .reset_index())

last_op = (panel[panel.operable_mw > 0].groupby("plant_code")
           .agg(last_year_operable=("year", "max"),
                first_year_operable=("year", "min"),
                max_operable_mw=("operable_mw", "max")).reset_index())

S = last_op.merge(ret_by_plant, on="plant_code", how="outer").merge(
    PL[["plant_code", "plant_name", "state", "lat", "lon"]].drop_duplicates("plant_code"),
    on="plant_code", how="left")

# A plant is FULLY retired when it has retirements and stops appearing as operable
# before the panel ends. If it is still operable in 2023 it never closed, whatever
# individual units did.
S["still_operable_2023"] = S.last_year_operable.eq(YEARS[-1])
S["plant_closed"] = S.last_year_operable.notna() & ~S.still_operable_2023
S["closure_year"] = np.where(S.plant_closed, S.last_year_operable + 1, np.nan)
S.to_csv(OUT / "eia860_plant_status.csv", index=False)
panel.to_csv(OUT / "eia860_plant_year.csv", index=False)

print(f"\nplant x year panel: {len(panel):,} rows")
print(f"plant status table: {len(S):,} plants")
print(f"  still operable in 2023        {int(S.still_operable_2023.sum()):,}")
print(f"  stopped appearing as operable {int(S.plant_closed.sum()):,}")
print(f"  have at least one RE unit     {int(S.n_retired_gens.notna().sum()):,}")
print("\n>> the number that matters for threat #1: plants with retired units that are")
print("   STILL OPERABLE in 2023 -- partial retirement, not closure:")
both = S[S.n_retired_gens.notna() & S.still_operable_2023]
print(f"   {len(both):,} plants ({len(both)/max(S.n_retired_gens.notna().sum(),1)*100:.0f}% "
      f"of all plants with a retirement)")
print(f"\nwrote {OUT/'eia860_plants.csv'}, {OUT/'eia860_plant_status.csv'}, "
      f"{OUT/'eia860_plant_year.csv'}")
