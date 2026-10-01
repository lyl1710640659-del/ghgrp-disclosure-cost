"""
35 · EIA-923: was the plant still GENERATING, not merely still listed   (2026-09-28)
================================================================================
Script 32 calls a GHGRP exit a "disclosure exit" when the linked EIA plant still has
operable generators the year after the facility stops reporting. Operable is a CAPACITY
concept: a plant can sit on the operable list at zero output for years. The stronger
question is whether it was still producing.

EIA-923 Schedule 2/3/4/5, Page 1 "Generation and Fuel Data", carries annual
`Net Generation (Megawatthours)` by plant. This file builds a plant x year generation
panel so script 32 can be re-run on an OUTPUT test instead of a capacity test.

FILE LAYOUT
  the header is not the first row -- four to six lines of preamble, and the exact
  count moves between years, so the header row is located by looking for "Plant Id".
  2010 is .xls with a different file name; 2013+ are .xlsx.
  Net generation is reported per plant x fuel x prime mover, so rows are summed to the
  plant. Negative values occur (pumped storage, plant load) and are kept as reported.

outputs  output/eia923_plant_year.csv
run      python3 35_eia923_generation.py --budget 150      (resumable, from analysis/)
"""
import re, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
OUT = Path("output")
D923 = Path("data/eia/923")
YEARS = range(2010, 2024)
CACHE = OUT / "_eia923_cache"
CACHE.mkdir(parents=True, exist_ok=True)


def norm(c):
    return re.sub(r"[^a-z0-9]", "", str(c).lower())


def sched_file(y):
    d = D923 / f"f923_{y}"
    c = [f for f in d.iterdir() if re.search(r"2_3_4_5|2_3_4_5", f.name, re.I)
         and f.suffix.lower() in (".xls", ".xlsx") and not f.name.startswith("~")]
    if not c:
        c = [f for f in d.iterdir() if re.search(r"SCHEDULES? 2", f.name, re.I)]
    return sorted(c)[0]


def page1(xl):
    for s in xl.sheet_names:
        k = norm(s)
        if k.startswith("page1") and "generation" in k:
            return s
    return xl.sheet_names[0]


def read_page1(path, sheet):
    """Locate the header row by looking for a cell equal to 'Plant Id'."""
    probe = pd.read_excel(path, sheet_name=sheet, header=None, nrows=12)
    hdr = None
    for i in range(len(probe)):
        vals = [norm(v) for v in probe.iloc[i].tolist()[:6]]
        if "plantid" in vals:
            hdr = i
            break
    if hdr is None:
        raise RuntimeError(f"no 'Plant Id' header row in {path}::{sheet}")
    d = pd.read_excel(path, sheet_name=sheet, skiprows=hdr)
    keys = {norm(c): c for c in d.columns}
    pid = [o for k, o in keys.items() if k == "plantid"][0]
    # the ANNUAL net generation column, not the twelve monthly ones
    MONTHS = ("january", "february", "march", "april", "may", "june", "july",
              "august", "september", "october", "november", "december")
    gen = [o for k, o in keys.items()
           if "netgeneration" in k and "megawatthours" in k
           and not any(m in k for m in MONTHS)]
    if not gen:
        gen = [o for k, o in keys.items() if k.startswith("netgeneration")
               and not any(m in k for m in MONTHS)]
    if not gen:
        raise RuntimeError(f"no annual net generation column in {path}::{sheet}\n"
                           f"  candidates: {[k for k in keys if 'netgen' in k][:6]}")
    out = pd.DataFrame({"plant_code": d[pid], "net_gen_mwh": d[gen[0]]})
    name = [o for k, o in keys.items() if k == "plantname"]
    if name:
        out["plant_name_923"] = d[name[0]]
    return out


BUDGET = float(sys.argv[sys.argv.index("--budget") + 1]) if "--budget" in sys.argv else 150.0
t0 = time.time()
todo = [y for y in YEARS if not (CACHE / f"gen923_{y}.csv").exists()]
print(f"{len(list(YEARS))-len(todo)}/{len(list(YEARS))} years cached, {len(todo)} to go")
for y in todo:
    if time.time() - t0 > BUDGET:
        print("budget reached; run again to continue")
        break
    ts = time.time()
    f = sched_file(y)
    xl = pd.ExcelFile(f)
    d = read_page1(f, page1(xl))
    d["year"] = y
    d.to_csv(CACHE / f"gen923_{y}.csv", index=False)
    print(f"  {y}  {f.name[:46]:46s} {len(d):>7,} rows  {time.time()-ts:4.0f}s", flush=True)

done = sorted(CACHE.glob("gen923_*.csv"))
if len(done) < len(list(YEARS)):
    print(f"\n{len(done)}/{len(list(YEARS))} cached. Run again.")
    sys.exit(0)

R = pd.concat([pd.read_csv(f) for f in done], ignore_index=True)
R["plant_code"] = pd.to_numeric(R.plant_code, errors="coerce")
R["net_gen_mwh"] = pd.to_numeric(R.net_gen_mwh, errors="coerce")
R = R[R.plant_code.notna()]
R["plant_code"] = R.plant_code.astype(int)

G = (R.groupby(["plant_code", "year"], as_index=False)
       .agg(net_gen_mwh=("net_gen_mwh", "sum"), n_rows=("net_gen_mwh", "size")))
G["generating"] = G.net_gen_mwh > 0
G.to_csv(OUT / "eia923_plant_year.csv", index=False)

print(f"\nplant x year generation panel: {len(G):,} rows, "
      f"{G.plant_code.nunique():,} plants, {G.year.min()}-{G.year.max()}")
print(f"  plant-years with positive net generation: {G.generating.mean()*100:.1f}%")
print(f"  plant-years at exactly zero:              {(G.net_gen_mwh == 0).mean()*100:.1f}%")
print(f"  plant-years negative (pumped storage, plant load): "
      f"{(G.net_gen_mwh < 0).mean()*100:.1f}%")

PY = pd.read_csv(OUT / "eia860_plant_year.csv")
m = PY.merge(G[["plant_code", "year", "generating"]], on=["plant_code", "year"], how="left")
m["generating"] = m.generating.fillna(False)
print("\n>> how much does 'operable' overstate 'generating'?")
print(f"   plant-years operable in EIA-860:            {len(m):,}")
print(f"   of those, generating in EIA-923:            "
      f"{m.generating.mean()*100:.1f}%")
print(f"   operable but NOT generating (or not in 923): "
      f"{(~m.generating).sum():,}")
print(f"\nwrote {OUT/'eia923_plant_year.csv'}")
