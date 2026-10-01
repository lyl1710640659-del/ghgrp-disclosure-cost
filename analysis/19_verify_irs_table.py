"""
19 · Verify the IRS numbers in t35 against the primary source
================================================================================
Every IRS figure in 18_irs_audit_benchmark.py is a second-hand quotation (Tax Policy
Center, TRAC, an IRS newsroom statement). This script checks them against IRS Data Book
Table 3-1 itself.

BEFORE RUNNING: put the workbook in analysis/data/irs/ -- see the README there.
www.irs.gov is blocked by this session's egress policy, so it has to be downloaded
by hand in a browser.

    https://www.irs.gov/pub/irs-soi/25db-3-01-ex-allyears.xlsx     (TY2010-2023)

run   python3 19_verify_irs_table.py                (from analysis/)
      python3 19_verify_irs_table.py --dump         (just show the sheet structure)
"""
import sys, re
from pathlib import Path
import numpy as np, pandas as pd

D = Path("data/irs")
OUT = Path("output")
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)

files = sorted(D.glob("*.xls*"))
if not files:
    print("no workbook in analysis/data/irs/ -- see the README there. Nothing to check.")
    sys.exit(0)
print("found:", [f.name for f in files])

TARGET = Path("data/irs/25db-3-01-ex-allyears.xlsx")
src = TARGET if TARGET.exists() else files[0]
print(f"reading {src.name}\n")

xl = pd.ExcelFile(src)
print("sheets:", xl.sheet_names)

for sh in xl.sheet_names:
    raw = pd.read_excel(src, sheet_name=sh, header=None)
    print(f"\n{'='*78}\nSHEET {sh!r}   shape {raw.shape}\n{'='*78}")
    # show enough of the top-left to identify the layout
    with pd.option_context("display.max_columns", 40, "display.max_colwidth", 44):
        print(raw.iloc[:14, :14].to_string())
    if "--dump" in sys.argv:
        continue

    # --- locate the corporation block and the asset-size rows -----------------
    col0 = raw[0].astype(str).str.strip()
    hits = {}
    PATS = {
        "all corporations": r"^total corporation income tax returns|^corporation income tax returns",
        "$250M+":           r"250,?000,?000|\$250 million",
        "$1B-$5B":          r"1,?000,?000,?000 under \$?5|\$1 billion under \$5",
        "$5B-$20B":         r"5,?000,?000,?000 under \$?20|\$5 billion under \$20",
        "$20B+":            r"20,?000,?000,?000 or more|\$20 billion or more",
        "all individuals":  r"^total individual income tax returns|^individual income tax returns",
    }
    for lab, pat in PATS.items():
        m = col0.str.contains(pat, case=False, regex=True, na=False)
        if m.any():
            hits[lab] = list(raw.index[m])
    if not hits:
        print("  (no recognisable row labels in column 0 on this sheet)")
        continue
    print("\n  matched row labels:")
    for lab, idx in hits.items():
        for i in idx:
            print(f"    row {i:>4}  [{lab}]  {col0.iloc[i][:90]}")
    print("\n  raw values on those rows (all columns):")
    for lab, idx in hits.items():
        for i in idx:
            vals = raw.iloc[i].tolist()
            print(f"    [{lab}] " + " | ".join("" if pd.isna(v) else str(v)[:14] for v in vals[:20]))

print("\n" + "=" * 78)
print("NEXT: read the layout above, then tell me and I will write the exact")
print("cell extraction and the row-by-row comparison against t35_irs_benchmark.csv.")
print("=" * 78)

t35 = OUT / "t35_irs_benchmark.csv"
if t35.exists():
    T = pd.read_csv(t35)
    print("\nwhat has to be checked (second-hand values currently in t35):")
    print(T[T.group == "actual (IRS audit)"][["label", "pct", "mature", "note"]].to_string(index=False))
