"""Export the GHGRP analysis panel to Stata (.dta) plus a descriptive statistics table.

    python export_stata.py

Writes to output/:
    ghgrp_panel.dta          the analysis dataset, variable-labelled
    descriptives.do          Stata do-file reproducing the table
    descriptives.xlsx        the table, plus a variable dictionary
    descriptives.md          the same table in markdown
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from ghgrp_load import load_ghgrp, add_eligibility

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- variables
# (stata_name, source_column, label, group)
SPEC = [
    # --- identifiers ---
    ("facility_id",   "FacilityId",    "EPA GHGRP facility identifier",                     "Identifiers"),
    ("year",          "year",          "Reporting year",                                    "Identifiers"),
    ("frs_id",        "FRSId",         "EPA Facility Registry Service identifier",          "Identifiers"),

    # --- emissions ---
    ("emissions",     "emissions",     "Total reported direct emissions (mt CO2e)",         "Emissions"),
    ("log_emissions", "log_em",        "Log total reported direct emissions",               "Emissions"),
    ("co2",           "CO2 emissions (non-biogenic)",        "CO2, non-biogenic (mt)",      "Emissions"),
    ("ch4",           "Methane (CH4) emissions",             "Methane (mt CO2e)",           "Emissions"),
    ("n2o",           "Nitrous Oxide (N2O) emissions",       "Nitrous oxide (mt CO2e)",     "Emissions"),
    ("co2_biogenic",  "Biogenic CO2 emissions (metric tons)","Biogenic CO2 (mt)",           "Emissions"),

    # --- regulatory status ---
    ("threshold_bound",  "threshold_bound",  "=1 if reports only by exceeding 25,000 tCO2e",   "Regulatory status"),
    ("always_covered",   "always_covered",   "=1 if source category reports regardless of level","Regulatory status"),
    ("always_cov_strict","always_covered_strict","=1 if subpart D/F/G/H (strict definition)",   "Regulatory status"),
    ("n_subparts",       "n_subparts",       "Number of subparts the facility reports under",   "Regulatory status"),
    ("cems",             "cems",             "=1 if facility uses continuous emissions monitoring","Regulatory status"),

    # --- 40 CFR 98.2(i) eligibility ---
    ("eligible",      "eligible",      "=1 if eligible to cease reporting under 98.2(i)",   "Cessation eligibility"),
    ("elig_5yr_25k",  "elig_5yr_25k",  "=1 if 5 consecutive years below 25,000 tCO2e",      "Cessation eligibility"),
    ("elig_3yr_15k",  "elig_3yr_15k",  "=1 if 3 consecutive years below 15,000 tCO2e",      "Cessation eligibility"),

    # --- panel structure ---
    ("exits",         "exits",         "=1 if last observed year and not right-censored",   "Panel structure"),
    ("enters",        "enters",        "=1 if first observed year and not left-censored",   "Panel structure"),
    ("gap_after",     "gap_after",     "=1 if facility is absent the following year",       "Panel structure"),
    ("first_year",    "first_year",    "First year the facility appears",                   "Panel structure"),
    ("last_year",     "last_year",     "Last year the facility appears",                    "Panel structure"),
    ("n_years",       "n_years",       "Number of years the facility is observed",          "Panel structure"),

    # --- geography and industry ---
    ("naics",         "naics",         "Primary NAICS code (6-digit)",                      "Geography and industry"),
    ("latitude",      "lat",           "Facility latitude",                                 "Geography and industry"),
    ("longitude",     "lon",           "Facility longitude",                                "Geography and industry"),

    # --- string variables (described, not summarised) ---
    ("facility_name", "facility_name", "Facility name",                                     "String variables"),
    ("state",         "state",         "State (two-letter)",                                "String variables"),
    ("city",          "city",          "City",                                              "String variables"),
    ("county",        "county",        "County",                                            "String variables"),
    ("naics3",        "naics3",        "NAICS 3-digit sector",                              "String variables"),
    ("subparts",      "subparts",      "Subparts reported under, comma-separated",          "String variables"),
    ("sectors",       "sectors",       "EPA sector label",                                  "String variables"),
]

STRING_VARS = {s for s, _, _, g in SPEC if g == "String variables"}


def build():
    de = add_eligibility(load_ghgrp(verbose=False), verbose=False)

    out = pd.DataFrame(index=de.index)
    labels, groups = {}, {}
    for stata_name, src, label, group in SPEC:
        if src not in de.columns:
            print(f"  ! missing source column {src!r}, skipped")
            continue
        s = de[src]
        if s.dtype == bool:
            s = s.astype("int8")
        elif str(s.dtype) == "Int64":
            s = pd.to_numeric(s, errors="coerce").astype("float64")
        elif s.dtype == object and stata_name in STRING_VARS:
            s = s.astype(str).str.slice(0, 200)
        out[stata_name] = s
        labels[stata_name] = label[:80]          # Stata caps variable labels at 80 chars
        groups[stata_name] = group
    return de, out, labels, groups


def summary_table(out, labels, groups):
    rows = []
    for v in out.columns:
        g = groups[v]
        if g == "String variables":
            rows.append(dict(Group=g, Variable=v, Label=labels[v],
                             N=int(out[v].notna().sum()), Unique=int(out[v].nunique()),
                             Mean=np.nan, SD=np.nan, Min=np.nan, Max=np.nan))
        else:
            s = pd.to_numeric(out[v], errors="coerce")
            rows.append(dict(Group=g, Variable=v, Label=labels[v],
                             N=int(s.notna().sum()), Unique=int(s.nunique()),
                             Mean=s.mean(), SD=s.std(), Min=s.min(), Max=s.max()))
    return pd.DataFrame(rows)


def fmt(x, v):
    if pd.isna(x):
        return ""
    if v in ("latitude", "longitude"):
        return f"{x:,.3f}"
    if abs(x) >= 1000 or v in ("naics", "frs_id", "facility_id"):
        return f"{x:,.0f}"
    if abs(x) < 10 and x != int(x):
        return f"{x:,.3f}"
    return f"{x:,.2f}"


def main():
    de, out, labels, groups = build()

    # ---- .dta ----
    dta = OUT / "ghgrp_panel.dta"
    out.to_stata(dta, write_index=False, version=118, variable_labels=labels,
                 data_label="EPA GHGRP facility panel, 2010-2023")
    print(f"wrote {dta}  ({len(out):,} obs x {len(out.columns)} vars, "
          f"{dta.stat().st_size/1e6:.1f} MB)")

    # ---- table ----
    tab = summary_table(out, labels, groups)
    disp = tab.copy()
    for c in ["Mean", "SD", "Min", "Max"]:
        disp[c] = [fmt(x, v) for x, v in zip(tab[c], tab.Variable)]
    disp["N"] = tab.N.map("{:,}".format)
    disp["Unique"] = tab.Unique.map("{:,}".format)

    # ---- data notes: things the table surfaces that need explaining ----
    em = out.emissions
    notes = pd.DataFrame({
        "Item": [
            "Negative emissions",
            "Zero emissions",
            "Log emissions missing",
            "Facilities below 1,000 tCO2e",
            "State count = 54",
            "Identifier moments",
            "NAICS moments",
            "Gas-specific variables",
            "Eligibility construction",
            "Exit definition",
        ],
        "Detail": [
            f"{int((em < 0).sum())} facility-year (facility 1002560, 2015, -1,196 tCO2e; "
            "subparts C and II, chemical manufacturing). Net-of-offset reporting; retained as filed.",
            f"{int((em == 0).sum())} facility-years across {int(out.loc[em == 0, 'facility_id'].nunique())} "
            "facilities, mostly subparts C and D — units idled but still under a reporting obligation.",
            f"{int(out.log_emissions.isna().sum())} observations = the zeros plus the one negative value.",
            f"{int((em < 1000).sum()):,} facility-years ({(em < 1000).mean():.2%}). Incumbents well "
            "below the threshold that have not yet exercised the 98.2(i) off-ramp.",
            "50 states plus DC, Guam, Puerto Rico and the US Virgin Islands.",
            "facility_id and frs_id are identifiers; their mean, SD, min and max carry no meaning "
            "and are reported only for completeness.",
            "naics is a categorical code; its moments are likewise uninterpretable. Use naics3 "
            "for sector groupings.",
            "co2, ch4, n2o and co2_biogenic are missing where the facility reports no emissions "
            "of that gas, not where the value is zero. Non-missing shares differ accordingly.",
            "eligible = 5 consecutive years below 25,000 OR 3 consecutive years below 15,000, "
            "computed on each facility's own reporting sequence (40 CFR 98.2(i)).",
            "exits = 1 in a facility's last observed year, provided that year is not the final "
            "year of the panel. Inferred from panel disappearance, not verified against EPA "
            "cessation filings.",
        ],
    })

    with pd.ExcelWriter(OUT / "descriptives.xlsx") as w:
        disp.to_excel(w, "Descriptive statistics", index=False)
        pd.DataFrame({
            "Item": ["Unit of observation", "Facility-years", "Facilities", "Years",
                     "Source", "Coverage", "Notes"],
            "Value": ["Facility x year", f"{len(out):,}", f"{out.facility_id.nunique():,}",
                      f"{int(out.year.min())}-{int(out.year.max())}",
                      "EPA GHGRP annual facility files (Direct Emitters sheet)",
                      "Facilities reporting >=25,000 tCO2e, plus source categories "
                      "covered regardless of level",
                      "Binary variables coded 0/1. Emissions in metric tons CO2 equivalent. "
                      "Eligibility follows 40 CFR 98.2(i)."],
        }).to_excel(w, "Dataset overview", index=False)
        notes.to_excel(w, "Data notes", index=False)
    print(f"wrote {OUT/'descriptives.xlsx'}")

    # ---- markdown ----
    lines = ["# GHGRP analysis panel — descriptive statistics", "",
             f"Facility-year observations: **{len(out):,}** | "
             f"Facilities: **{out.facility_id.nunique():,}** | "
             f"Years: **{int(out.year.min())}–{int(out.year.max())}**", "",
             "Source: EPA GHGRP annual facility files (Direct Emitters). "
             "Emissions in metric tons CO2 equivalent. Binary variables coded 0/1.", ""]
    for g in dict.fromkeys(groups.values()):
        sub = disp[disp.Group == g]
        if sub.empty:
            continue
        lines += [f"## {g}", "",
                  "| Variable | Label | N | Mean | SD | Min | Max |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for r in sub.itertuples():
            lines.append(f"| `{r.Variable}` | {r.Label} | {r.N} | {r.Mean} | {r.SD} | "
                         f"{r.Min} | {r.Max} |")
        if g == "String variables":
            lines.append("")
            lines.append("*String variables — unique values rather than moments: " +
                         ", ".join(f"`{r.Variable}` ({r.Unique})" for r in sub.itertuples()) + ".*")
        lines.append("")

    lines += ["## Data notes", ""]
    for r in notes.itertuples():
        lines.append(f"- **{r.Item}.** {r.Detail}")
    lines.append("")
    (OUT / "descriptives.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT/'descriptives.md'}")

    return tab


if __name__ == "__main__":
    main()
