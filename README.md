# The private cost of mandatory environmental disclosure

**Yiling (Elaine) Long** · Northwestern University, MSc Social and Economic Policy — capstone project (work in progress)

What does it cost a firm to be required to disclose? This project studies the US EPA
Greenhouse Gas Reporting Program (GHGRP), whose reporting obligation switches on at
**25,000 tCO₂e** per year and can be switched off under the statutory exit at
**40 CFR 98.2(i)** (five consecutive years below 25,000, or three below 15,000). The
threshold creates a notch: a facility just above the line carries the reporting cost, one
just below it does not. The design asks how firms respond to that notch, and what their
response — or its absence — implies about the private cost of disclosure.

> **Status (September 2026).** Analysis is ongoing and no paper draft is posted here. The
> repository contains the code and the aggregate figures and tables behind the current
> results. Some earlier readings were revised during the project; where a notebook has been
> superseded, the table below says so.

---

## Current results in brief

- **Bunching below the threshold is concentrated where the state rules bind.** In the
  California and Washington panels (which, unlike the federal data, observe facilities below
  25,000), excess mass below the line is significant for excluded regions of 2,000–5,000 t,
  peaks at 5,000 t and disappears by 6,000 t. In California the 25,000 line also triggers
  cap-and-trade obligations, which predict a much wider notch than federal reporting alone.
- **The federal reporting cost is small relative to output.** A fixed-fee notch predicts
  bunching over a width of ΔC/π. With a minimum detectable width of 376 tCO₂e at 80% power,
  the private cost of the reporting obligation is bounded at about **1.5%** of a threshold
  facility's annual emissions, expressed in its own output so that no external price is needed.
- **Exit from reporting is not exit from production.** Linking GHGRP to EIA-860/923, 66% of
  facilities that used the statutory exit were still generating electricity the following
  year, against 9% of facilities that were not eligible to exit.
- **Firm characteristics do not predict behaviour at the line.** Parent assets, revenue,
  employment and parent scale (Compustat, GHGRP parent-company data) are insignificant across
  six specifications.
- **Compliance is not explained by enforcement.** The expected federal penalty for not
  reporting is far below the estimated reporting burden, yet roughly 8,800 facilities report
  each year. The probabilities of detection that firms behave as if they hold line up with
  IRS corporate audit rates rather than with GHGRP's own enforcement record.

---

## Repository layout

```
analysis/
├── 00–04_*.ipynb          core notebooks (setting, identification, manipulation tests,
│                          exit and cost bound, enforcement) — executed
├── 14_update_0903.ipynb   consolidated update after the first round of revisions
├── ghgrp_load.py          shared loader: federal panel + 98.2(i) exit eligibility
├── 06–39_*.py             one script per analysis step (see table below)
├── src/                   plain-text sources of the notebooks (jupytext) and Stata export
├── stata/                 independent Stata implementation of the bunching estimator
├── marx_mle/              Python port of the Marx (2018) dynamic-bunching MLE,
│                          with a simulation recovery test
└── output/                aggregate figures (f*.png) and tables (t*.csv)
```

| scripts | what they do |
|---|---|
| `06`–`08` | density test on state panels; exit by air-permit status; data description |
| `09`–`11` | growth near vs. away from the threshold; origin of the gap; bunching with explicit windows (supersede parts of notebook `02`) |
| `12`, `20`, `26`, `29` | Marx (2018) dynamic bunching: estimator, MLE, power, profile-likelihood intervals |
| `13`, `21` | external validity of the CA/WA samples; reweighting |
| `15`–`17`, `25`, `27`, `28` | diagnostics: truncation, new arrivals, excluded-region sweep, dominated region, the CA 2011–12 anomaly |
| `18`, `19` | perceived detection probability against IRS audit rates (with verification against the source table) |
| `22`–`24` | minimum detectable effects; industry vs. economic sources of the start-point gap; enforcement experience |
| `30`–`35` | EIA-860/923 and EPA power-plant crosswalk: statutory exit vs. physical closure; parent-company links |
| `36`–`39` | QCEW industry aggregates; Compustat-based bounds; firm characteristics at the threshold; matching state panels to GHGRP |

---

## Data

All emissions, enforcement and energy data are public. **Raw files are not included** —
they total several GB. Compustat is licensed (WRDS) and is **not** redistributed; only
aggregate results derived from it appear in `output/`.

| source | role |
|---|---|
| [EPA GHGRP](https://www.epa.gov/ghgreporting/data-sets), per-year workbooks 2010–2023 | main federal panel |
| [California ARB MRR](https://ww2.arb.ca.gov/mrr-data), 2011–2023 | state panel, reporting threshold 10,000 t |
| [Washington Ecology GHG reporting](https://ecology.wa.gov/air-climate/reducing-greenhouse-gas-emissions/tracking-greenhouse-gases/mandatory-greenhouse-gas-reports), 2012–2024 | state panel, reporting threshold 10,000 t |
| [EPA FRS](https://www.epa.gov/frs), [ECHO](https://echo.epa.gov/tools/data-downloads), ICIS FE&C | facility keys; inspection, violation and enforcement history |
| GHGRP Reported Parent Companies | facility → parent links |
| [EIA-860 / EIA-923](https://www.eia.gov/electricity/data/), EPA Power Plant crosswalk | plant status and generation |
| [BLS QCEW](https://www.bls.gov/cew/) | state × industry aggregates |
| Compustat North America (WRDS, licensed) | parent financials |

Place raw files under `analysis/data/` following the layout in `ghgrp_load.py`.

## Reproducing

Python 3 with `pandas`, `numpy`, `scipy`, `statsmodels`, `matplotlib` and
[`rddensity`](https://rdpackages.github.io/rddensity/). Notebooks are generated from
`analysis/src/` with [jupytext](https://jupytext.readthedocs.io); scripts run from
`analysis/`, e.g. `python3 11_bunching_windows.py`.
