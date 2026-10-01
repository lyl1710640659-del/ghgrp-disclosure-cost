*=============================================================================
*  GHGRP analysis panel — variables and descriptive statistics
*  Elaine · July 2026
*
*  Dataset : output/ghgrp_panel.dta
*            94,378 facility-years · 8,778 facilities · 2010-2023
*            Unit of observation: facility x year.
*            Source: EPA GHGRP annual facility files (public download).
*
*  The variables fall into two groups, reported separately below:
*    A. AS REPORTED TO EPA  — taken from the annual files unchanged.
*    B. CONSTRUCTED BY ME   — my own definitions. These carry the analysis,
*                             so the construction rule is stated for each.
*
*  Run from the analysis/ folder:   do descriptives_basic.do
*=============================================================================

clear all
set more off
version 17

capture log close
log using "output/descriptives_basic.log", replace text

use "output/ghgrp_panel.dta", clear

* make the wide identifiers readable rather than scientific notation
format frs_id %14.0f
format facility_id %9.0f


*=============================================================================
* 1.  What is in the dataset
*=============================================================================

describe


*=============================================================================
* 2.  PANEL A — Variables as reported to EPA
*=============================================================================
* Nothing here is my construction. Emissions are in metric tons of CO2
* equivalent, as filed by the facility.

summarize facility_id frs_id year ///
          emissions log_emissions ///
          co2 ch4 n2o co2_biogenic ///
          naics latitude longitude ///
          , separator(0)

* String identifiers — unique values rather than moments
foreach v of varlist facility_name state city county naics3 subparts sectors {
    quietly levelsof `v', local(lv)
    display as text %-16s "`v'" as result %8.0fc `: word count `lv'' ///
            as text "  unique values"
}


*=============================================================================
* 3.  PANEL B — Variables I constructed
*=============================================================================
* These are my definitions, not EPA's. They are what the analysis rests on,
* so each construction rule is given below and each is checked afterwards.
*
*   threshold_bound    facility does NOT report under subparts D, F, G, H, HH
*   always_covered     facility DOES report under one of those subparts, which
*                      must report regardless of emissions level, so neither
*                      the 25,000 nor the 15,000 threshold binds it
*   always_cov_strict  same, D/F/G/H only (drops landfills, whose permanent
*                      coverage I have not yet verified against Table A-3)
*   n_subparts         count of subparts listed for the facility that year
*   cems               EPA's continuous-emissions-monitoring flag, recoded Y/N to 1/0
*
*   elig_5yr_25k       5 consecutive years below 25,000 tCO2e
*   elig_3yr_15k       3 consecutive years below 15,000 tCO2e
*   eligible           either of the above  -> may cease reporting under
*                      40 CFR 98.2(i), having notified EPA
*
*   exits              =1 in a facility's LAST observed year, provided that
*                      year is not the final year of the panel
*   enters             =1 in a facility's FIRST observed year, provided that
*                      year is not the first year of the panel
*   gap_after          facility absent the following year but present later
*   first_year / last_year / n_years   panel span for the facility

summarize threshold_bound always_covered always_cov_strict n_subparts cems ///
          elig_5yr_25k elig_3yr_15k eligible ///
          exits enters gap_after first_year last_year n_years ///
          , separator(0)


*=============================================================================
* 4.  Checks on the constructed variables
*=============================================================================
* These are the ones worth interrogating, so here is what they imply.

* -- 4a. Do the two eligibility routes overlap, and how often does each bind?
tabulate elig_5yr_25k elig_3yr_15k, cell

* -- 4b. The comparison the design rests on
tabulate eligible exits, row

* -- 4c. Is the always-covered group actually always covered?
*        If the definition is right, its emissions distribution should extend
*        well below 25,000 without the facilities leaving.
tabstat emissions, by(always_covered) ///
        statistics(n mean sd p10 p50 p90 min max) columns(statistics) nototal

* -- 4d. Panel balance: exits is only meaningful if facilities really do leave
tabulate year exits, row nofreq


*=============================================================================
* 5.  Data anomalies — what the min and max columns surface
*=============================================================================

* -- one negative emissions value
list facility_id year emissions subparts state if emissions < 0, noobs

* -- facilities reporting exactly zero
count if emissions == 0
display "zero-emission facility-years: " r(N)
quietly levelsof facility_id if emissions == 0, local(z)
display "distinct facilities involved: " `: word count `z''

* -- these two together are why log_emissions has fewer observations
count if missing(log_emissions)
display "missing log_emissions: " r(N) "  (= the zeros plus the one negative)"

* -- gas variables are missing, not zero, when a gas is not reported
misstable summarize co2 ch4 n2o co2_biogenic

log close


*=============================================================================
*  Notes for the reader
*-----------------------------------------------------------------------------
*  - facility_id, frs_id and naics are identifiers or categorical codes. Their
*    means and standard deviations are reported for completeness only.
*
*  - "exits" is inferred from a facility disappearing from the panel. EPA also
*    collects a Notification to Discontinue Reporting; I have not yet
*    reconciled the two, and that is the first item on my list.
*
*  - "exits" cannot distinguish a facility that stops reporting from one that
*    shuts down. Separating them needs establishment-level survival data,
*    which this public release does not contain. This is the largest open gap.
*
*  - The subpart letters behind always_covered come from EPA's industry-type
*    table, not from a line-by-line reading of 40 CFR 98 Table A-3.
*    always_cov_strict exists so that results can be checked against the
*    narrower definition.
*=============================================================================
