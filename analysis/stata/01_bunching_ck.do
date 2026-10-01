*==============================================================================
* 01 · Chetty-Kleven polynomial-counterfactual bunching, hand-coded in Stata
*
* WHY THIS FILE EXISTS
* --------------------
* John, 20 August: "A potentially useful Stata command from the ssc is
* 'bunching.' Be sure to read the help file and understand everything that it
* asks for."
*
* Read the help file (02_ssc_bunching_explore.do does that first) and what it
* asks for is: kink(#), tax0(#), tax1(#), m(#). That command -- Bertanha,
* McCallum, Payne & Seegert -- estimates an elasticity at a KINK in a marginal
* tax schedule, with partial identification. Our setting is a NOTCH: crossing
* 25,000 tCO2e does not change a marginal rate, it adds a fixed annual
* compliance cost. There is no tax0 and no tax1 to supply.
*
* So this file hand-codes the estimator the Python actually uses -- the
* polynomial-counterfactual excess-mass estimator of Chetty, Friedman, Olsen &
* Pistaferri (2011), as set out in Kleven's 2016 handbook chapter, the one John
* sent on 15 July. Writing it independently in a second language IS the
* reconciliation: if Stata and Python disagree, one implementation is wrong; if
* they agree and the estimates are still unstable across windows, the sample is
* the problem, not the code.
*
* WHAT IT DOES
*   1. the headline estimate at 25,000 on a clean fit range
*   2. the full window sweep John asked for: excluded region 1,000-10,000 in
*      1,000 increments, symmetric AND asymmetric, at the real cutoff and every
*      placebo that can get a fit range clear of the 10,000 state truncation
*   3. writes stata_bunching_results.csv, to be diffed against
*      output/t26_window_sweep_full.csv
*
* RUN FROM:  analysis/
*   python3 stata/00_export_for_stata.py      // first, to build the .dta
*   do stata/01_bunching_ck.do
*==============================================================================
clear all
set more off
version 14
*--- locate ourselves: every path below is relative to analysis/ ------------
* If Stata's working directory is analysis/stata/ (the usual mistake), step up.
if !fileexists("stata/state_panel.dta") & fileexists("state_panel.dta") cd ..
if !fileexists("stata/state_panel.dta") {
    di as err "state_panel.dta not found."
    di as err "Stata's working directory is: `c(pwd)'"
    di as err "Fix: cd into the analysis/ folder, then run"
    di as err "     python3 stata/00_export_for_stata.py"
    di as err "     do stata/01_bunching_ck.do"
    exit 601
}
capture mkdir "stata/out"

*------------------------------------------------------------------ estimator
capture program drop ckbunch
program define ckbunch, rclass
    * ckbunch varname, CUToff() [BINwidth() FITHW() EXCLo() EXCHi() POLy() REPS()]
    *
    * excluded region: [cutoff-exclo, cutoff)  plus  [cutoff, cutoff+exchi)
    *   exchi(0) gives the ASYMMETRIC (below-only) excluded region
    * excess mass is measured over the below-cutoff excluded bins only, and is
    * normalised by the mean counterfactual density there, so it reads as
    * "extra facility-years per counterfactual facility-year"
    syntax varname(numeric), CUToff(real) [BINwidth(real 1000) FITHW(real 10000) ///
        EXCLo(real 2000) EXCHi(real 0) POLy(int 4) REPS(int 200) SEEd(int 90210)]

    quietly {
        preserve
        local lo = `cutoff' - `fithw'
        local hi = `cutoff' + `fithw'
        keep if inrange(`varlist', `lo', `hi' - 0.001)
        count
        if r(N) < 100 {
            restore
            return scalar b = .
            return scalar lo = .
            return scalar hi = .
            return scalar nfit = r(N)
            exit
        }

        * --- bin, and fill empty bins so the polynomial sees a complete grid
        gen long _bin = floor((`varlist' - `lo') / `binwidth')
        contract _bin, freq(_c)
        local nb = round((`hi' - `lo') / `binwidth')
        tempfile obs
        save `obs', replace
        clear
        set obs `nb'
        gen long _bin = _n - 1
        merge 1:1 _bin using `obs', nogen
        replace _c = 0 if missing(_c)

        gen double _mid = `lo' + (_bin + 0.5) * `binwidth'
        gen double _d   = _mid - `cutoff'
        gen double z    = _d / 1000

        * --- excluded region and the below-cutoff part of it
        gen byte _excl  = (_d >= -`exclo') & (_d < 0)
        if `exchi' > 0 {
            replace _excl = 1 if (_d >= 0) & (_d < `exchi')
        }
        gen byte _below = _excl & (_d < 0)
        count if _below
        if r(N) == 0 {
            restore
            return scalar b = .
            exit
        }

        * --- polynomial in z, plus one dummy per excluded bin
        forvalues k = 1/`poly' {
            gen double z`k' = z^`k'
        }
        levelsof _bin if _excl, local(EB)
        local D
        foreach e of local EB {
            gen byte d`e' = (_bin == `e')
            local D `D' d`e'
        }

        reg _c z1-z`poly' `D'
        gen double _cf = _b[_cons]
        forvalues k = 1/`poly' {
            replace _cf = _cf + _b[z`k'] * z`k'
        }
        predict double _fit, xb
        gen double _res = _c - _fit

        summ _c  if _below, meanonly
        local O = r(sum)
        local nexcl = r(N)
        summ _cf if _below, meanonly
        local C = r(sum)
        local B = `O'/`C' - 1

        * --- residual bootstrap
        set seed `seed'
        local N0 = _N
        gen long   _idx  = .
        gen double _cnew = .
        gen double _cfn  = .
        gen double _bs   = .
        if _N < `reps' set obs `reps'
        forvalues r = 1/`reps' {
            replace _idx  = ceil(runiform() * `N0') in 1/`N0'
            replace _cnew = max(_fit + _res[_idx], 0) in 1/`N0'
            capture reg _cnew z1-z`poly' `D' in 1/`N0'
            if _rc continue
            replace _cfn = _b[_cons] in 1/`N0'
            forvalues k = 1/`poly' {
                replace _cfn = _cfn + _b[z`k'] * z`k' in 1/`N0'
            }
            summ _cnew if _below in 1/`N0', meanonly
            local o = r(sum)
            summ _cfn  if _below in 1/`N0', meanonly
            local c = r(sum)
            if `c' != 0 replace _bs = `o'/`c' - 1 in `r'
        }
        _pctile _bs, p(2.5 97.5)
        local L = r(r1)
        local H = r(r2)
        summ _c, meanonly
        local NFIT = r(sum)
        restore
    }

    return scalar b     = `B'
    return scalar lo    = `L'
    return scalar hi    = `H'
    return scalar sig   = (`L' > 0) | (`H' < 0)
    return scalar nfit  = `NFIT'
    return scalar nexcl = `nexcl'
end

*------------------------------------------------------------------- 1 · headline
use "stata/state_panel.dta", clear

di as txt _n "{hline 78}"
di as txt "1 · headline estimate, CA+WA, fit range [15,000 35,000], 3,000 excluded below"
di as txt "{hline 78}"
ckbunch co2e, cutoff(25000) fithw(10000) exclo(3000) exchi(0) poly(4) reps(300)
di as res "   excess mass = " %6.3f r(b) "   95% CI [" %6.3f r(lo) ", " %6.3f r(hi) "]"
di as res "   bins fitted on " %9.0fc r(nfit) " facility-years"
di as txt "   Python (t20b, same spec) gave +0.767 [0.369, 1.231]. Compare."

*--------------------------------------------------- 2 · the full window sweep
* Two constraints carried over from the Python, both of which matter:
*   (a) the fit range must stay ENTIRELY above 10,000, because the state data
*       are themselves truncated at their own 10,000 reporting threshold
*   (b) 10,000 therefore cannot be a positive control, for the same reason --
*       there is no cutoff in these data where bunching is known to exist
tempname P
postfile `P' str6 sample double cutoff str24 shape double excl_width double fit_lo double fit_hi ///
    double nfit double nexcl double excess double ci_lo double ci_hi double sig ///
    using "stata/out/stata_bunching_results.dta", replace

local FLOOR = 12000
foreach S in CA WA BOTH {
    foreach CUT in 25000 22000 28000 30000 35000 40000 {
        local FH = min(10000, `CUT' - `FLOOR')
        if `FH' < 6000 continue
        forvalues W = 1000(1000)10000 {
            if `W' > `FH' - 2000 continue
            foreach SH in sym asym {
                local EH = cond("`SH'"=="sym", `W', 0)
                local SHN = cond("`SH'"=="sym", "symmetric", "asymmetric (below only)")

                use "stata/state_panel.dta", clear
                if "`S'" == "CA" keep if src == "CA"
                if "`S'" == "WA" keep if src == "WA"

                capture ckbunch co2e, cutoff(`CUT') fithw(`FH') exclo(`W') ///
                    exchi(`EH') poly(4) reps(200)
                if _rc continue
                if missing(r(b)) continue
                local _b     = r(b)
                local _lo    = r(lo)
                local _hi    = r(hi)
                local _sig   = r(sig)
                local _nfit  = r(nfit)
                local _nexcl = r(nexcl)
                post `P' ("`S'") (`CUT') ("`SHN'") (`W') (`CUT'-`FH') (`CUT'+`FH') ///
                    (`_nfit') (`_nexcl') (`_b') (`_lo') (`_hi') (`_sig')
            }
        }
    }
}
postclose `P'

use "stata/out/stata_bunching_results.dta", clear
export delimited using "stata/out/stata_bunching_results.csv", replace

*------------------------------------------------------- 3 · read the sweep
di as txt _n "{hline 78}"
di as txt "2 · share of cells significant, by cutoff  (BOTH = CA+WA pooled)"
di as txt "    If the REAL cutoff is not clearly ahead of the placebos, the"
di as txt "    estimator is not telling us anything about 25,000."
di as txt "{hline 78}"
tabstat sig if sample=="BOTH", by(cutoff) stat(mean count) col(stat)

di as txt _n "3 · excess mass at 25,000 by excluded-region width, asymmetric"
list excl_width excess ci_lo ci_hi sig if sample=="BOTH" & cutoff==25000 ///
    & shape=="asymmetric (below only)", noobs sepby(excl_width) abbrev(12)

di as txt _n "4 · the same, at the 35,000 and 40,000 placebos"
list cutoff excl_width excess ci_lo ci_hi sig if sample=="BOTH" ///
    & inlist(cutoff,35000,40000) & shape=="asymmetric (below only)", noobs abbrev(12)

di as txt _n "{hline 78}"
di as txt "NEXT: diff stata/out/stata_bunching_results.csv against"
di as txt "      output/t26_window_sweep_full.csv  (same grid, same estimator)."
di as txt "  agree + unstable  -> the SAMPLE cannot support this estimator;"
di as txt "                       CA/WA can be written up as a failed attempt"
di as txt "  disagree          -> one implementation is wrong. Find out which."
di as txt "{hline 78}"
