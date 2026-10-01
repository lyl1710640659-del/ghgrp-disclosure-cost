"""
20 · Marx (2018) section-6 MLE on our panels.

The estimator is implemented in marx_mle/marx_mle.py, translated line by line from
the author's R file. marx_mle/test_recovery.py plants a known bunching share in
simulated data and gets it back (0.250 planted -> 0.256 / 0.239 recovered, attrition
0.050 -> 0.048), which is what licenses the numbers below.

WHICH SAMPLES THE MLE CAN AND CANNOT TAKE
  The model needs the bottom of the observed support (rmin) to sit STRICTLY BELOW the
  notch. Otherwise the growth that reaches the notch and the growth that reaches the
  censoring floor are the same number, and the bunching point mass is not separable
  from the attrition point mass.

    CA + WA                    floor 10,000, notch 25,000   -> runs
    federal always-covered     no floor at all              -> runs (placebo)
    federal threshold-bound    floor IS 25,000              -> CANNOT RUN

  That last line is node 4 again, in a new estimator. The MLE does not rescue the
  federal threshold-bound sample; it does give the state panels a second, quite
  different reading, and it gives the placebo group a real test.

outputs  output/t36_marx_mle.csv
run      python3 20_marx_mle_run.py        (from analysis/; takes several minutes)
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, ".")
sys.path.insert(0, "marx_mle")
from ghgrp_load import load_cached
from marx_mle import Spec, fit_staged

OUT = Path("output")
NOTCH = np.log(25_000)
STATE_FLOOR = np.log(10_000)

ca = pd.read_csv(OUT / "ca_mrr_panel.csv")
wa = pd.read_csv(OUT / "wa_ghg_panel.csv"); wa = wa[wa["Reporter Type"] == "Facility"]
for d in (ca, wa):
    d["co2e"] = pd.to_numeric(d.co2e, errors="coerce")


def pairs(df, idcol, y0=None, y1=None):
    """Consecutive-year pairs. A facility that is present in t and ABSENT in t+1 is
    kept with a missing outcome -- that is an attritor, and the MLE uses it."""
    x = df.dropna(subset=["co2e"]).sort_values([idcol, "year"]).copy()
    x["lead"] = x.groupby(idcol).co2e.shift(-1)
    x["ly"] = x.groupby(idcol).year.shift(-1)
    x.loc[x.ly != x.year + 1, "lead"] = np.nan          # gap or exit -> attrition
    x = x[x.co2e > 0]
    if y0 is not None:
        x = x[x.year >= y0]
    if y1 is not None:
        x = x[x.year <= y1 - 1]
    return x.co2e.values, x.lead.values


de = load_cached()
d = de.sort_values(["FacilityId", "year"]).copy()
d["lead_em"] = d.groupby("FacilityId").emissions.shift(-1)
d["lead_year"] = d.groupby("FacilityId").year.shift(-1)
d.loc[d.lead_year != d.year + 1, "lead_em"] = np.nan
ac = d[d.always_covered & (d.emissions > 0)]

ac_s = ac.sample(n=min(len(ac), 10_500), random_state=11)

ca_l, ca_n = pairs(ca, "arb_id")
wa_l, wa_n = pairs(wa, "Reporter")
both_l = np.concatenate([ca_l, wa_l]); both_n = np.concatenate([ca_n, wa_n])
wac_l, wac_n = pairs(wa, "Reporter", None, 2023)        # WA 2012-2022, reporting only

SAMPLES = [
    ("CA + WA, notch 25,000",        both_l, both_n, NOTCH,          STATE_FLOOR),
    ("CA only, notch 25,000",        ca_l,   ca_n,   NOTCH,          STATE_FLOOR),
    ("WA 2012-2022, notch 25,000",   wac_l,  wac_n,  NOTCH,          STATE_FLOOR),
    # subsampled to the size of CA+WA so it is a like-for-like placebo (and so it
    # finishes inside the shell's three-minute cap)
    ("federal always-covered (placebo)",
     ac_s.emissions.values, ac_s.lead_em.values,      NOTCH,          STATE_FLOOR),
    # placebo cutoff inside the state panels: nothing happens at 40,000
    ("CA + WA, PLACEBO notch 40,000", both_l, both_n, np.log(40_000), STATE_FLOOR),
]

# Each device shell call is capped at three minutes and background jobs do not
# survive it, so samples are run one at a time and appended:
#     python3 20_marx_mle_run.py 0 1      runs SAMPLES[0] and SAMPLES[1]
PICK = [int(a) for a in sys.argv[1:]] or list(range(len(SAMPLES)))
CSV = OUT / "t36_marx_mle.csv"
rows = []
for idx in PICK:
    name, lt, l1, notch, rmin = SAMPLES[idx]
    sp = Spec(notch, rmin, full=False)
    t0 = time.time()
    try:
        o = fit_staged(lt, l1, sp, maxiter=2000, restarts=1, seed=11)
    except Exception as e:                                   # noqa: BLE001
        print(f"{name}: FAILED -- {e}")
        continue
    at_notch = float((np.abs(o["gt"] - o["tonotch"]) < 1e-9).mean())
    censored_share = float((o["gt"] <= o["lb"] + 1e-12).mean())
    rows.append(dict(sample=name, notch=int(round(np.exp(notch))),
                     rmin=int(round(np.exp(rmin))), n=o["n"],
                     bunch_below=round(o["bunch_below"], 4),
                     bunch_above=round(o["bunch_above"], 4),
                     excess=round(o["excess"], 5),
                     attrition=round(o["attrition_at_rmin"], 4),
                     miss_instead_of_crossing=round(o["miss_instead_of_crossing"], 4),
                     LR_vs_no_bunching=round(o["lr_vs_no_bunching"], 1),
                     nll=round(o["nll"], 1), raw_share_at_notch=round(at_notch, 4),
                     raw_share_censored=round(censored_share, 4),
                     secs=int(time.time() - t0)))
    print(f"  done {name}  ({time.time()-t0:.0f}s)", flush=True)

T = pd.DataFrame(rows)
if CSV.exists():
    old = pd.read_csv(CSV)
    T = pd.concat([old[~old["sample"].isin(T["sample"])], T], ignore_index=True)
T.to_csv(CSV, index=False)
pd.set_option("display.width", 250)
print("\n=== Marx (2018) section-6 MLE ===")
print(T.to_string(index=False))
print("\n LR_vs_no_bunching is 2*(logL with bunching free - logL with it pinned at zero).")
print(" One restricted parameter, so compare against chi2(1): 3.84 at 5%, 6.63 at 1%.")
print(" Read the placebo rows FIRST. Neither the always-covered group nor the 40,000")
print(" cutoff has anything to bunch at, so whatever they report is the floor of what")
print(" this estimator will produce from nothing.")
