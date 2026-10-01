"""
38 · Firm characteristics and behaviour at the threshold        (advisor items 9 and 3, second half)
================================================================================
John's ninth point, and the half of his third point that was never answered:

    "Continue to work on integrating data about firm/sector characteristics (revenue,
     employment, etc.) so we can eventually examine firm behavior around the thresholds."
    "Whether the gap in start points is for industry or economic reasons is an
     empirical question."

working note 09 answered the INDUSTRY half: industry explains 2.0% (NAICS3) to 8.2% (NAICS6)
of the variance in where facilities start, so nine tenths of it is within industries.
The economic half needs firm characteristics, which are now linked (working note 17, `19`).
Nothing had actually been run on them. This file runs it.

TWO TIERS, because coverage differs by an order of magnitude
  Tier 1  parent characteristics built from GHGRP itself -- 99.8% of facility-years.
          The variable that matters is `parent_already_reports`: the parent owns at
          least one ALWAYS-COVERED facility, so it is already inside the GHGRP whatever
          this facility does. For such a parent the MARGINAL disclosure cost of one more
          reporting facility is close to zero -- the forms, the consultant, the internal
          process all exist. 30.4% of threshold-bound facilities are in this position.
          ⭐ If disclosure cost drives behaviour at the line, these facilities should
          avoid it LESS. That is a test of the paper's mechanism, not a control.
  Tier 2  Compustat size and profitability -- public firms only, 49% of emissions.

OUTCOMES (behaviour at the threshold)
  in_band      emissions parked in [20,000, 25,000) -- the avoidance indicator used in
               working note 10
  eligible     reached 98.2(i) eligibility (five years under 25,000 / three under 15,000)
  exit         stopped reporting, conditional on being eligible
  start        first-year log emissions relative to the threshold -- John ③'s start gap

outputs  output/t65_firm_chars_behaviour.csv  output/t66_start_gap_economic.csv
         output/f45_firm_chars.png
run      python3 38_firm_chars_threshold.py            (from analysis/)
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from ghgrp_load import load_cached

OUT = Path("output")
CUT = 25_000

# ---------------------------------------------------------------- 1 · build
de = load_cached()
de = de[de.emissions > 0].copy()
LP = pd.read_csv(OUT / "link_parent.csv")
LP = LP[LP.is_primary][["FacilityId", "year", "parent_name_clean", "pct_ownership"]]
D = de.merge(LP, on=["FacilityId", "year"], how="left")

# --- Tier 1: parent aggregates, from GHGRP only
agg = (D.dropna(subset=["parent_name_clean"])
         .groupby(["parent_name_clean", "year"])
         .agg(parent_n_fac=("FacilityId", "nunique"),
              parent_tot_em=("emissions", "sum"),
              parent_n_ac=("always_covered", "sum"),
              parent_n_states=("state", "nunique")).reset_index())
agg["parent_already_reports"] = (agg.parent_n_ac > 0).astype(int)
D = D.merge(agg, on=["parent_name_clean", "year"], how="left")
# ⚠️ parent_tot_em INCLUDES the facility itself, and the median parent owns exactly one
# facility -- for those, log(parent_tot_em) IS log(own emissions), so regressing a
# facility's own starting level on it is an identity, not a finding. The first version
# of this file did that and reported R2 = 0.31 for "parent scale". Leave-one-out fixes
# it: parent emissions EXCLUDING the focal facility, with a flag for the singletons.
D["parent_em_other"] = D.parent_tot_em - D.emissions
D["solo_facility"] = (D.parent_n_fac <= 1).astype(int)
D["log_parent_em_oth"] = np.log(D.parent_em_other.clip(lower=0) + 1.0)
D["log_parent_n"] = np.log(D.parent_n_fac)

# --- Tier 2: Compustat
CSF = OUT / "link_compustat_financials.csv"
if CSF.exists():
    CS = pd.read_csv(CSF)
    CS = (CS[["parent_name_clean", "year", "conm", "revt", "at", "emp", "oibdp",
              "pi_ebitda", "cs_naics3"]]
          .drop_duplicates(["parent_name_clean", "year"]))
    D = D.merge(CS, on=["parent_name_clean", "year"], how="left")
    D["public"] = D.conm.notna().astype(int)
    for c, nc in [("at", "log_assets"), ("revt", "log_revenue"), ("emp", "log_emp")]:
        D[nc] = np.log(pd.to_numeric(D[c], errors="coerce").where(lambda s: s > 0))
else:
    D["public"] = 0

# --- outcomes
D["in_band"] = ((D.emissions >= 20_000) & (D.emissions < CUT)).astype(int)
D["naics3"] = D.naics3.astype(str)
print(f"panel: {len(D):,} facility-years | with parent {D.parent_name_clean.notna().mean()*100:.1f}% "
      f"| public {D.public.mean()*100:.1f}%")
print(f"threshold-bound with a parent that already reports: "
      f"{D[D.threshold_bound].parent_already_reports.mean()*100:.1f}%")


# ---------------------------------------------------------------- 2 · estimator
def ols(df, y, xs, fe=("year", "naics3"), cluster="parent_name_clean"):
    d = df.dropna(subset=[y] + list(xs) + [cluster]).copy()
    if len(d) < 200:
        return None
    X = d[list(xs)].astype(float)
    for f in fe:
        du = pd.get_dummies(d[f].astype(str), prefix=f, drop_first=True, dtype=float)
        X = pd.concat([X, du], axis=1)
    X = sm.add_constant(X)
    m = sm.OLS(d[y].astype(float), X).fit(
        cov_type="cluster", cov_kwds={"groups": d[cluster].astype(str)})
    return m, d


def row(tag, y, sample, xs, fe=("year", "naics3")):
    r = ols(sample, y, xs, fe)
    if r is None:
        print(f"  {tag:<52s} too few observations"); return []
    m, d = r
    out = []
    for x in xs:
        out.append(dict(outcome=y, spec=tag, var=x, coef=m.params[x], se=m.bse[x],
                        t=m.tvalues[x], p=m.pvalues[x], n=int(m.nobs),
                        n_clusters=d[m.model.data.orig_exog.index.name or "index"].nunique()
                        if False else d.parent_name_clean.nunique(),
                        ymean=float(d[y].mean())))
        print(f"  {tag:<52s} {x:<24s} {m.params[x]:+8.4f} "
              f"(se {m.bse[x]:.4f}, t {m.tvalues[x]:+5.2f})  n={int(m.nobs):,}")
    return out


# ---------------------------------------------------------------- 3 · Tier 1
print("\n" + "=" * 92)
print("3 . TIER 1 -- parent characteristics from GHGRP itself (99.8% coverage)")
print("=" * 92)
res = []
band = D[D.threshold_bound & D.emissions.between(15_000, 40_000)]
print(f"\n[A] parked in [20k, 25k) | threshold-bound, emissions in [15k, 40k], "
      f"n = {len(band):,}, mean {band.in_band.mean():.3f}")
res += row("A1 already-reports only", "in_band", band, ["parent_already_reports"])
res += row("A2 + parent scale", "in_band", band,
           ["parent_already_reports", "log_parent_n", "log_parent_em_oth", "solo_facility"])

elig = D[D.threshold_bound]
print(f"\n[B] reached 98.2(i) eligibility | threshold-bound, n = {len(elig):,}, "
      f"mean {elig.eligible.mean():.3f}")
res += row("B1 already-reports only", "eligible", elig, ["parent_already_reports"])
res += row("B2 + parent scale", "eligible", elig,
           ["parent_already_reports", "log_parent_n", "log_parent_em_oth", "solo_facility"])

ex = D[D.threshold_bound & D.eligible]
print(f"\n[C] stopped reporting | eligible threshold-bound only, n = {len(ex):,}, "
      f"mean {ex.exits.mean():.3f}")
res += row("C1 already-reports only", "exits", ex, ["parent_already_reports"])
res += row("C2 + parent scale", "exits", ex,
           ["parent_already_reports", "log_parent_n", "log_parent_em_oth", "solo_facility"])


# ---------------------------------------------------------------- 4 · Tier 2
print("\n" + "=" * 92)
print("4 . TIER 2 -- Compustat size and profitability (public firms only)")
print("=" * 92)
pub = D[D.public == 1]
print(f"  public-firm facility-years: {len(pub):,} "
      f"({pub.emissions.sum()/D.emissions.sum()*100:.1f}% of emissions)")
bandp = pub[pub.threshold_bound & pub.emissions.between(15_000, 40_000)]
print(f"\n[D] parked in [20k, 25k) | public threshold-bound, n = {len(bandp):,}")
for xs, tag in [(["log_assets"], "D1 assets"),
                (["log_revenue"], "D2 revenue"),
                (["log_emp"], "D3 employment"),
                (["log_assets", "parent_already_reports", "log_parent_n"],
                 "D4 assets + parent controls")]:
    res += row(tag, "in_band", bandp, xs)

eligp = pub[pub.threshold_bound]
print(f"\n[E] reached 98.2(i) eligibility | public threshold-bound, n = {len(eligp):,}")
for xs, tag in [(["log_assets"], "E1 assets"), (["log_revenue"], "E2 revenue")]:
    res += row(tag, "eligible", eligp, xs)

R = pd.DataFrame(res)
R.to_csv(OUT / "t65_firm_chars_behaviour.csv", index=False)


# ---------------------------------------------------------------- 5 · advisor item 3, second half
print("\n" + "=" * 92)
print("5 . JOHN ③, the economic half: does firm size explain WHERE facilities start?")
print("=" * 92)
print("""  working note 09 closed the industry half: NAICS3 explains 2.0% of the variance in start
  position within threshold-bound, NAICS6 8.2%. The target is the 91.8% that is left.
  The test is the INCREMENTAL R-squared of firm characteristics on top of NAICS6.""")

first = (D[D.threshold_bound].sort_values(["FacilityId", "year"])
           .drop_duplicates("FacilityId", keep="first").copy())
first["start"] = np.log(first.emissions / CUT)
first["naics6"] = first.naics.astype(str).str[:6]
print(f"\n  threshold-bound facilities with a first year: {len(first):,}")


def r2(d, xs, fe):
    d = d.dropna(subset=["start"] + list(xs))
    if len(d) < 100:
        return np.nan, 0
    X = pd.DataFrame(index=d.index)
    if xs:
        X = pd.concat([X, d[list(xs)].astype(float)], axis=1)
    for f in fe:
        X = pd.concat([X, pd.get_dummies(d[f].astype(str), prefix=f,
                                         drop_first=True, dtype=float)], axis=1)
    X = sm.add_constant(X)
    return sm.OLS(d["start"].astype(float), X).fit().rsquared, len(d)


rows = []
# the ladder has to be run on ONE constant sample or the R2s are not comparable
base = first.dropna(subset=["start", "log_parent_n", "log_parent_em_oth", "solo_facility"])
ladder = [("nothing (mean only)", [], []),
          ("NAICS3", [], ["naics3"]),
          ("NAICS6", [], ["naics6"]),
          ("parent scale (leave-one-out)", ["log_parent_n", "log_parent_em_oth", "solo_facility"], []),
          ("NAICS6 + parent scale (LOO)", ["log_parent_n", "log_parent_em_oth", "solo_facility"], ["naics6"])]
print(f"\n  ladder A -- all threshold-bound facilities with parent data (n = {len(base):,}):")
for lab, xs, fe in ladder:
    v, n = r2(base, xs, fe)
    rows.append(dict(sample="all TB with parent", spec=lab, r2=round(v, 4), n=n))
    print(f"    {lab:<28s} R2 = {v:.4f}")

basep = first.dropna(subset=["start", "log_assets", "log_parent_n", "log_parent_em_oth", "solo_facility"])
ladder2 = [("nothing (mean only)", [], []),
           ("NAICS6", [], ["naics6"]),
           ("firm size only (assets)", ["log_assets"], []),
           ("NAICS6 + assets", ["log_assets"], ["naics6"]),
           ("NAICS6 + assets + parent scale (LOO)",
            ["log_assets", "log_parent_n", "log_parent_em_oth", "solo_facility"], ["naics6"])]
print(f"\n  ladder B -- public firms only, constant sample (n = {len(basep):,}):")
for lab, xs, fe in ladder2:
    v, n = r2(basep, xs, fe)
    rows.append(dict(sample="public TB", spec=lab, r2=round(v, 4), n=n))
    print(f"    {lab:<32s} R2 = {v:.4f}")
S5 = pd.DataFrame(rows)
S5.to_csv(OUT / "t66_start_gap_economic.csv", index=False)


# ---------------------------------------------------------------- 6 · reconcile
print("\n" + "=" * 92)
print("6 . reconciling with working note 09 (2.0% / 8.2%) -- different sample, not a conflict")
print("=" * 92)
print("""  working note 09 reported a BETWEEN-GROUP VARIANCE SHARE on the near-threshold WINDOW
  sample; the ladder above is an R-squared of industry dummies on the FIRST YEAR of
  every threshold-bound facility. Same idea, different sample. Checking that the
  window restriction reproduces the smaller number:""")
for lab, sub in [("all first years (ladder A above)", base),
                 ("first years, start in [15k, 40k]",
                  base[base.emissions.between(15_000, 40_000)]),
                 ("first years, start in [20k, 30k]",
                  base[base.emissions.between(20_000, 30_000)])]:
    v3, n3 = r2(sub, [], ["naics3"])
    v6, _ = r2(sub, [], ["naics6"])
    print(f"    {lab:<36s} n={n3:>5,}  NAICS3 R2 = {v3:.3f}   NAICS6 R2 = {v6:.3f}")
print("""
  => the window restriction moves the numbers around (0.037-0.060 for NAICS3) but does
     NOT reproduce 09's 2.0% by itself, so the gap is not only the window: 09 decomposes
     the variance across ALL facility-years in the near-threshold window, this ladder is
     an R-squared on ONE observation per facility (its first year). Different unit of
     observation, different estimand.
     ⚠️ Do not present 2.0% and 5.5% as the same number measured twice. Quote 09's for
     the window-panel decomposition, these for the first-year population, and say which
     is which. Both support the same conclusion -- industry explains little -- and
     neither is wrong.""")

# ---------------------------------------------------------------- 7 · figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.9))

key = R[(R.outcome == "in_band") & (R["var"].isin(
    ["parent_already_reports", "log_assets", "log_revenue", "log_emp", "log_parent_n"]))]
key = key.drop_duplicates(["spec", "var"]).tail(9).iloc[::-1]
yy = np.arange(len(key))
lo, hi = key.coef - 1.96 * key.se, key.coef + 1.96 * key.se
ax[0].hlines(yy, lo, hi, lw=3.4, color="#4C72B0", alpha=.75)
ax[0].plot(key.coef, yy, "o", ms=7, color="k", zorder=3)
ax[0].axvline(0, color="k", lw=1)
ax[0].set_yticks(yy)
ax[0].set_yticklabels([f"{v[:22]}\n[{s[:3]}]" for v, s in zip(key["var"], key.spec)],
                      fontsize=7)
ax[0].set_xlabel("effect on P(parked in 20-25k band)")
ax[0].set_title("(a) NOTHING about the firm predicts parking\n"
                "below the line (threshold-bound, 15-40k)", fontsize=10)

lad = S5[S5["sample"] == "public TB"]
names = ["nothing", "NAICS6", "assets only", "NAICS6\n+assets", "NAICS6+assets\n+parent"]
ax[1].bar(np.arange(len(lad)), lad.r2.values, .6,
          color=["#dddddd", "#4C72B0", "#C44E52", "#8172B2", "#55A868"])
for i, v in enumerate(lad.r2.values):
    ax[1].text(i, v + .008, f"{v:.3f}", ha="center", fontsize=9)
ax[1].set_xticks(np.arange(len(lad)))
ax[1].set_xticklabels(names[:len(lad)], fontsize=7.5)
ax[1].set_ylabel("R² for the facility's starting position")
ax[1].set_title("(b) John ③, the economic half: firm size adds\n"
                "0.4 pp on top of industry", fontsize=10)

la = S5[S5["sample"] == "all TB with parent"]
ax[2].bar(np.arange(len(la)), la.r2.values, .6,
          color=["#dddddd", "#bbbbbb", "#4C72B0", "#C44E52", "#55A868"])
for i, v in enumerate(la.r2.values):
    ax[2].text(i, v + .008, f"{v:.3f}", ha="center", fontsize=9)
ax[2].set_xticks(np.arange(len(la)))
ax[2].set_xticklabels(["nothing", "NAICS3", "NAICS6", "parent\nscale", "NAICS6+\nparent"],
                      fontsize=7.5)
ax[2].set_ylabel("R²")
ax[2].set_title("(c) full sample: industry ~20%, parent scale +3 pp,\n"
                "about 70% explained by neither", fontsize=10)

for a_ in ax:
    a_.spines[["top", "right"]].set_visible(False)
fig.suptitle("f45 - firm characteristics and behaviour at the threshold (John ⑨ / ③)",
             fontsize=12, y=1.03)
fig.tight_layout()
fig.savefig(OUT / "f45_firm_chars.png", dpi=150, bbox_inches="tight")
print(f"\nwrote t65_firm_chars_behaviour.csv, t66_start_gap_economic.csv, f45_firm_chars.png")
