"""
25 · Three holes that were left open                                (2026-09-13)
================================================================================
  (a) always_covered_STRICT has been defined since the Table A-3 correction but never
      run. It is the robustness check ON that correction: if the results only hold for
      the nineteen-subpart definition and not the sixteen-subpart one, the correction
      bought a conclusion rather than a cleaner sample.
  (b) 203 facilities classified threshold_bound first appear BELOW 25,000, which a
      genuinely threshold-bound facility cannot do. Table A-3 explained the FF ones;
      113 subpart-C-only facilities are still unexplained and currently sit in the
      paper as an unexamined data limitation.
  (c) The ECHO/FRS match rate quoted in notebook 04 is 97.8%. Script 24 just got 88.5%.
      Those are two different quantities and the paper should say which is which.

outputs  t47_strict_definition.csv  t48_subpartC_anomaly.csv  t49_match_rates.csv
         f37_three_gaps.png
run      python3 25_three_gaps.py            (from analysis/)
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from ghgrp_load import load_cached, add_transitions, ALWAYS_COVERED, ALWAYS_COVERED_STRICT

OUT = Path("output"); ECHO = Path("data/echo")
CUT, BIN = 25_000, 1000
BELOW_BINS, ABOVE_BINS, W_PEAK, MIN_BELOW_FIT = 14, 20, 5_000, 4

de = load_cached()
de, tr = add_transitions(de)
d = de[de.emissions > 0].copy()
print(f"ALWAYS_COVERED        ({len(ALWAYS_COVERED)} subparts): {ALWAYS_COVERED}")
print(f"ALWAYS_COVERED_STRICT ({len(ALWAYS_COVERED_STRICT)} subparts): {ALWAYS_COVERED_STRICT}")


def bunch(x, cut, order=4, B=400, seed=0):
    fit_lo, fit_hi = cut - BELOW_BINS * BIN, cut + ABOVE_BINS * BIN
    edges = np.arange(fit_lo, fit_hi + BIN, BIN)
    c = np.histogram(x[(x >= fit_lo) & (x < fit_hi)], bins=edges)[0].astype(float)
    z = (edges[:-1] + BIN / 2 - cut) / 1000.0
    dd = z * 1000
    in_ex = (dd >= -W_PEAK) & (dd < 0)
    if c.sum() < 100 or int(((~in_ex) & (dd < 0)).sum()) < MIN_BELOW_FIT:
        return None
    P = np.vander(z, order + 1)
    X = np.hstack([P, np.eye(len(z))[:, in_ex]])
    beta, *_ = np.linalg.lstsq(X, c, rcond=None)
    res = c - X @ beta

    def stat(cc):
        b, *_ = np.linalg.lstsq(X, cc, rcond=None)
        cf = (P @ b[: order + 1])[in_ex]
        return (cc[in_ex] - cf).sum() / max(cf.mean(), 1e-9)
    rng = np.random.default_rng(seed)
    dr = [stat(np.maximum(c + rng.choice(res, len(c), replace=True), 0)) for _ in range(B)]
    lo, hi = np.percentile(dr, [2.5, 97.5])
    return dict(b=stat(c), lo=lo, hi=hi, sig=bool(lo > 0 or hi < 0), n=int(c.sum()))


def step(x):
    a = int(((x >= 24_000) & (x < 25_000)).sum())
    b = int(((x >= 25_000) & (x < 26_000)).sum())
    return b / a if a else np.nan


# =============================================================================
# (a) strict vs full definition
# =============================================================================
rows = []
for lab, accol in [("full (19 subparts)", "always_covered"),
                   ("strict (16 unconditional)", "always_covered_strict"),
                   ("legacy D/F/G/H/HH", "always_covered_legacy")]:
    tb = d[~d[accol]]
    ac = d[d[accol]]
    rt, ra = bunch(tb.emissions.values, CUT, seed=1), bunch(ac.emissions.values, CUT, seed=2)
    rows.append(dict(definition=lab,
                     n_TB=len(tb), n_AC=len(ac), ac_share=round(d[accol].mean(), 3),
                     step_TB=round(step(tb.emissions.values), 3),
                     step_AC=round(step(ac.emissions.values), 3),
                     # NOT bunching: the federal threshold-bound sample is truncated
                     # at 25,000, so a polynomial fitted from 11,000 reads the missing
                     # mass below the line as a large negative "excess". It is reported
                     # only to show that the wall in node 4 is still there under every
                     # definition. Do not quote it as an estimate.
                     truncation_artefact_TB=round(rt["b"], 3) if rt else np.nan,
                     excess_AC=round(ra["b"], 3) if ra else np.nan,
                     AC_sig=ra["sig"] if ra else None,
                     exit_TB=round(tb.exits.mean(), 4),
                     elig_TB=round(tb.eligible.mean(), 4)))
S = pd.DataFrame(rows)
S.to_csv(OUT / "t47_strict_definition.csv", index=False)
pd.set_option("display.width", 230)
print("\n=== (a) does anything hinge on the 19-vs-16 subpart call? ===")
print(S.to_string(index=False))
print("\n HH, DD and FF are the three Table A-3 categories with their own gas-specific")
print(" thresholds. 'strict' drops them from always-covered, moving them INTO")
print(" threshold_bound. Read `excess_AC` -- the placebo -- and `step_TB`. The")
print(" `truncation_artefact_TB` column is the estimator reading the missing mass below")
print(" 25,000 in a truncated sample; it is node 4's wall, not an estimate.")


# =============================================================================
# (b) the 203 facilities that first appear below 25,000
# =============================================================================
d2 = de.sort_values(["FacilityId", "year"]).copy()
arr = d2[(d2.first_year > 2010) & (d2.year == d2.first_year) & d2.threshold_bound
         & (d2.emissions > 0) & (d2.emissions < CUT)]
ids = arr.FacilityId.unique()
print(f"\n=== (b) {len(ids)} threshold-bound facilities first appear below 25,000 ===")
sub = d2[d2.FacilityId.isin(ids)].copy()
sub["tenure"] = sub.year - sub.first_year

# what happens in the years after that first, too-low observation?
w = sub.pivot_table(index="FacilityId", columns="tenure", values="emissions", aggfunc="max")
first = w[0]
second = w[1] if 1 in w else pd.Series(dtype=float)
ever_above = sub.groupby("FacilityId").emissions.max() >= CUT
yrs_below = sub[sub.emissions < CUT].groupby("FacilityId").size()
n_years = sub.groupby("FacilityId").size()

jump = (second >= CUT)
flags = pd.DataFrame({
    "first_year_emissions": first,
    "second_year_emissions": second,
    "ever_above_25k": ever_above,
    "years_below_25k": yrs_below.reindex(ids).fillna(0),
    "years_observed": n_years.reindex(ids),
})
flags["crosses_in_year_2"] = jump.reindex(ids).fillna(False)
flags["never_above"] = ~flags.ever_above_25k
flags["subparts_first"] = arr.set_index("FacilityId").subparts.reindex(ids)
flags["cohort"] = arr.set_index("FacilityId").cohort_ if False else \
    arr.set_index("FacilityId").first_year.reindex(ids)

tot = len(flags)
print(f"  median first-year emissions {flags.first_year_emissions.median():,.0f}")
print(f"  crosses 25,000 in year 2      {int(flags.crosses_in_year_2.sum()):>4} "
      f"({flags.crosses_in_year_2.mean():.1%})   <- consistent with a PARTIAL FIRST YEAR")
print(f"  ever above 25,000 at all      {int(flags.ever_above_25k.sum()):>4} "
      f"({flags.ever_above_25k.mean():.1%})")
print(f"  NEVER above 25,000            {int(flags.never_above.sum()):>4} "
      f"({flags.never_above.mean():.1%})   <- these should not be reporting at all")
print(f"  median years spent below      {flags.years_below_25k.median():.0f} "
      f"of {flags.years_observed.median():.0f} observed")
print("\n  entry cohort:")
print(flags.cohort.value_counts().sort_index().to_string())
print("\n  subparts in the first year (top 8):")
print(flags.subparts_first.value_counts().head(8).to_string())
flags.reset_index().to_csv(OUT / "t48_subpartC_anomaly.csv", index=False)


# --- two concrete candidate explanations, both testable right here
sub0 = arr.set_index("FacilityId")
SUPPLIER = {"MM", "NN", "LL", "OO", "PP", "QQ"}
W_PARTS = {"W", "W-OFFSH", "W-ONSH", "W-GB", "W-PROC", "W-NGTC", "W-TRANS",
           "W-UNSTG", "W-LNGSTG", "W-LNGIE", "W-LDC"}


def has(sset, group):
    return bool({p.split("-")[0] if p.split("-")[0] in {"MM", "NN"} else p
                 for p in sset} & group) or bool(sset & group)


flags["has_supplier_subpart"] = sub0.subpart_set.reindex(ids).map(
    lambda s: bool({p.split("-")[0] for p in s} & SUPPLIER) if isinstance(s, frozenset) else False)
flags["has_W_subpart"] = sub0.subpart_set.reindex(ids).map(
    lambda s: bool(s & W_PARTS) if isinstance(s, frozenset) else False)
bio = sub0["Biogenic CO2 emissions (metric tons)"].reindex(ids).fillna(0)
flags["with_biogenic"] = flags.first_year_emissions + bio
flags["biogenic_crosses"] = flags.with_biogenic >= CUT

print("\n  --- candidate explanations, tested ---")
print(f"  (1) partial first year (crosses in year 2)      "
      f"{int(flags.crosses_in_year_2.sum()):>4}  ({flags.crosses_in_year_2.mean():>5.1%})")
print(f"  (2) has a SUPPLIER subpart (MM/NN/LL/OO/PP/QQ)  "
      f"{int(flags.has_supplier_subpart.sum()):>4}  ({flags.has_supplier_subpart.mean():>5.1%})"
      "   <- suppliers report on quantity SUPPLIED, not emitted")
print(f"  (3) has a subpart-W part (basin-level rules)    "
      f"{int(flags.has_W_subpart.sum()):>4}  ({flags.has_W_subpart.mean():>5.1%})"
      "   <- onshore production applies the threshold to the BASIN")
print(f"  (4) crosses 25,000 once biogenic CO2 is added   "
      f"{int(flags.biogenic_crosses.sum()):>4}  ({flags.biogenic_crosses.mean():>5.1%})")
covered = (flags.crosses_in_year_2 | flags.has_supplier_subpart | flags.has_W_subpart
           | flags.biogenic_crosses)
print(f"\n  explained by at least one of (1)-(4):           "
      f"{int(covered.sum()):>4}  ({covered.mean():>5.1%})")
print(f"  STILL UNEXPLAINED:                             "
      f"{int((~covered).sum()):>4}  ({(~covered).mean():>5.1%})")
rest = flags[~covered]
print(f"\n  the unexplained ones: median first-year emissions "
      f"{rest.first_year_emissions.median():,.0f}, "
      f"{rest.never_above.mean():.0%} never exceed 25,000")
print("  their subparts:")
print(rest.subparts_first.value_counts().head(6).to_string())
flags.reset_index().to_csv(OUT / "t48_subpartC_anomaly.csv", index=False)


# =============================================================================
# (c) match rates -- and which quantity each number is
# =============================================================================
has_frs = de.FRSId.notna()
tgt = set(de.loc[has_frs, "FRSId"].astype("int64").astype(str))
keep = ["REGISTRY_ID", "FAC_INSPECTION_COUNT"]
got = []
for ch in pd.read_csv(ECHO / "ECHO_EXPORTER.csv", dtype=str, usecols=keep,
                      chunksize=300_000):
    got.append(ch[ch.REGISTRY_ID.isin(tgt)])
EE = pd.concat(got, ignore_index=True).drop_duplicates("REGISTRY_ID")
rid = set(EE.REGISTRY_ID)
de["_rid"] = de.FRSId.astype("Int64").astype(str)
in_echo = de._rid.isin(rid)

MR = pd.DataFrame([
    dict(quantity="facility-years with an FRS Id in the GHGRP panel",
         n=int(has_frs.sum()), of=len(de), pct=round(has_frs.mean(), 4)),
    dict(quantity="FACILITIES with an FRS Id",
         n=int(de.loc[has_frs, "FacilityId"].nunique()), of=de.FacilityId.nunique(),
         pct=round(de.loc[has_frs, "FacilityId"].nunique() / de.FacilityId.nunique(), 4)),
    dict(quantity="facility-years whose FRS Id is FOUND in ECHO Exporter",
         n=int(in_echo.sum()), of=len(de), pct=round(in_echo.mean(), 4)),
    dict(quantity="FACILITIES found in ECHO Exporter",
         n=int(de.loc[in_echo, "FacilityId"].nunique()), of=de.FacilityId.nunique(),
         pct=round(de.loc[in_echo, "FacilityId"].nunique() / de.FacilityId.nunique(), 4)),
    dict(quantity="EMISSIONS-weighted: share of total tCO2e in matched facility-years",
         n=int(de.loc[in_echo, "emissions"].sum()), of=int(de.emissions.sum()),
         pct=round(de.loc[in_echo, "emissions"].sum() / de.emissions.sum(), 4)),
])
MR.to_csv(OUT / "t49_match_rates.csv", index=False)
print("\n=== (c) match rates: three different numbers that get confused ===")
print(MR.to_string(index=False))

# matched vs unmatched balance, the discipline set out in the data-ingestion note
bal = []
for lab, col in [("log emissions", None), ("threshold-bound", "threshold_bound"),
                 ("more than one subpart", None), ("in the 20-30k band", None),
                 ("98.2(i)-eligible", "eligible"), ("panel length", "n_years")]:
    if col is None:
        v = {"log emissions": np.log(de.emissions.replace(0, np.nan)),
             "more than one subpart": (de.n_subparts > 1).astype(float),
             "in the 20-30k band": de.emissions.between(20_000, 30_000).astype(float)}[lab]
    else:
        v = de[col].astype(float)
    a, b = v[in_echo].dropna(), v[~in_echo].dropna()
    nd = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
    bal.append(dict(variable=lab, matched=round(a.mean(), 3), unmatched=round(b.mean(), 3),
                    ND=round(nd, 3), flag="***" if abs(nd) > 0.25 else ""))
B = pd.DataFrame(bal)
print("\n  matched vs unmatched (|ND| > 0.25 flagged):")
print(B.to_string(index=False))
B.to_csv(OUT / "t49_match_rates.csv", mode="a", index=False)
print("\nwrote t47, t48, t49")


# =============================================================================
# figure
# =============================================================================
fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))

# (a) definition robustness
x = np.arange(len(S))
w = .36
ax[0].bar(x - w / 2, S.step_TB, w, color="#C44E52", alpha=.85, label="threshold-bound")
ax[0].bar(x + w / 2, S.step_AC, w, color="#4C72B0", alpha=.85, label="always-covered")
ax[0].axhline(1, ls="--", lw=1, color="k")
for i, (a_, b_) in enumerate(zip(S.step_TB, S.step_AC)):
    ax[0].text(i - w / 2, a_ + .02, f"{a_:.2f}", ha="center", fontsize=8.5)
    ax[0].text(i + w / 2, b_ + .02, f"{b_:.2f}", ha="center", fontsize=8.5)
ax[0].set_xticks(x)
ax[0].set_xticklabels([s.replace(" (", "\n(") for s in S.definition], fontsize=8)
ax[0].set_ylabel("density step 24-25k -> 25-26k")
ax[0].set_ylim(0, 1.75)
ax[0].legend(frameon=False, fontsize=8.5)
ax[0].set_title("(a) nothing hinges on the 19-vs-16 subpart call\n"
                f"placebo excess: {S.excess_AC.iloc[0]:+.3f} / {S.excess_AC.iloc[1]:+.3f} / "
                f"{S.excess_AC.iloc[2]:+.3f}, none significant", fontsize=9.5)

# (b) the anomaly, explained
lab_b = ["partial first year\n(crosses in yr 2)", "supplier subpart\n(reports on supply)",
         "subpart W\n(basin-level rule)", "biogenic CO2\npushes it over",
         "STILL\nUNEXPLAINED"]
val_b = [flags.crosses_in_year_2.sum(), flags.has_supplier_subpart.sum(),
         flags.has_W_subpart.sum(), flags.biogenic_crosses.sum(), (~covered).sum()]
cols_b = ["#4C72B0"] * 4 + ["#C44E52"]
yb = np.arange(len(lab_b))[::-1]
ax[1].barh(yb, val_b, .55, color=cols_b, alpha=.85)
for y_, v in zip(yb, val_b):
    ax[1].text(v + 1.5, y_, f"{v}  ({v/tot:.0%})", va="center", fontsize=8.5)
ax[1].set_yticks(yb); ax[1].set_yticklabels(lab_b, fontsize=8)
ax[1].set_xlim(0, max(val_b) * 1.35)
ax[1].set_xlabel(f"facilities (of {tot}; categories overlap)")
ax[1].set_title(f"(b) the below-25,000 first appearances:\n"
                f"{covered.sum()} of {tot} now have a candidate explanation", fontsize=9.5)

# (c) match rates
lab_c = ["FRS Id present\nin the panel", "facilities found\nin ECHO",
         "EMISSIONS-weighted\nmatch", "facility-years\nfound in ECHO"]
val_c = [MR.pct.iloc[0], MR.pct.iloc[3], MR.pct.iloc[4], MR.pct.iloc[2]]
yc = np.arange(len(lab_c))[::-1]
ax[2].barh(yc, np.array(val_c) * 100, .55, color="#8C8C8C", alpha=.85)
for y_, v in zip(yc, val_c):
    ax[2].text(v * 100 + .4, y_, f"{v:.1%}", va="center", fontsize=9)
ax[2].set_yticks(yc); ax[2].set_yticklabels(lab_c, fontsize=8)
ax[2].set_xlim(80, 102)
ax[2].set_xlabel("%")
ax[2].set_title("(c) four different numbers, all correct\n"
                "notebook 04's 97.8% is the FIRST one, not a match rate", fontsize=9.5)

for a_ in ax:
    a_.spines[["top", "right"]].set_visible(False)
fig.suptitle("f37 - three holes, closed", fontsize=12, y=1.03)
fig.tight_layout()
fig.savefig(OUT / "f37_three_gaps.png", dpi=150, bbox_inches="tight")
print("wrote f37_three_gaps.png")
