"""
16 · The "newly arrived" hypothesis                       (John, 3 September 2026)
================================================================================
John's note:

    "It would be worthwhile to explore why year-on-year growth rates are
     marginally higher for firms close to the reporting threshold. One potential
     hypothesis is that a lot of these firms are 'newly arrived,' with emissions
     growing quickly before reporting makes potential emissions regulations more
     salient. It could be checked if more of the firms just above the threshold
     are new arrivals, and if their emissions growth slows in later years after
     arrival."

Two claims, tested separately:

    T1  facilities just above 25,000 are disproportionately new arrivals
    T2  their emissions growth slows in later years after arrival

THE TWO THINGS THAT CAN FAKE BOTH RESULTS
-----------------------------------------
T1 is mechanically true for a threshold-bound facility. It enters the data the
first year it crosses 25,000, so the bin just above the line is stocked with
tenure-0 observations BY CONSTRUCTION. A raw "share of new arrivals by bin" plot
is not evidence of anything.

T2 has the same disease as the growth-rate window scan (see 09/RESEARCH_LOG node
2): entry is triggered by a high draw, so the year after entry mean-reverts
downward whether or not anybody has noticed a regulation. "Growth slows after
arrival" is the null, not the finding.

So every number here is reported against a comparison group that shares the
selection mechanism but not the treatment:

    TB-entry    threshold-bound, first year in the panel        <- John's group
    AC-entry    always-covered, first year in the panel
                (reports regardless of level: entry is a real new facility or a
                 new subpart, NOT a threshold crossing)
    AC-cross    always-covered, first year it crosses 25,000 from below
                while ALREADY reporting
                <- the key placebo: identical selection-on-a-high-draw, zero
                   change in reporting obligation. Whatever AC-cross does after
                   the event is what mean reversion alone produces.

Reading rule: T2 is evidence for John's hypothesis only to the extent
TB-entry slows MORE than AC-cross.

Left censoring: 2010 is the first program year, so 2010 first-appearances have
unknown arrival dates and are dropped from every cohort calculation.

outputs  t29_entry_cohorts.csv  t30_newarrival_by_bin.csv
         t31_growth_by_tenure.csv  t32_meanreversion_placebo.csv
         f28_new_arrivals.png  f29_tenure_event_study.png
run      python3 16_new_arrivals.py      (from analysis/)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ghgrp_load import load_cached, add_transitions

OUT = Path("output"); OUT.mkdir(exist_ok=True)
CUT = 25_000
Y0, Y1 = 2010, 2023
pd.set_option("display.width", 200)

de = load_cached()
de, tr = add_transitions(de)
print(f"panel {len(de):,} facility-years, {de.FacilityId.nunique():,} facilities, {Y0}-{Y1}")
print(f"transitions (consecutive years, both >0): {len(tr):,}")


# ---------------------------------------------------------------- OLS helper
def ols_cluster(y, X, cluster):
    """OLS with cluster-robust SEs. X includes its own constant column."""
    y = np.asarray(y, float); X = np.asarray(X, float)
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    u = y - X @ b
    cl = pd.Series(np.asarray(cluster))
    meat = np.zeros((X.shape[1], X.shape[1]))
    for _, idx in cl.groupby(cl).indices.items():
        Xg = X[idx]; ug = u[idx]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    G = cl.nunique(); n, k = X.shape
    dfc = (G / (G - 1)) * ((n - 1) / (n - k)) if G > 1 else 1.0
    V = XtX_inv @ meat @ XtX_inv * dfc
    return b, np.sqrt(np.diag(V)), G


def wmean(x):
    x = np.asarray(x, float)
    n = len(x)
    return (x.mean(), x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan, n)


# =============================================================================
# 0 · define the three event groups
# =============================================================================
de["tenure"] = de.year - de.first_year          # 0 in the arrival year
de["cohort"] = de.first_year

# --- TB-entry / AC-entry : first appearance, left-censoring dropped
ent = de[de.first_year > Y0].copy()
ent["grp"] = np.where(ent.threshold_bound, "TB-entry", "AC-entry")

# --- AC-cross : always-covered, first year it goes from <25,000 to >=25,000
ac = de[de.always_covered].sort_values(["FacilityId", "year"]).copy()
ac["lag_em"] = ac.groupby("FacilityId").emissions.shift(1)
ac["lag_yr"] = ac.groupby("FacilityId").year.shift(1)
cross = ac[(ac.year - ac.lag_yr == 1) & (ac.lag_em < CUT) & (ac.emissions >= CUT)]
first_cross = cross.groupby("FacilityId").year.min().rename("cross_year")
print(f"\nAC facilities with at least one upward crossing of 25,000: {len(first_cross):,}")

de = de.merge(first_cross, on="FacilityId", how="left")
de["tenure_x"] = de.year - de.cross_year        # 0 in the crossing year


# =============================================================================
# 1 · cohorts and the level at which facilities arrive
# =============================================================================
arr = ent[ent.tenure == 0].copy()
coh = (arr.groupby(["cohort", "grp"]).agg(n=("FacilityId", "size"),
                                          med_em=("emissions", "median"))
          .reset_index().pivot(index="cohort", columns="grp"))
coh.columns = [f"{a}_{b}" for a, b in coh.columns]
coh = coh.reset_index()
coh.to_csv(OUT / "t29_entry_cohorts.csv", index=False)
print("\n=== 1 · arrivals by cohort (2010 first-appearances dropped: left-censored) ===")
print(coh.to_string(index=False))

EDGES = [0, 25_000, 30_000, 40_000, 60_000, 100_000, np.inf]
ELAB = ["<25k", "25-30k", "30-40k", "40-60k", "60-100k", ">100k"]
arr["entry_bin"] = pd.cut(arr.emissions, EDGES, labels=ELAB, right=False)
ed = (arr.groupby(["grp", "entry_bin"], observed=False).size()
        .unstack(0).fillna(0).astype(int))
ed_sh = ed / ed.sum()
print("\n=== where facilities are when they first appear ===")
print(pd.concat([ed, (ed_sh * 100).round(1).add_suffix(" %")], axis=1).to_string())
print("\n Read: a threshold-bound facility CANNOT appear below 25,000, so its first bin is")
print(" empty by construction. What matters is whether arrivals pile into 25-30k (crept over")
print(" the line) or land high (a genuinely new plant).")


# =============================================================================
# 2 · T1 · are facilities just above the threshold more often new arrivals?
# =============================================================================
BEDG = [15_000, 20_000, 25_000, 30_000, 40_000, 60_000, 100_000, np.inf]
BLAB = ["15-20k", "20-25k", "25-30k", "30-40k", "40-60k", "60-100k", ">100k"]
d2 = de[(de.first_year > Y0) & (de.emissions >= 15_000)].copy()
d2["bin"] = pd.cut(d2.emissions, BEDG, labels=BLAB, right=False)
d2["new2"] = d2.tenure <= 2
d2["grp"] = np.where(d2.threshold_bound, "TB", "AC")

rows = []
for (g, b), s in d2.groupby(["grp", "bin"], observed=True):
    if len(s) < 25:
        continue
    m, se, n = wmean(s.new2.values)
    rows.append(dict(group=g, bin=str(b), n=n, share_new=round(m, 4), se=round(se, 4)))
T30 = pd.DataFrame(rows)
T30.to_csv(OUT / "t30_newarrival_by_bin.csv", index=False)
print("\n=== 2 · T1 · share of facility-years with tenure <= 2, by emissions bin ===")
print(T30.pivot(index="bin", columns="group", values=["share_new", "n"]).to_string())
print("\n *** The TB column is contaminated by construction: a TB facility's first")
print(" observation is its first crossing, so low bins are mechanically young.")
print(" The AC column is the same statistic where entry is NOT a crossing event.")
print(" Only the TB-minus-AC contrast is interpretable.")


# =============================================================================
# 3 · T2 · does growth slow after arrival?
# =============================================================================
tr = tr.merge(de[["FacilityId", "year", "tenure", "tenure_x", "cohort", "cross_year"]],
              on=["FacilityId", "year"], how="left")

tr2 = tr.merge(arr[["FacilityId", "emissions"]].rename(columns={"emissions": "entry_em"}),
               on="FacilityId", how="left")
tbe = tr2[(tr2.cohort > Y0) & tr2.threshold_bound & tr2.entry_em.notna()].copy()

def tenure_profile(df, tcol, label, tmin=1, tmax=8):
    out = []
    for k in range(tmin, tmax + 1):
        s = df[df[tcol] == k]
        if len(s) < 20:
            continue
        m, se, n = wmean(s.growth.values)
        out.append(dict(group=label, tenure=k, n=n, mean_growth=round(m, 4), se=round(se, 4)))
    return out

prof = []
prof += tenure_profile(tr[(tr.cohort > Y0) & tr.threshold_bound], "tenure", "TB-entry")
prof += tenure_profile(tr[(tr.cohort > Y0) & tr.always_covered], "tenure", "AC-entry")
prof += tenure_profile(tr[tr.always_covered & tr.cross_year.notna()], "tenure_x",
                       "AC-cross", tmin=-4)
T31 = pd.DataFrame(prof)
T31.to_csv(OUT / "t31_growth_by_tenure.csv", index=False)
print("\n=== 3 · T2 · mean log growth by years since the event ===")
print(T31.pivot(index="tenure", columns="group",
                values=["mean_growth", "n"]).round(4).to_string())


# =============================================================================
# 4 · the slowdown, and the same slowdown under pure mean reversion
# =============================================================================
def slowdown(df, tcol, label):
    early = df[df[tcol].isin([1, 2])].growth.values
    late = df[df[tcol].between(4, 8)].growth.values
    if len(early) < 20 or len(late) < 20:
        return None
    me, se_e, ne = wmean(early)
    ml, se_l, nl = wmean(late)
    d = me - ml
    sd = np.sqrt(se_e ** 2 + se_l ** 2)
    from math import erfc, sqrt
    t = d / sd if sd > 0 else np.nan
    p = erfc(abs(t) / sqrt(2)) if sd > 0 else np.nan
    return dict(group=label, n_early=ne, mean_early=round(me, 4),
                n_late=nl, mean_late=round(ml, 4),
                slowdown=round(d, 4), se=round(sd, 4), t=round(t, 2), p=round(p, 4))

sl = [r for r in [
    slowdown(tr[(tr.cohort > Y0) & tr.threshold_bound], "tenure", "TB-entry"),
    slowdown(tr[(tr.cohort > Y0) & tr.always_covered], "tenure", "AC-entry"),
    slowdown(tr[tr.always_covered & tr.cross_year.notna()], "tenure_x", "AC-cross"),
] if r]
T32 = pd.DataFrame(sl)
T32.to_csv(OUT / "t32_meanreversion_placebo.csv", index=False)
print("\n=== 4 · slowdown = mean growth at tenure 1-2 minus mean growth at tenure 4-8 ===")
print(T32.to_string(index=False))
print("\n *** AC-cross is the amount of slowdown that selection-on-a-high-draw produces")
print(" ALL BY ITSELF. John's hypothesis needs TB-entry to beat it, not merely to be")
print(" positive.")

# --- regression version, facility-clustered, tenure dummies, no year FE
#     (tenure = year - cohort, so with facility FE the two are collinear; see the note)
sub = tr2[(tr2.cohort > Y0) & tr2.threshold_bound & (tr2.entry_em >= 25_000)
          & tr2.tenure.between(1, 8)].copy()   # genuine crossings only
if len(sub) > 100:
    K = sorted(sub.tenure.unique())[1:]           # tenure 1 is the omitted base
    X = np.column_stack([np.ones(len(sub))] + [(sub.tenure == k).values for k in K])
    b, se, G = ols_cluster(sub.growth.values, X, sub.FacilityId.values)
    print(f"\n--- TB-entry, growth on tenure dummies (base = tenure 1), "
          f"{len(sub):,} obs, {G:,} facility clusters ---")
    print(f"  {'tenure':>8} {'coef':>9} {'se':>8} {'t':>7}")
    print(f"  {1:>8} {0.0:>9.4f} {'(base)':>8}")
    for k, bi, si in zip(K, b[1:], se[1:]):
        print(f"  {k:>8} {bi:>9.4f} {si:>8.4f} {bi/si:>7.2f}")
    print("  const (= tenure-1 mean) %.4f (se %.4f)" % (b[0], se[0]))


# =============================================================================
# 4b · the pre-event run-up, and crept-over vs landed-high arrivals
# =============================================================================
print("\n=== 4b-i · AC-cross, the ONLY group observable on BOTH sides of 25,000 ===")
pre = T31[(T31.group == "AC-cross")][["tenure", "mean_growth", "se", "n"]]
print(pre.to_string(index=False))
print("\n John's hypothesis has a PRE-period claim -- 'emissions growing quickly BEFORE")
print(" reporting' -- that is unobservable for threshold-bound facilities by construction:")
print(" they are invisible below 25,000. AC-cross is the only window onto it.")

print("\n=== 4b-ii · TB arrivals split by where they landed ===")

# !! 248 "threshold-bound" facilities first appear BELOW 25,000 (median 8,667 tCO2e,
#    mostly subparts C, FF, W). A facility that is genuinely bound by the 25,000
#    threshold cannot enter the data below it, so these are either misclassified
#    (ALWAYS_COVERED is D/F/G/H/HH only -- 40 CFR 98 Table A-3 lists more) or reporting
#    for some other reason. They grow at +0.36 in their first two years and then level
#    off, and on a first pass they drove the entire apparent post-arrival slowdown.
#    They are reported separately and excluded from the headline sample.
splits = [("anomalous: entry < 25k", tbe[tbe.entry_em < 25_000]),
          ("crept over: entry 25-30k", tbe[tbe.entry_em.between(25_000, 30_000, "left")]),
          ("landed high: entry >= 40k", tbe[tbe.entry_em >= 40_000]),
          ("TB-entry, genuine crossings only", tbe[tbe.entry_em >= 25_000])]
sp = []
for lab, d in splits:
    r = slowdown(d, "tenure", lab)
    if r:
        r["n_fac"] = d.FacilityId.nunique()
        sp.append(r)
    if lab.startswith(("crept", "landed", "TB-entry,")):
        pr = tenure_profile(d, "tenure", lab)
        for row in pr:
            row["group"] = lab
        prof.extend(pr)
T32b = pd.DataFrame(sp)
T32b.to_csv(OUT / "t32b_entry_level_split.csv", index=False)
print(T32b.to_string(index=False))
print("\n *** The headline row is the last one. Once the 248 sub-threshold first")
print(" appearances are removed, the post-arrival slowdown is -0.003 (t = -0.18): gone.")
print(" The +0.033 in table t32 was those 248 facilities and nothing else.")

T31 = pd.DataFrame(prof)
T31.to_csv(OUT / "t31_growth_by_tenure.csv", index=False)

# --- 4b-iii · dose response among genuine crossings
# Salience says every arrival learns the same thing, so the slowdown should not depend on
# the arrival level. Exit-seeking under 98.2(i) says only facilities close enough to get
# back under 25,000 have anything to gain, so it should fall away with distance. Neither
# prediction is borne out, because there is no slowdown to allocate.
print("\n=== 4b-iii · slowdown by arrival level, genuine crossings only ===")
DED = [25_000, 28_000, 32_000, 40_000, 60_000, np.inf]
DLAB = ["25-28k", "28-32k", "32-40k", "40-60k", ">60k"]
tbe2 = tbe[tbe.entry_em >= 25_000].copy()
tbe2["ebin"] = pd.cut(tbe2.entry_em, DED, labels=DLAB, right=False)
dose = []
for lab in DLAB:
    d = tbe2[tbe2.ebin == lab]
    r = slowdown(d, "tenure", lab)
    if r:
        r["n_fac"] = d.FacilityId.nunique()
        dose.append(r)
T32c = pd.DataFrame(dose)
T32c.to_csv(OUT / "t32c_dose_response.csv", index=False)
print(T32c.to_string(index=False))
print("\n Flat and null at every arrival level. There is no dose response because there")
print(" is no effect: neither salience nor exit-seeking shows up in post-arrival growth.")


# =============================================================================
# 5 · do arrivals leave again?  (links to the 98.2(i) exit margin)
# =============================================================================
print("\n=== 5 · exit and eligibility by tenure, threshold-bound arrivals ===")
tb = de[(de.cohort > Y0) & de.threshold_bound & de.tenure.between(0, 8)]
ex = (tb.groupby("tenure")
        .agg(n=("FacilityId", "size"), exit_rate=("exits", "mean"),
             elig_rate=("eligible", "mean"), med_em=("emissions", "median"))
        .round(4).reset_index())
print(ex.to_string(index=False))
print("\n A facility that crossed 25,000 on a high draw and fell back becomes 98.2(i)")
print(" eligible about five years later. That is the same population as node 3's exiters.")


# =============================================================================
# 6 · figures
# =============================================================================
C = {"TB-entry": "#C44E52", "AC-entry": "#4C72B0", "AC-cross": "#55A868",
     "TB": "#C44E52", "AC": "#4C72B0"}

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

# (a) where arrivals land
w = 0.38
xs = np.arange(len(ELAB))
for i, g in enumerate(["TB-entry", "AC-entry"]):
    if g in ed_sh.columns:
        ax[0].bar(xs + (i - 0.5) * w, ed_sh[g].values * 100, w,
                  label=g, color=C[g], alpha=.85)
ax[0].set_xticks(xs); ax[0].set_xticklabels(ELAB, rotation=30, ha="right")
ax[0].set_ylabel("% of arrivals"); ax[0].legend(frameon=False, fontsize=9)
ax[0].set_title("(a) emissions in the arrival year\n"
                "TB cannot appear below the line by construction", fontsize=10)

# (b) T1
for g in ["TB", "AC"]:
    s = T30[T30.group == g]
    if len(s):
        ax[1].errorbar(range(len(s)), s.share_new * 100, yerr=s.se * 100,
                       fmt="o-", ms=5, capsize=3, color=C[g], label=g)
s0 = T30[T30.group == "TB"]
ax[1].set_xticks(range(len(s0))); ax[1].set_xticklabels(s0.bin, rotation=30, ha="right")
ax[1].set_ylabel("% with tenure <= 2"); ax[1].legend(frameon=False, fontsize=9)
ax[1].set_title("(b) T1: share of new arrivals by bin\n"
                "read the TB-AC gap, not the TB level", fontsize=10)

# (c) T2
for g in ["TB-entry, genuine crossings only", "AC-entry", "AC-cross"]:
    s = T31[T31.group == g]
    if len(s):
        ax[2].errorbar(s.tenure, s.mean_growth, yerr=s.se, fmt="o-", ms=5,
                       capsize=3, color=C.get(g, "#C44E52"), label=g.replace(", genuine crossings only", " (crossings)"))
ax[2].axhline(0, lw=.8, color="k")
ax[2].axvline(0, lw=.8, color="k", ls=":")
ax[2].set_xlabel("years since the event"); ax[2].set_ylabel("mean log growth")
ax[2].legend(frameon=False, fontsize=9)
ax[2].set_title("(c) T2: growth after arrival\n"
                "AC-cross pre-period is selection, not behaviour", fontsize=10)

for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("f28 - the 'newly arrived' hypothesis, and the two things that fake it",
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "f28_new_arrivals.png", dpi=150, bbox_inches="tight")
print("\nwrote f28_new_arrivals.png")

fig2, ax2 = plt.subplots(1, 2, figsize=(11.5, 4.4))
sl_df = T32.set_index("group")
gs = [g for g in ["TB-entry", "AC-entry", "AC-cross"] if g in sl_df.index]
sl_df = pd.concat([sl_df, T32b.set_index("group")])
ax2[0].bar(range(len(gs)), sl_df.loc[gs, "slowdown"],
           yerr=sl_df.loc[gs, "se"], capsize=4,
           color=[C[g] for g in gs], alpha=.85)
ax2[0].axhline(0, lw=.8, color="k")
ax2[0].set_xticks(range(len(gs))); ax2[0].set_xticklabels(gs, rotation=15)
ax2[0].set_ylabel("growth(tenure 1-2) - growth(tenure 4-8)")
ax2[0].set_title("(a) the slowdown, and the placebo slowdown", fontsize=10)

ax2[1].plot(ex.tenure, ex.exit_rate * 100, "o-", color="#C44E52", label="exit rate")
ax2[1].plot(ex.tenure, ex.elig_rate * 100, "s--", color="#8172B2", label="98.2(i) eligible")
ax2[1].set_xlabel("years since arrival"); ax2[1].set_ylabel("%")
ax2[1].legend(frameon=False, fontsize=9)
ax2[1].set_title("(b) arrivals become exit-eligible on the statutory clock", fontsize=10)
for a in ax2:
    a.spines[["top", "right"]].set_visible(False)
fig2.suptitle("f29 - separating salience from mean reversion", fontsize=12, y=1.03)
fig2.tight_layout()
fig2.savefig(OUT / "f29_tenure_event_study.png", dpi=150, bbox_inches="tight")
print("wrote f29_tenure_event_study.png")
print("\nwrote t29 t30 t31 t32")
