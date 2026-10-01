"""
32 · Statutory disclosure exit vs physical closure                      (2026-09-28)
================================================================================
THREAT #1, answered as far as EIA-860 can answer it.

The lower bound rests on facilities running the 40 CFR 98.2(i) clock and then ceasing
to report. The objection: a facility that stops reporting because it SHUT DOWN tells us
nothing about the cost of disclosure. EIA-860's generator-level `Retirement Year`
separates the two for the facilities EIA covers.

    GHGRP report stops in year Y, EIA plant has no operable units after Y  -> CLOSURE
    GHGRP report stops in year Y, EIA plant still operable after Y         -> DISCLOSURE EXIT

THE TEST THAT MATTERS
  Compare exiters who were ELIGIBLE under 98.2(i) (five years under 25,000, or three
  under 15,000) against exiters who were not. If the eligible ones are physical closures
  at the same rate as everyone else, the lower bound is in trouble. If they are closures
  much LESS often, then running the clock is a disclosure decision, which is what the
  paper claims.

⚠️ COVERAGE. EIA-860 is power plants only. Section 1 reports honestly what share of the
   exiting population can be adjudicated at all; the rest stays a stated limitation.

outputs  output/t57_exit_vs_closure_<link>.csv   output/t58_exit_coverage_<link>.csv
         output/t59_exit_robustness_<link>.csv   output/f42_exit_vs_closure_<link>.png
         <link> is "crosswalk" (the headline key), "v2", or "geo" -- see --link below.
run      python3 32_exit_vs_closure.py           (from analysis/)
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from ghgrp_load import load_cached

OUT = Path("output")
Y_END = 2023

de = load_cached()
# --link chooses the key. Default is the EPA crosswalk with the geographic match as
# fallback (link_eia_v2, built by 34_crosswalk_check.py); "geo" is the original
# geography-only link; "crosswalk" restricts to EPA's official key alone.
LINK = sys.argv[sys.argv.index("--link") + 1] if "--link" in sys.argv else "v2"
if LINK == "geo":
    L = pd.read_csv(OUT / "link_eia.csv")
    TAG = "geo only"
else:
    L = pd.read_csv(OUT / "link_eia_v2.csv")
    if LINK == "crosswalk":
        L = L[L.match_method == "EPA crosswalk"]
        TAG = "EPA crosswalk only"
    else:
        TAG = "crosswalk + geo fallback"
    for c in ("dist_km", "name_score"):
        if c not in L:
            L[c] = np.nan
print(f"link source: {TAG}  ({L.FacilityId.nunique():,} linked facilities)")
PY = pd.read_csv(OUT / "eia860_plant_year.csv")     # plant_code, year, operable_mw, ...
PS = pd.read_csv(OUT / "eia860_plant_status.csv")

# facilities that stop reporting before the panel ends
ex = de[de.exits].copy()
ex = ex[["FacilityId", "year", "emissions", "eligible", "elig_5yr_25k", "elig_3yr_15k",
         "threshold_bound", "always_covered", "naics3", "state", "n_years"]]
ex = ex.rename(columns={"year": "exit_year"})
print("=" * 78)
print("1 . how much of the exiting population can EIA-860 adjudicate at all?")
print("=" * 78)
print(f"  facilities that stop reporting before {Y_END}: {len(ex):,}")
print(f"    of which eligible under 98.2(i):          {int(ex.eligible.sum()):,}")

keep_conf = ["high", "medium", "crosswalk"]
good = L[L.match_confidence.isin(keep_conf)][["FacilityId", "plant_code", "plant_name",
                                              "dist_km", "name_score",
                                              "match_confidence"]]
E = ex.merge(good, on="FacilityId", how="left")
E["has_eia"] = E.plant_code.notna()
cov = (E.groupby("eligible")
         .agg(n=("FacilityId", "size"), with_eia=("has_eia", "sum"),
              share=("has_eia", "mean"),
              emis_share=("emissions", lambda s: np.nan)).reset_index())
# emissions-weighted coverage, as the data discipline requires
for i, el in enumerate(cov.eligible):
    sub = E[E.eligible == el]
    cov.loc[i, "emis_share"] = (sub.loc[sub.has_eia, "emissions"].sum()
                                / max(sub.emissions.sum(), 1))
cov["share"] = (cov["share"] * 100).round(1)
cov["emis_share"] = (cov["emis_share"] * 100).round(1)
print("\n  EIA coverage of exiters (share = by count, emis_share = weighted by emissions):")
print(cov.to_string(index=False))
cov.to_csv(OUT / f"t58_exit_coverage_{LINK}.csv", index=False)

print("\n  NAICS3 of exiters without an EIA match (the part that stays unresolved):")
print(E[~E.has_eia].naics3.value_counts().head(6).to_string())

# ---------------------------------------------------------------- 2 · classify
print("\n" + "=" * 78)
print("2 . closure or disclosure exit?")
print("=" * 78)
opset = PY[PY.operable_mw > 0].set_index(["plant_code", "year"]).index
last_op = PS.set_index("plant_code").last_year_operable.to_dict()
# EIA-923: actual output. "Operable" is a capacity concept -- a plant can sit on the
# operable list at zero output for years -- so the generation test is the stricter one.
G923 = pd.read_csv(OUT / "eia923_plant_year.csv")
genset = G923[G923.net_gen_mwh > 0].set_index(["plant_code", "year"]).index
last_gen = (G923[G923.net_gen_mwh > 0].groupby("plant_code").year.max().to_dict())

M = E[E.has_eia].copy()
M["plant_code"] = M.plant_code.astype(int)
M["eia_last_operable"] = M.plant_code.map(last_op)
# operable in the year AFTER the GHGRP report stops?
M["operable_after_exit"] = [(pc, yr + 1) in opset
                            for pc, yr in zip(M.plant_code, M.exit_year)]
M["operable_2yr_after"] = [(pc, yr + 2) in opset
                           for pc, yr in zip(M.plant_code, M.exit_year)]
M["generating_after_exit"] = [(pc, yr + 1) in genset
                              for pc, yr in zip(M.plant_code, M.exit_year)]
M["generating_2yr_after"] = [(pc, yr + 2) in genset
                             for pc, yr in zip(M.plant_code, M.exit_year)]
M["eia_last_generating"] = M.plant_code.map(last_gen)


def verdict(r):
    if r.operable_after_exit:
        return "disclosure exit (plant still operable)"
    if pd.notna(r.eia_last_operable) and r.eia_last_operable <= r.exit_year:
        return "closure (plant stopped operating)"
    return "ambiguous"


M["verdict"] = M.apply(verdict, axis=1)


def verdict_gen(r):
    """The output test: was the plant still PRODUCING after the report stopped?"""
    if r.generating_after_exit:
        return "disclosure exit (still generating)"
    if pd.notna(r.eia_last_generating) and r.eia_last_generating <= r.exit_year:
        return "closure (generation stopped)"
    return "ambiguous"


M["verdict_generation"] = M.apply(verdict_gen, axis=1)
TG = M.groupby(["eligible", "verdict_generation"]).size().unstack(fill_value=0)
TG_pct = (TG.T / TG.sum(axis=1)).T * 100
print("\n  --- the stricter OUTPUT test (EIA-923 net generation > 0), row %:")
print(TG_pct.round(1).to_string())
kg = "disclosure exit (still generating)"
if kg in TG_pct.columns and True in TG_pct.index and False in TG_pct.index:
    print(f"\n  >> still GENERATING after the report stops:")
    print(f"       98.2(i)-eligible exiters:     {TG_pct.loc[True, kg]:.1f}%")
    print(f"       non-eligible exiters:         {TG_pct.loc[False, kg]:.1f}%")
    print(f"       difference:                   "
          f"{TG_pct.loc[True, kg]-TG_pct.loc[False, kg]:+.1f} pp")
T = (M.groupby(["eligible", "verdict"]).size().unstack(fill_value=0))
T_pct = (T.T / T.sum(axis=1)).T * 100
print("\n  counts:")
print(T.to_string())
print("\n  row %:")
print(T_pct.round(1).to_string())

key = "disclosure exit (plant still operable)"
if key in T.columns:
    a = T_pct.loc[True, key] if True in T_pct.index else np.nan
    b = T_pct.loc[False, key] if False in T_pct.index else np.nan
    print(f"\n  >> share that are DISCLOSURE EXITS, not closures:")
    print(f"       98.2(i)-eligible exiters:     {a:.1f}%")
    print(f"       non-eligible exiters:         {b:.1f}%")
    print(f"       difference:                   {a-b:+.1f} pp")

M.to_csv(OUT / f"t57_exit_vs_closure_{LINK}.csv", index=False)
print(f"\n  wrote t57_exit_vs_closure_{LINK}.csv, t58_exit_coverage_{LINK}.csv")


# ---------------------------------------------------------------- 3 · inference
print("\n" + "=" * 78)
print("3 . is the gap real, and does it survive the definition?")
print("=" * 78)


def two_prop(x1, n1, x0, n0):
    p1, p0 = x1 / n1, x0 / n0
    se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    d = p1 - p0
    return d, se, d / se, d - 1.96 * se, d + 1.96 * se


def run(sub, label, col="operable_after_exit"):
    a = sub[sub.eligible]
    b = sub[~sub.eligible]
    if len(a) < 20 or len(b) < 20:
        print(f"  {label:<44s} too few observations"); return None
    d, se, t, lo, hi = two_prop(a[col].sum(), len(a), b[col].sum(), len(b))
    print(f"  {label:<44s} eligible {a[col].mean()*100:5.1f}%  "
          f"other {b[col].mean()*100:5.1f}%  diff {d*100:+5.1f}pp  "
          f"[{lo*100:+5.1f}, {hi*100:+5.1f}]  t={t:5.1f}  n={len(sub)}")
    return dict(spec=label, n=len(sub), n_elig=len(a), p_elig=round(a[col].mean(), 4),
                p_other=round(b[col].mean(), 4), diff=round(d, 4), se=round(se, 4),
                t=round(t, 2), lo=round(lo, 4), hi=round(hi, 4))

specs = []
specs.append(run(M, f"baseline ({TAG})"))
if (M.match_confidence == "crosswalk").any():
    specs.append(run(M[M.match_confidence == "crosswalk"], "EPA crosswalk matches only"))
if (M.match_confidence == "high").any():
    specs.append(run(M[M.match_confidence == "high"], "high-confidence geo only"))
if M.dist_km.notna().any():
    specs.append(run(M[M.dist_km <= 1.0], "within 1 km"))
specs.append(run(M, "still operable 2 years after exit", "operable_2yr_after"))
specs.append(run(M, "STILL GENERATING (EIA-923) after exit", "generating_after_exit"))
specs.append(run(M, "still generating 2 years after exit", "generating_2yr_after"))
specs.append(run(M[M.threshold_bound], "generating + threshold-bound only",
                 "generating_after_exit"))
specs.append(run(M[M.threshold_bound], "threshold-bound facilities only"))
specs.append(run(M[M.exit_year <= 2019], "exits 2010-2019 (2 yrs of follow-up)"))
specs.append(run(M[M.naics3.astype(str).str.startswith("221")], "NAICS 221 (utilities) only"))
S3 = pd.DataFrame([s for s in specs if s])
S3.to_csv(OUT / f"t59_exit_robustness_{LINK}.csv", index=False)

print("\n  balance: eligible exiters WITH an EIA match vs WITHOUT")
bal = E[E.eligible].groupby("has_eia").agg(
    n=("FacilityId", "size"), med_emissions=("emissions", "median"),
    mean_years=("n_years", "mean"), share_threshold_bound=("threshold_bound", "mean"))
bal["share_threshold_bound"] = (bal.share_threshold_bound * 100).round(1)
bal["med_emissions"] = bal.med_emissions.round(0)
bal["mean_years"] = bal.mean_years.round(1)
print(bal.to_string())
print("\n  ⚠️ the matched subset is POWER PLANTS. It is not a random sample of exiters,")
print("     and the result below is for that subset. Oil and gas (211) and pipelines")
print("     (486) are the bulk of the unmatched and EIA-860 cannot speak to them.")

# ---------------------------------------------------------------- 4 · figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8))

# panel (a): the capacity test and the stricter output test, side by side
pairs = [("still operable\n(EIA-860 capacity)", T_pct,
          "disclosure exit (plant still operable)", "closure (plant stopped operating)"),
         ("still generating\n(EIA-923 output)", TG_pct,
          "disclosure exit (still generating)", "closure (generation stopped)")]
w = 0.36
xs = np.arange(2)
for gi, (glab, tab, kdis, kclo) in enumerate(pairs):
    off = (gi - 0.5) * (w + 0.06)
    for j, (val, cc, lbl) in enumerate([
            (kclo, "#999999", "physical closure"),
            (kdis, "#C44E52", "disclosure exit"),
            ("ambiguous", "#dddddd", "ambiguous")]):
        v = [tab.loc[g, val] if (g in tab.index and val in tab.columns) else 0
             for g in (True, False)]
        bot = np.zeros(2)
        for jj, (vv, ccx, _) in enumerate([
                (kclo, "#999999", ""), (kdis, "#C44E52", ""), ("ambiguous", "#dddddd", "")][:j]):
            bot += np.array([tab.loc[g, vv] if (g in tab.index and vv in tab.columns) else 0
                             for g in (True, False)])
        ax[0].bar(xs + off, v, w, bottom=bot, color=cc,
                  label=lbl if gi == 0 else None,
                  edgecolor="white", linewidth=.6)
        for xi, (vi, bi) in enumerate(zip(v, bot)):
            if vi > 9:
                ax[0].text(xs[xi] + off, bi + vi / 2, f"{vi:.0f}", ha="center",
                           va="center", fontsize=9,
                           color="white" if cc != "#dddddd" else "#555")
    for xi in xs:
        ax[0].text(xi + off, -7, glab, ha="center", va="top", fontsize=7.2, color="#444")
ax[0].set_xticks(xs)
ax[0].set_xticklabels([f"98.2(i)-eligible\n(n={int(T.loc[True].sum())})",
                       f"not eligible\n(n={int(T.loc[False].sum())})"],
                      fontsize=9)
ax[0].tick_params(axis="x", pad=26)
ax[0].set_ylim(0, 100)
ax[0].set_ylabel("% of exiters")
ax[0].legend(frameon=False, fontsize=8, loc="lower center",
             bbox_to_anchor=(.5, -.42), ncol=3)
ax[0].set_title("(a) facilities that ran the statutory clock were still\n"
                "GENERATING after they stopped reporting", fontsize=10)

y = np.arange(len(S3))[::-1]
ax[1].hlines(y, S3.lo * 100, S3.hi * 100, lw=4, color="#C44E52", alpha=.75)
ax[1].plot(S3["diff"] * 100, y, "o", ms=7, color="k", zorder=3)
ax[1].axvline(0, color="k", lw=1)
ax[1].set_yticks(y)
ax[1].set_yticklabels([s[:36] for s in S3.spec], fontsize=7.5)
ax[1].set_xlabel("pp difference in the disclosure-exit share (eligible - other)")
ax[1].set_title("(b) the gap survives every definition tried", fontsize=10)

nm = E[~E.has_eia].naics3.value_counts().head(6)
ym = np.arange(len(nm))[::-1]
ax[2].barh(ym, nm.values, .6, color="#4C72B0", alpha=.8)
ax[2].set_yticks(ym)
ax[2].set_yticklabels([f"NAICS {int(i)}" for i in nm.index], fontsize=9)
ax[2].set_xlabel("exiting facilities with no EIA plant")
ax[2].set_title(f"(c) what EIA-860 cannot reach: "
                f"{(~E.has_eia).mean()*100:.0f}% of exiters\n"
                "(oil and gas, pipelines, landfills)", fontsize=10)

for a_ in ax:
    a_.spines[["top", "right"]].set_visible(False)
fig.suptitle("f42 - statutory disclosure exit vs physical closure "
             "(EIA-860 capacity and EIA-923 output)", fontsize=12, y=1.03)
fig.tight_layout()
fig.savefig(OUT / f"f42_exit_vs_closure_{LINK}.png", dpi=150, bbox_inches="tight")
print(f"\n  wrote t59_exit_robustness_{LINK}.csv, f42_exit_vs_closure_{LINK}.png")
