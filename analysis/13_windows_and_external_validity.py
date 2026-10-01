"""
13 · Two of John's 20 August items, done properly.

ITEM 3 — "Investigate the potentially observed bunching in CA and WA further. Be very
clear about what the windows are, and try different windows yourself. They can be
symmetric or asymmetric around the threshold. Perhaps one framework is asymmetric in
1,000-ton increments up to 10,000 behind the threshold, and symmetric around the
threshold in 1,000-ton increments up to 10,000."

  Script 11 only ran 1/2/3/5k. This runs every 1,000 from 1,000 to 10,000, both shapes,
  at the real cutoff and every placebo that can get a clean fit range. Two constraints
  carried over from 11: the fit range must stay entirely above 10,000 (the state data
  are truncated there), and 10,000 cannot be a positive control for the same reason.

ITEM 1 — "You should see if there are any systematic differences between firms in
California and Washington and the rest of the firms in the threshold-bound sample, as
this will speak to the external validity of using CA and WA firms for the rest of the
country."

  This had not been attempted. Note what it is NOT: it is not CA vs WA (script 11 did
  that). It is CA+WA against the rest of the NATIONAL threshold-bound sample.

  The clean way to run it is inside the federal panel, using the state field, because
  there every facility has the same covariates measured the same way. So the comparison
  is: federally-reporting threshold-bound facilities located in CA or WA, against
  threshold-bound facilities everywhere else. Reported as normalised differences, the
  same statistic as the balance table in notebook 01, where |ND| > 0.25 is the usual
  flag.

Outputs
  output/t26_window_sweep_full.csv
  output/t27_external_validity.csv
  output/f26_windows_and_validity.png
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from ghgrp_load import load_cached

OUT = Path("output")

# =============================================================================
# ITEM 3 · the full window sweep
# =============================================================================
ca = pd.read_csv(OUT / "ca_mrr_panel.csv")
wa = pd.read_csv(OUT / "wa_ghg_panel.csv"); wa = wa[wa["Reporter Type"] == "Facility"]
S = {"CA": pd.to_numeric(ca.co2e, errors="coerce").dropna().values,
     "WA": pd.to_numeric(wa.co2e, errors="coerce").dropna().values}
S["CA+WA"] = np.concatenate([S["CA"], S["WA"]])

FLOOR = 12_000            # fit range may not reach the state truncation at 10,000


def bunch(x, cut, bin_w=1000, fit_hw=10_000, excl_lo=2000, excl_hi=0, order=4,
          B=300, seed=0):
    lo, hi = cut - fit_hw, cut + fit_hw
    edges = np.arange(lo, hi + bin_w, bin_w)
    c = np.histogram(x[(x >= lo) & (x < hi)], bins=edges)[0].astype(float)
    z = (edges[:-1] + bin_w / 2 - cut) / 1000.0
    if c.sum() < 100:
        return None
    in_excl = (z * 1000 >= -excl_lo) & ((z * 1000 < excl_hi) if excl_hi > 0 else (z < 0))
    in_below = in_excl & (z < 0)
    if in_below.sum() == 0 or (~in_excl).sum() < order + 4:
        return None
    P = np.vander(z, order + 1)
    X = np.hstack([P, np.eye(len(z))[:, in_excl]])
    beta, *_ = np.linalg.lstsq(X, c, rcond=None)
    resid = c - X @ beta

    def stat(counts):
        b, *_ = np.linalg.lstsq(X, counts, rcond=None)
        d = (P @ b[: order + 1])[in_below]
        return (counts[in_below] - d).sum() / max(d.mean(), 1e-9)

    rng = np.random.default_rng(seed)
    draws = [stat(np.maximum(c + rng.choice(resid, size=len(c), replace=True), 0))
             for _ in range(B)]
    lo_ci, hi_ci = np.percentile(draws, [2.5, 97.5])
    return dict(b=stat(c), lo=lo_ci, hi=hi_ci, sig=bool(lo_ci > 0 or hi_ci < 0),
                n_fit=int(c.sum()), n_excl=int(c[in_below].sum()))


CUTS = [(25_000, "REAL"), (22_000, "placebo"), (28_000, "placebo"),
        (30_000, "placebo"), (35_000, "placebo"), (40_000, "placebo")]
rows = []
for name, x in S.items():
    for cut, kind in CUTS:
        fh = min(10_000, cut - FLOOR)
        if fh < 6_000:
            continue
        for w in range(1000, 10_001, 1000):
            if w > fh - 2000:                     # excluded region must leave room to fit
                continue
            for shp, (el, eh) in [("symmetric", (w, w)),
                                  ("asymmetric (below only)", (w, 0))]:
                r = bunch(x, cut, fit_hw=fh, excl_lo=el, excl_hi=eh,
                          seed=abs(hash((name, cut, w, shp))) % 2**31)
                if r is None:
                    continue
                rows.append(dict(sample=name, cutoff=cut, kind=kind, excl_shape=shp,
                                 excl_width=w, fit_range=f"[{cut-fh:,}, {cut+fh:,}]",
                                 n_fit=r["n_fit"], n_excl_below=r["n_excl"],
                                 excess=round(r["b"], 3), lo=round(r["lo"], 3),
                                 hi=round(r["hi"], 3), sig=r["sig"]))
W = pd.DataFrame(rows)
W.to_csv(OUT / "t26_window_sweep_full.csv", index=False)

pd.set_option("display.width", 250)
print("=== ITEM 3 · excess mass at 25,000, every excluded-region width 1,000–10,000 ===")
for shp in ["symmetric", "asymmetric (below only)"]:
    sub = W[(W.cutoff == 25_000) & (W.excl_shape == shp)]
    print(f"\n--- {shp} ---")
    print(sub.pivot(index="excl_width", columns="sample",
                    values="excess").round(3).to_string())
    print("  significant cells: " +
          (", ".join(f"{r['sample']}@{r.excl_width//1000}k" for _, r in sub[sub.sig].iterrows())
           or "none"))

print("\n=== placebo behaviour, CA+WA, asymmetric ===")
pl = W[(W["sample"] == "CA+WA") & (W.excl_shape == "asymmetric (below only)")]
print(pl.pivot(index="excl_width", columns="cutoff", values="excess").round(3).to_string())
print("\n  share of cells significant, by cutoff:")
print(W[W["sample"] == "CA+WA"].groupby("cutoff").sig.mean().round(2).to_string())

# =============================================================================
# ITEM 1 · external validity — CA/WA against the rest of the national sample
# =============================================================================
de = load_cached()
tb = de[de.threshold_bound].copy()
tb["in_cawa"] = tb.state.isin(["CA", "WA"])
tb["log_em"] = np.log(tb.emissions.replace(0, np.nan))
tot = tb[["CO2 emissions (non-biogenic)", "Methane (CH4) emissions",
          "Nitrous Oxide (N2O) emissions"]].sum(axis=1)
tb["ch4_share"] = (tb["Methane (CH4) emissions"] / tot.replace(0, np.nan)).clip(0, 1)
tb["co2_share"] = (tb["CO2 emissions (non-biogenic)"] / tot.replace(0, np.nan)).clip(0, 1)
tb["cems_flag"] = tb.cems.astype(float)   # already boolean in the loader; an earlier version parsed it as "Y"/"N" text and silently returned 0 for both groups
tb["near_thr"] = tb.emissions.between(20_000, 30_000).astype(float)
tb["multi_subpart"] = (tb.n_subparts > 1).astype(float)

VARS = [("log emissions", "log_em"), ("methane share", "ch4_share"),
        ("CO2 share", "co2_share"), ("number of subparts", "n_subparts"),
        ("more than one subpart", "multi_subpart"), ("CEMS installed", "cems_flag"),
        ("in the 20–30k band", "near_thr"), ("panel length (years)", "n_years"),
        ("first reporting year", "first_year"), ("exits this year", "exits"),
        ("98.2(i)-eligible", "eligible")]

rows = []
A, Bg = tb[tb.in_cawa], tb[~tb.in_cawa]
for lab, col in VARS:
    a, b = pd.to_numeric(A[col], errors="coerce"), pd.to_numeric(Bg[col], errors="coerce")
    a, b = a.dropna(), b.dropna()
    if len(a) < 30 or len(b) < 30:
        continue
    nd = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
    rows.append(dict(variable=lab, ca_wa=round(a.mean(), 3), rest=round(b.mean(), 3),
                     ND=round(nd, 3), flag="***" if abs(nd) > 0.25 else ""))
E = pd.DataFrame(rows)

sec = (pd.crosstab(tb.naics3, tb.in_cawa, normalize="columns") * 100).round(1)
sec.columns = ["rest of US", "CA + WA"]
sec["difference"] = (sec["CA + WA"] - sec["rest of US"]).round(1)
sec = sec.reindex(sec["CA + WA"].sort_values(ascending=False).index).head(12)

print("\n\n=== ITEM 1 · CA+WA vs the rest of the national threshold-bound sample ===")
print(f"  facility-years: CA+WA {len(A):,}   rest {len(Bg):,}"
      f"   ({len(A)/(len(A)+len(Bg)):.1%} of the sample)")
print(f"  facilities:     CA+WA {A.FacilityId.nunique():,}   rest {Bg.FacilityId.nunique():,}")
print("\n  normalised differences (|ND| > 0.25 is the usual flag):")
print(E.to_string(index=False))
print("\n  industry composition, top 12 NAICS3 in CA+WA (% of facility-years):")
print(sec.to_string())
E.assign(block="normalised differences").to_csv(OUT / "t27_external_validity.csv", index=False)
sec.to_csv(OUT / "t27b_external_validity_sectors.csv")

# =============================================================================
# figure
# =============================================================================
plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))

for shp, c, m in [("symmetric", "#4C72B0", "s"), ("asymmetric (below only)", "#C44E52", "o")]:
    s = W[(W["sample"] == "CA+WA") & (W.cutoff == 25_000) & (W.excl_shape == shp)] \
        .sort_values("excl_width")
    ax[0].errorbar(s.excl_width / 1000, s.excess,
                   yerr=[s.excess - s.lo, s.hi - s.excess], fmt=m + "-", ms=5,
                   capsize=3, lw=1.9, color=c, label=shp)
ax[0].axhline(0, color="k", lw=1, ls=":")
ax[0].set_xlabel("excluded region, thousand tCO$_2$e"); ax[0].set_ylabel("excess mass below 25,000")
ax[0].set_title("(a) CA+WA at the real threshold,\nevery width from 1,000 to 10,000", fontsize=10)
ax[0].legend(fontsize=8)

for cut, kind in CUTS:
    s = W[(W["sample"] == "CA+WA") & (W.cutoff == cut) &
          (W.excl_shape == "asymmetric (below only)")].sort_values("excl_width")
    if not len(s):
        continue
    real = kind == "REAL"
    ax[1].plot(s.excl_width / 1000, s.excess, "o-", ms=4,
               lw=2.4 if real else 1.1, alpha=1 if real else .5,
               color="#C44E52" if real else None,
               label=f"{cut//1000}k" + (" REAL" if real else ""))
ax[1].axhline(0, color="k", lw=1, ls=":")
ax[1].set_xlabel("excluded region, thousand tCO$_2$e below the cutoff")
ax[1].set_ylabel("excess mass")
ax[1].set_title("(b) The real cutoff against every placebo\nthat gets a clean fit range", fontsize=10)
ax[1].legend(fontsize=7.5, ncol=2)

y = np.arange(len(E))
cols = ["#C44E52" if abs(v) > 0.25 else "#9AA6A2" for v in E.ND]
ax[2].barh(y, E.ND, color=cols, height=.65)
ax[2].axvline(0, color="k", lw=1)
for v in (-0.25, 0.25):
    ax[2].axvline(v, color="#C44E52", lw=1, ls="--")
ax[2].set_yticks(y); ax[2].set_yticklabels(E.variable, fontsize=8.5); ax[2].invert_yaxis()
ax[2].set_xlabel("normalised difference,  CA+WA $-$ rest of US")
ax[2].set_title("(c) External validity: are CA/WA facilities like\nthe rest of the "
                "threshold-bound sample?", fontsize=10)

plt.tight_layout()
plt.savefig(OUT / "f26_windows_and_validity.png", bbox_inches="tight")
print("\nwrote t26, t27, t27b and f26_windows_and_validity.png")
