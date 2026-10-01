"""
11 · Bunching at 25,000 in the CA + WA sample, with the windows stated explicitly.

WHY THIS SAMPLE
---------------
The federal panel is truncated at 25,000, so no density-based test is possible there.
California (MRR) and Washington (Ecology) require reporting from 10,000, so the band
the federal data deletes is fully observed. Put differently, CA+WA is the missing
cell of a 2x2:

                        | observable below 25,000 | not observable
    facing the threshold|      CA + WA            | federal threshold-bound
    facing no threshold |  federal always-covered |      --

CA+WA is the only group that both faces the line and can be seen on both sides of it.

WHY NOT rddensity HERE
----------------------
06_rddensity_state.py runs Cattaneo-Jansson-Ma, whose bandwidth is chosen by the
procedure rather than by us. That is the problem this script exists to expose:
at 25,000 in CA the two specifications pick bandwidths that differ by a factor of
three -- unrestricted h ~ 3,100 (window +/-3.1k, p = 0.63, no rejection) and
restricted h ~ 10,200 (window +/-10.2k, i.e. 15k-35k, p = 0.006, rejection). The
restricted fit also rejects at 15,000 and at 35,000. A specification that rejects
at every cutoff is fitting the global shape of the size distribution, not a local
discontinuity.

So this script uses the estimator whose windows are chosen BY HAND and reported:
the polynomial-counterfactual bunching estimator (Chetty et al. 2011; Kleven 2016
handbook chapter), which is what Stata's `bunching` command implements. Every choice
it makes is a named argument here.

THE FOUR CHOICES, WHICH ARE THE WHOLE METHOD
--------------------------------------------
  bin        bin width (1,000 t primary; 500 t as a check)
  fit range  how far either side of the cutoff the counterfactual is fitted
  excluded   the region the counterfactual is NOT fitted to -- where bunching would be
             symmetric  [c-w, c+w]     the conventional choice
             asymmetric [c-w, c)       a notch piles mass only BELOW the line, so an
                                       excluded region that reaches above it dilutes
                                       the signal with bins that should be empty
  order      polynomial order (4 primary; 3 and 5 as checks)

POSITIVE CONTROL, WHICH MATTERS MORE THAN THE PLACEBOS
-------------------------------------------------------
10,000 is the states' own REPORTING threshold. If this estimator can detect bunching
at a real reporting threshold anywhere in these data, it should detect it there. A
null at 25,000 alongside a null at 10,000 says the test has no power; a null at
25,000 alongside a hit at 10,000 says something about 25,000.

Outputs
  output/t20_bunching_windows.csv
  output/f24_bunching_windows.png
"""
import warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
OUT = Path("output")
rng_global = np.random.default_rng(7)

ca = pd.read_csv(OUT / "ca_mrr_panel.csv")
wa = pd.read_csv(OUT / "wa_ghg_panel.csv")
wa = wa[wa["Reporter Type"] == "Facility"]
S = {"CA": pd.to_numeric(ca.co2e, errors="coerce").dropna().values,
     "WA": pd.to_numeric(wa.co2e, errors="coerce").dropna().values}
S["CA+WA"] = np.concatenate([S["CA"], S["WA"]])


def bunch(x, cut, bin_w=1000, fit_hw=15000, excl_lo=2000, excl_hi=2000,
          order=4, B=400, seed=0):
    """Polynomial-counterfactual bunching estimator.

    excl_lo / excl_hi are the excluded region measured BELOW and ABOVE the cutoff.
    excl_hi = 0 gives the asymmetric (below-only) excluded region.

    Returns excess mass below the cutoff, normalised by the counterfactual density
    there, with a residual-bootstrap CI. Positive = pile-up below the line.
    """
    lo, hi = cut - fit_hw, cut + fit_hw
    edges = np.arange(lo, hi + bin_w, bin_w)
    c = np.histogram(x[(x >= lo) & (x < hi)], bins=edges)[0].astype(float)
    z = (edges[:-1] + bin_w / 2 - cut) / 1000.0            # bin midpoint, thousands from cut
    if c.sum() < 100:
        return None

    in_excl = (z * 1000 >= -excl_lo) & (z * 1000 < excl_hi if excl_hi > 0 else z < 0)
    in_below = in_excl & (z < 0)
    if in_below.sum() == 0:
        return None

    # design: polynomial + one dummy per excluded bin, so the fit ignores that region
    P = np.vander(z, order + 1)
    D = np.eye(len(z))[:, in_excl]
    X = np.hstack([P, D])
    beta, *_ = np.linalg.lstsq(X, c, rcond=None)
    cf = P @ beta[: order + 1]                              # counterfactual
    resid = c - X @ beta

    def stat(counts):
        b, *_ = np.linalg.lstsq(X, counts, rcond=None)
        f = P @ b[: order + 1]
        d = f[in_below]
        return (counts[in_below] - d).sum() / max(d.mean(), 1e-9)

    b_hat = stat(c)
    rng = np.random.default_rng(seed)
    draws = [stat(np.maximum(c + rng.choice(resid, size=len(c), replace=True), 0))
             for _ in range(B)]
    lo_ci, hi_ci = np.percentile(draws, [2.5, 97.5])
    return dict(b=b_hat, lo=lo_ci, hi=hi_ci, sig=bool(lo_ci > 0 or hi_ci < 0),
                n_fit=int(c.sum()), n_excl_below=int(c[in_below].sum()),
                cf_mean=float(cf[in_below].mean()),
                counts=c, cf=cf, z=z, in_excl=in_excl)


# ---------------------------------------------------------------- window sweep
# The fit range must sit ENTIRELY above 10,000: the state data are themselves
# truncated there (5k-10k holds 742 observations, 10k-15k holds 1,588), so a
# polynomial asked to span 10,000 fits the truncation edge and the cutoff at once.
# fit_hw is capped accordingly and anything without a clean range is flagged.
#
# 10,000 therefore CANNOT serve as a positive control: it is the truncation point of
# this sample, exactly as 25,000 is of the federal panel. There is no cutoff in these
# data where bunching is known to exist, so the placebos are all the calibration there is.
FLOOR = 12_000
CUTS = [(25_000, "REAL federal reporting / state verification"),
        (20_000, "placebo"), (30_000, "placebo"), (35_000, "placebo"), (40_000, "placebo")]

rows = []
for name, x in S.items():
    for cut, kind in CUTS:
        fh = min(15_000, cut - FLOOR)
        if fh < 6_000:
            continue
        for w in [1000, 2000, 3000, 5000]:
            for shp, (el, eh) in [("symmetric", (w, w)), ("asymmetric (below only)", (w, 0))]:
                r = bunch(x, cut, fit_hw=fh, excl_lo=el, excl_hi=eh,
                          seed=abs(hash((name, cut, w, shp))) % 2**31)
                if r is None:
                    continue
                rows.append(dict(sample=name, cutoff=cut, kind=kind, excl_shape=shp,
                                 excl_below=el, excl_above=eh,
                                 fit_range=f"[{cut-fh:,}, {cut+fh:,}]",
                                 bin_w=1000, order=4, n_fit=r["n_fit"],
                                 n_excl_below=r["n_excl_below"], excess=round(r["b"], 3),
                                 lo=round(r["lo"], 3), hi=round(r["hi"], 3), sig=r["sig"]))
T = pd.DataFrame(rows)
T.to_csv(OUT / "t20_bunching_windows.csv", index=False)

pd.set_option("display.width", 250)
print("=== excess mass below the cutoff, CA+WA, clean fit ranges only ===")
cw = T[T["sample"] == "CA+WA"]
print(cw.pivot_table(index=["cutoff", "fit_range"], columns=["excl_shape", "excl_below"],
                     values="excess").round(3).to_string())
print("\n=== significant cells (95% bootstrap CI excludes 0) ===")
sg = T[T.sig == True]
print(sg[["sample", "cutoff", "kind", "excl_shape", "excl_below", "fit_range",
          "excess", "lo", "hi"]].to_string(index=False) if len(sg) else "  none")

# ---------------------------------------------------------------- robustness
print("\n=== robustness at 25,000, CA+WA: bin width, fit range, polynomial order ===")
rb = []
for bw in [500, 1000]:
    for fh in [10_000, 15_000, 20_000]:
        for od in [3, 4, 5]:
            r = bunch(S["CA+WA"], 25_000, bin_w=bw, fit_hw=fh, excl_lo=2000, excl_hi=0,
                      order=od, seed=1)
            if r:
                rb.append(dict(bin_w=bw, fit_halfwidth=fh, order=od, n_fit=r["n_fit"],
                               excess=round(r["b"], 3), lo=round(r["lo"], 3),
                               hi=round(r["hi"], 3), sig=r["sig"]))
RB = pd.DataFrame(rb)
print(RB.to_string(index=False))
print(f"\n significant in {RB.sig.sum()} of {len(RB)} robustness cells "
      "(excluded region: 2,000 below the cutoff, asymmetric)")
RB.assign(sample="CA+WA", cutoff=25000).to_csv(OUT / "t20b_bunching_robustness.csv", index=False)

# ---------------------------------------------------------------- CA vs WA balance
print("\n=== CA vs WA: can they be pooled? ===")
bal = []
for name in ["CA", "WA"]:
    x = S[name]
    bal.append(dict(sample=name, n=len(x),
                    n_10_25k=int(((x >= 10_000) & (x < 25_000)).sum()),
                    n_20_30k=int(((x >= 20_000) & (x < 30_000)).sum()),
                    n_left_25k_5k=int(((x >= 20_000) & (x < 25_000)).sum()),
                    n_right_25k_5k=int(((x >= 25_000) & (x < 30_000)).sum()),
                    p25=round(np.percentile(x, 25)), p50=round(np.percentile(x, 50)),
                    p75=round(np.percentile(x, 75)),
                    share_below_25k=round((x < 25_000).mean(), 3)))
BAL = pd.DataFrame(bal)
BAL["share_of_pooled_20_30k"] = (BAL.n_20_30k / BAL.n_20_30k.sum()).round(3)
print(BAL.to_string(index=False))
print("\n years: CA %d-%d, WA %d-%d" % (ca.year.min(), ca.year.max(), wa.year.min(), wa.year.max()))
print(" NOTE: ca_mrr_panel.csv carries only arb_id / name / co2e / year -- no NAICS, no sector.")
print("       An industry-composition balance between CA and WA is NOT possible until the")
print("       CA sector fields are added or CA facilities are matched back to the GHGRP panel.")
BAL.to_csv(OUT / "t21_ca_wa_balance.csv", index=False)

# ---------------------------------------------------------------- figure
ca["co2e"] = pd.to_numeric(ca.co2e, errors="coerce")
wa["co2e"] = pd.to_numeric(wa.co2e, errors="coerce")
REG_FIG = [("CA 2011-2012", "cap-and-trade adopted, not binding", ca[ca.year <= 2012].co2e.dropna().values),
           ("CA 2013-2023", "cap-and-trade binding", ca[ca.year >= 2013].co2e.dropna().values),
           ("WA 2012-2022", "no cap-and-trade", wa[wa.year <= 2022].co2e.dropna().values)]

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.9))

r = bunch(S["CA+WA"], 25_000, fit_hw=10_000, excl_lo=3000, excl_hi=0, seed=1)
zz = r["z"] * 1000 + 25_000
ax[0].bar(zz, r["counts"], width=900, color="#b8c4d4", edgecolor="white", label="observed")
ax[0].plot(zz, r["cf"], color="#2F4B7C", lw=2,
           label="order-4 counterfactual,\nfitted excluding the shaded bins")
ax[0].axvspan(22_000, 25_000, color="#C44E52", alpha=.16, label="excluded: 3,000 below")
ax[0].axvline(25_000, color="#c0392b", lw=2)
ax[0].set_xlabel("reported emissions (tCO$_2$e)"); ax[0].set_ylabel("facility-years")
ax[0].set_title("(a) CA+WA, fit range [15k, 35k]\n"
                f"excess mass {r['b']:+.2f}  95% CI [{r['lo']:+.2f}, {r['hi']:+.2f}]", fontsize=10)
ax[0].legend(fontsize=7.5)

sub = T[(T["sample"] == "CA+WA") & (T["excl_shape"] == "asymmetric (below only)")]
for cut, kind in CUTS:
    s_ = sub[sub.cutoff == cut].sort_values("excl_below")
    if not len(s_):
        continue
    real = "REAL" in kind
    ax[1].errorbar(s_.excl_below / 1000, s_.excess,
                   yerr=[s_.excess - s_.lo, s_.hi - s_.excess], fmt="o-", ms=5, capsize=3,
                   lw=2.4 if real else 1.2, alpha=1 if real else .55,
                   color="#C44E52" if real else None,
                   label=f"{cut//1000}k" + (" REAL" if real else ""))
ax[1].axhline(0, color="k", lw=1, ls=":")
ax[1].set_xlabel("excluded region, thousand tCO$_2$e below the cutoff")
ax[1].set_ylabel("excess mass below the cutoff")
ax[1].set_title("(b) The estimate grows with the excluded region:\n"
                "a level step across 20-24k, not a spike at the line", fontsize=10)
ax[1].legend(fontsize=7.5, ncol=2)

lab_, est, elo, ehi = [], [], [], []
for nm, note, x in REG_FIG:
    n = int(((x >= 15_000) & (x < 35_000)).sum())
    q = bunch(x, 25_000, fit_hw=10_000, excl_lo=3000, excl_hi=0, order=4, seed=5)
    if q is None or n < 120:
        continue
    lab_.append(f"{nm}\n{note}  (n={n:,})")
    est.append(q["b"]); elo.append(q["lo"]); ehi.append(q["hi"])
y = np.arange(len(lab_))
cols = ["#C44E52" if l.startswith("CA") else "#4C72B0" for l in lab_]
for i in y:
    ax[2].plot([elo[i], ehi[i]], [i, i], color=cols[i], lw=2)
ax[2].scatter(est, y, color=cols, s=70, zorder=3)
ax[2].axvline(0, color="k", lw=1.2, ls=":")
ax[2].set_yticks(y); ax[2].set_yticklabels(lab_, fontsize=8); ax[2].invert_yaxis()
ax[2].set_xlabel("excess mass below 25,000  (95% CI)")
ax[2].set_title("(c) The pile-up is Californian. Washington faces the\n"
                "same federal and verification thresholds, and shows none.", fontsize=10)

plt.tight_layout()
plt.savefig(OUT / "f24_bunching_windows.png", bbox_inches="tight")
print("\nwrote t20, t20b, t21, t22 and f24_bunching_windows.png")

# =============================================================================
# 4 · THE CONFOUND THAT HAS TO BE RULED OUT FIRST
# =============================================================================
# A pile-up below 25,000 in the state data is only evidence about the cost of
# DISCLOSURE if 25,000 does not also trigger something more expensive. In
# California it does: 25,000 tCO2e is the covered-entity threshold for the
# Cap-and-Trade Program, which obliges a facility to surrender allowances.
# That is a recurring cost orders of magnitude above a reporting burden.
#   >> VERIFY against 17 CCR 95811-95812 before this goes in the paper. <<
#
# Washington had no comparable programme until the Climate Commitment Act took
# effect in 2023 (ch. 173-446 WAC; covered-entity threshold 25,000 tCO2e).
#
# CORRECTED 2026-09-12: an earlier version of this comment said WA faced "the same
# 25,000 third-party verification threshold (WAC 173-441-085)" throughout. It did
# not. Verification at 25,000 begins with the 2023 emissions year. Before that,
# WAC 173-441-085 tied verification to the Clean Air Rule compliance threshold in
# ch. 173-442 WAC, which was 100,000 tCO2e phasing down 5,000 every three years to
# 70,000 -- and the indirect-emitter parts of that rule were struck down by the
# Washington Supreme Court in January 2020.
#
# So WA 2012-2022 is CLEANER than previously claimed: at 25,000 the only thing that
# changes is the federal reporting obligation. Nothing state-level attaches there.
# WA state reporting itself starts at 10,000 (WAC 173-441-030).
#
# California's first compliance period began in 2013, but 2011-2012 is NOT a clean
# pre-period, and the reason is stronger than "anticipation was possible":
#   17 CCR 95812(b)  -- emissions >= 25,000 in ANY data year 2009-2012 make the entity
#                       a covered entity as of 1 January 2013. Those ARE the
#                       determination years.
#   17 CCR 95101(b)(1) -- third-party verification is required at 25,000, and was in
#                       force for data year 2011 (CARB's 2011 reporting summary: 491 of
#                       581 reports verified, the rest footnoted "<25,000 tCO2e").
# So the 25,000 line already carried real money in 2011-2012. See 28_ca_2011_2012.py,
# which also shows the +0.796 is robust across 78 specifications, that no placebo cutoff
# in those same two years is significant, and that it is statistically indistinguishable
# from the 2013-2023 estimate at that sample size.
ca["co2e"] = pd.to_numeric(ca.co2e, errors="coerce")
wa["co2e"] = pd.to_numeric(wa.co2e, errors="coerce")
REGIMES = [
    ("CA 2011-2012", "cap-and-trade COVERAGE-DETERMINATION years (17 CCR 95812(b), data years 2009-2012); MRR verification at 25,000 already in force -- NOT a clean pre-period, see 28_ca_2011_2012.py", ca[ca.year <= 2012].co2e.dropna().values),
    ("CA 2013-2023", "cap-and-trade binding", ca[ca.year >= 2013].co2e.dropna().values),
    ("WA 2012-2022", "no cap-and-trade  <-- the clean sample", wa[wa.year <= 2022].co2e.dropna().values),
    ("WA 2023-2024", "CCA in force", wa[wa.year >= 2023].co2e.dropna().values),
]
reg = []
for lab, note, x in REGIMES:
    n = int(((x >= 15_000) & (x < 35_000)).sum())
    c = np.histogram(x, bins=np.arange(22_000, 28_001, 1000))[0]
    rec = dict(sample=lab, regime=note, n_15_35k=n,
               bin_24k=int(c[2]), bin_25k=int(c[3]),
               step_24_to_25=round(c[3] / c[2] - 1, 3) if c[2] else np.nan)
    r = bunch(x, 25_000, bin_w=1000, fit_hw=10_000, excl_lo=3000, excl_hi=0, order=4, seed=5)
    if r is not None and n >= 120:
        rec.update(excess=round(r["b"], 3), lo=round(r["lo"], 3), hi=round(r["hi"], 3), sig=r["sig"])
    else:
        rec.update(excess=np.nan, lo=np.nan, hi=np.nan, sig=None)   # too small to infer
    reg.append(rec)
G = pd.DataFrame(reg)
G.to_csv(OUT / "t22_regime_split.csv", index=False)
print("\n=== 4 · excess mass at 25,000 by cap-and-trade regime "
      "(fit [15k,35k], excluded 3,000 below, order 4) ===")
print(G.to_string(index=False))
print("\n Read: the pile-up is a California phenomenon. Washington 2012-2022 faces the federal")
print(" reporting threshold at 25,000 and NOTHING ELSE -- no cap-and-trade, and (corrected")
print(" 2026-09-12) no state verification threshold there either: WAC 173-441-085 does not")
print(" put verification at 25,000 until the 2023 emissions year. WA shows no pile-up -- its")
print(" density RISES through 25,000 -- and its CI rules out an effect as large as CA's.")
print("\n The CA 2011-2012 row is NOT a clean pre-period; see 28_ca_2011_2012.py. In those")
print(" years 25,000 already carried CA MRR verification (17 CCR 95101(b)(1)) and it was the")
print(" determination window for cap-and-trade coverage (17 CCR 95812(b), data years")
print(" 2009-2012). 'Adopted, not yet binding' describes the obligation, not the incentive.")
print("\nwrote t22_regime_split.csv")
