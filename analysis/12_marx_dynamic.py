"""
12 · Marx (2018) dynamic bunching — the estimator that survives our truncation.

WHY THIS, AND WHY NOW
---------------------
Node 4 of the research log says the standard bunching design is dead in the federal
panel: facilities below 25,000 do not report, so the density below the cutoff does not
exist. Script 09 then found that the escape route we had been using -- growth-rate
space -- is contaminated in a second way: the window is defined on E_{t-1}, which is
the denominator of the growth rate, so window membership is mechanically tied to
growth. Facilities enter a [20k, 30k] window disproportionately in a low year (60.6%
below their own mean at 25,000, against 49.8% at a 40,000 placebo) and revert.

Marx (2018, MPRA 88647) is built for exactly this pair of problems: serial dependence
that pollutes the density, and extensive-margin exit that empties the region above the
notch. His fix is to stop comparing "near" against "away" and instead
**condition on the base-year level explicitly**.

THE IDEA, IN ONE PARAGRAPH
--------------------------
Write r = log(emissions) in the base year and g = log(emissions next year) - r. A
facility lands exactly on the notch next year when

        g  =  log(25,000) - r        <-  call this g*(r)

g*(r) is DIFFERENT FOR EVERY BASE-YEAR LEVEL. A facility at 30,000 needs g = -0.18 to
reach the line; one at 60,000 needs g = -0.88. So in the (r, g) plane, "landing on the
notch" is a diagonal ridge, not a point. That is what makes the counterfactual
identified: within one growth bin, the cell that hits the notch belongs to one
base-year level, and every other base-year level in that same growth bin is an
untreated control. Marx's equation 3 is exactly this --

    Y_{i,t+1} = beta * NearNotch_it + SUM_k SUM_g alpha_{k,g} * r_it^k * 1[gbin_it = g]

-- a separate polynomial in base-year level WITHIN each growth bin, plus one dummy for
the cells where base-year level plus growth straddles the notch.

WHY IT UNLOCKS THE FEDERAL PANEL
--------------------------------
The estimator never uses the density below 25,000. It uses facilities that START above
the line and asks how they are distributed across downward growth rates. Truncation
removes facilities that are below the line for years; it does not remove a facility's
first step downward, which is still reported (and under 98.2(i) it keeps reporting for
several years after). **So the design that is impossible in level space is available in
this space** -- and unlike script 09's near/far comparison, conditioning on r means the
starting-point selection is absorbed rather than inherited.

WHAT IS AND IS NOT REPLICATED HERE
----------------------------------
This implements the LOGIC of Marx's equation 3 with bin counts as the dependent
variable, and the separate-polynomial-per-growth-bin counterfactual. It is NOT a
line-by-line replication, and it does NOT implement his section-6 MLE, which estimates
bunching and attrition jointly on a flexible Laplace latent density. The MLE is where
his headline result comes from (2.6% bunching against 8-9% extensive margin), and his
R code is the way to get it. Treat the numbers here as directional until then.

Outputs
  output/t23_marx_dynamic.csv     estimates by sample
  output/t24_marx_cells.csv       cell-level detail, for inspection
  output/f25_marx_dynamic.png
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
CUT = 25_000
RHO = np.log(CUT)

W_R, W_G = 0.10, 0.10           # bin widths in log points, for r and g
R_HI = 1.50                     # top of the base-year range
G_LO, G_HI = -1.30, 0.60        # growth range kept

# The base-year range must stop where the SAMPLE is truncated, or the polynomial is
# asked to fit the truncation edge. An earlier version of this script set R_LO = -0.10
# for every sample, which let one base-year bin BELOW 25,000 into the federal fit --
# exactly the truncated region -- and that single cell (observed 359 against a
# counterfactual of 731) drove the pooled federal estimate negative. Each sample now
# gets its own floor:
#   federal threshold-bound   0.00   truncated at the notch itself
#   always-covered           -0.85   not truncated; floor set to match the state panels
#   CA / WA                  -0.85   truncated at their own 10,000 reporting threshold
R_LO_DEFAULT = 0.0
POLY = 2                        # order of the within-growth-bin polynomial in r
N_EXCL = 1                      # notch-hitting r bins excluded from each fit


# ---------------------------------------------------------------------------
def build(level_t, level_t1, r_lo=R_LO_DEFAULT):
    """(r, g) cells from base-year and next-year levels, both in tonnes."""
    r = np.log(level_t) - RHO                        # base year, relative to the notch
    g = np.log(level_t1) - np.log(level_t)           # growth
    ok = np.isfinite(r) & np.isfinite(g) & (r >= r_lo) & (r < R_HI) & (g >= G_LO) & (g < G_HI)
    return r[ok], g[ok]


def marx(r, g, r_lo=R_LO_DEFAULT, poly=POLY, n_excl=N_EXCL, seed=0, B=400):
    """Marx-style dynamic estimate.

    For each growth bin, fit a polynomial in base-year level across base-year bins,
    excluding the bins whose (level + growth) straddles the notch, and read off the
    excess in those excluded cells. Aggregate over growth bins.
    """
    R_LO = r_lo
    rb = np.floor((r - R_LO) / W_R).astype(int)
    gb = np.floor((g - G_LO) / W_G).astype(int)
    nR = int(np.ceil((R_HI - R_LO) / W_R))
    nG = int(np.ceil((G_HI - G_LO) / W_G))
    r_mid = R_LO + (np.arange(nR) + .5) * W_R
    g_mid = G_LO + (np.arange(nG) + .5) * W_G

    C = np.zeros((nG, nR))
    for a, b in zip(gb, rb):
        if 0 <= a < nG and 0 <= b < nR:
            C[a, b] += 1

    rows, obs_tot, cf_tot = [], 0.0, 0.0
    for a in range(nG):
        # the base-year bin whose level + this growth lands on the notch: r + g = 0
        star = -g_mid[a]
        if star < R_LO or star >= R_HI:
            continue
        hit = int(np.floor((star - R_LO) / W_R))
        excl = set(range(max(hit - n_excl + 1, 0), min(hit + 1, nR)))
        keep = [b for b in range(nR) if b not in excl and C[a, b] > 0]
        if len(keep) < poly + 4 or C[a, list(excl)].sum() < 5:
            continue
        X = np.vander(r_mid[keep], poly + 1)
        beta, *_ = np.linalg.lstsq(X, C[a, keep], rcond=None)
        cf = np.vander(r_mid[sorted(excl)], poly + 1) @ beta
        obs = C[a, sorted(excl)].sum()
        cfs = max(cf.sum(), 1e-9)
        obs_tot += obs; cf_tot += cfs
        rows.append(dict(g_bin=round(g_mid[a], 3), notch_r_bin=round(r_mid[hit], 3),
                         implied_level=round(CUT * np.exp(r_mid[hit])),
                         # interior = at least two base-year bins sit BELOW the notch
                         # cell, so the counterfactual interpolates instead of
                         # extrapolating off the edge of the sample
                         interior=bool(hit >= 2),
                         n_growth_bin=int(C[a].sum()), observed=int(obs),
                         counterfactual=round(cfs, 1),
                         excess=round(obs / cfs - 1, 3)))
    if not rows:
        return None, pd.DataFrame()

    B_hat = obs_tot / cf_tot - 1

    # bootstrap over facility-years
    rng = np.random.default_rng(seed)
    n = len(r); draws = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        o, c = 0.0, 0.0
        rb2, gb2 = rb[i], gb[i]
        C2 = np.zeros((nG, nR))
        for a2, b2 in zip(gb2, rb2):
            if 0 <= a2 < nG and 0 <= b2 < nR:
                C2[a2, b2] += 1
        for a2 in range(nG):
            star = -g_mid[a2]
            if star < R_LO or star >= R_HI:
                continue
            hit = int(np.floor((star - R_LO) / W_R))
            excl = set(range(max(hit - n_excl + 1, 0), min(hit + 1, nR)))
            keep = [b2 for b2 in range(nR) if b2 not in excl and C2[a2, b2] > 0]
            # BUG FIX 2026-09-12: this guard has to be IDENTICAL to the one used for the
            # point estimate above. It was missing the "< 5 observations in the excluded
            # cells" condition, so the bootstrap aggregated over a LARGER set of growth
            # bins than the point estimate and was therefore estimating something else.
            # With many cells the two nearly coincide; with few they do not, which is why
            # every CA row came back with its point estimate OUTSIDE its own CI.
            if len(keep) < poly + 4 or C2[a2, sorted(excl)].sum() < 5:
                continue
            Xk = np.vander(r_mid[keep], poly + 1)
            bk, *_ = np.linalg.lstsq(Xk, C2[a2, keep], rcond=None)
            o += C2[a2, sorted(excl)].sum()
            c += max((np.vander(r_mid[sorted(excl)], poly + 1) @ bk).sum(), 1e-9)
        if c > 0:
            draws.append(o / c - 1)
    lo, hi = np.percentile(draws, [2.5, 97.5]) if draws else (np.nan, np.nan)
    return dict(excess=B_hat, lo=lo, hi=hi, sig=bool(lo > 0 or hi < 0),
                n=n, cells=len(rows), C=C, r_mid=r_mid, g_mid=g_mid), pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# samples
# ---------------------------------------------------------------------------
de = load_cached()
d = de.sort_values(["FacilityId", "year"]).copy()
d["lead_em"] = d.groupby("FacilityId").emissions.shift(-1)
d["lead_year"] = d.groupby("FacilityId").year.shift(-1)
d = d[(d.lead_year == d.year + 1)]                       # consecutive years only
tb = d[d.threshold_bound & (d.emissions > 0) & (d.lead_em > 0)]
ac = d[d.always_covered & (d.emissions > 0) & (d.lead_em > 0)]

ca = pd.read_csv(OUT / "ca_mrr_panel.csv")
wa = pd.read_csv(OUT / "wa_ghg_panel.csv"); wa = wa[wa["Reporter Type"] == "Facility"]
ca["co2e"] = pd.to_numeric(ca.co2e, errors="coerce")
wa["co2e"] = pd.to_numeric(wa.co2e, errors="coerce")


def state_pairs(df, idcol, y0=None, y1=None):
    x = df.dropna(subset=["co2e"]).sort_values([idcol, "year"]).copy()
    x["lead"] = x.groupby(idcol).co2e.shift(-1)
    x["ly"] = x.groupby(idcol).year.shift(-1)
    x = x[(x.ly == x.year + 1) & (x.co2e > 0) & (x.lead > 0)]
    # restrict on the BASE year, so a pair straddling a regime change is excluded
    if y0 is not None:
        x = x[x.year >= y0]
    if y1 is not None:
        x = x[x.year <= y1 - 1]          # base year t, outcome year t+1, both in regime
    return x.co2e.values, x.lead.values


SAMPLES = {
    # name: (base-year levels, next-year levels, base-year floor)
    "federal, threshold-bound":          (tb.emissions.values, tb.lead_em.values,  0.00),
    "federal, always-covered (placebo)": (ac.emissions.values, ac.lead_em.values,  0.00),
    "CA, all years":                     (*state_pairs(ca, "arb_id"),              0.00),
    "WA, all years":                     (*state_pairs(wa, "Reporter"),            0.00),
    # the state panels and always-covered are NOT truncated at 25,000, so they can be
    # run with base-year bins below the notch, which makes the counterfactual an
    # interpolation rather than an extrapolation. The federal threshold-bound sample
    # cannot -- which is the whole reason node 4 exists.
    "always-covered, floor -0.85":       (ac.emissions.values, ac.lead_em.values, -0.85),
    "CA, floor -0.85":                   (*state_pairs(ca, "arb_id"),             -0.85),
    "WA, floor -0.85":                   (*state_pairs(wa, "Reporter"),           -0.85),
    # --- regime splits, added 2026-09-12 (Elaine's 3 Sept note: "Marx for CA & WA").
    # The floor is -0.85 throughout: it puts the lowest base-year bin at
    # 25,000 * exp(-0.85) = 10,690, one step clear of the states' own 10,000 threshold,
    # so the counterfactual interpolates instead of extrapolating.
    #
    # What 25,000 means in each cell (verified in Meetings/0924/01):
    #   CA 2011-2012  cap-and-trade adopted but not yet binding
    #   CA 2013-2023  cap-and-trade binding (allowance surrender)
    #   WA 2012-2022  federal reporting ONLY -- the one clean window
    #   WA 2023-2024  Climate Commitment Act cap-and-invest binding
    # The previous label "WA (no cap-and-trade)" was wrong for 2023-24.
    "CA 2011-2012, C&T not yet binding": (*state_pairs(ca, "arb_id", None, 2013), -0.85),
    "CA 2013-2023, C&T binding":         (*state_pairs(ca, "arb_id", 2013, None), -0.85),
    "WA 2012-2022, reporting only":      (*state_pairs(wa, "Reporter", None, 2023), -0.85),
    "WA 2023-2024, CCA binding":         (*state_pairs(wa, "Reporter", 2023, None), -0.85),
}

res, cells = {}, {}
rows = []
for name, (a, b, rlo) in SAMPLES.items():
    r, g = build(np.asarray(a, float), np.asarray(b, float), r_lo=rlo)
    out, det = marx(r, g, r_lo=rlo, seed=abs(hash(name)) % 2**31)
    if out is None:
        rows.append(dict(sample=name, n=len(r), note="too few usable cells")); continue
    res[name] = out; cells[name] = det
    rows.append(dict(sample=name, floor=rlo, n=out["n"], growth_bins_used=out["cells"],
                     interior_cells=int(det.interior.sum()),
                     excess=round(out["excess"], 3), lo=round(out["lo"], 3),
                     hi=round(out["hi"], 3), sig=out["sig"]))
T = pd.DataFrame(rows)
T.to_csv(OUT / "t23_marx_dynamic.csv", index=False)
pd.set_option("display.width", 220)
print("=== Marx-style dynamic bunching at 25,000 ===")
print("  floor 0.00  = base year strictly above the notch. The only specification the")
print("               federal panel supports, so it is the one to compare across samples.")
print("  floor -0.85 = base-year bins allowed below the notch (possible only where the")
print("               sample is not truncated there). READ THE PLACEBO ROW FIRST.")
print(T.to_string(index=False))
ac0 = T[T["sample"].str.startswith("always-covered")]
print("\n  placebo check: always-covered faces no notch, so any non-zero estimate there is")
print("  the estimator's own bias. At floor 0.00 it is null; at floor -0.85 it is not.")
print(ac0.to_string(index=False))

if "federal, threshold-bound" in cells:
    print("\n=== federal threshold-bound: the notch cell moves with the base-year level ===")
    det = cells["federal, threshold-bound"]
    print(det.head(14).to_string(index=False))
    pd.concat([v.assign(sample=k) for k, v in cells.items()], ignore_index=True) \
      .to_csv(OUT / "t24_marx_cells.csv", index=False)

# ---------------------------------------------------------------------------
# extensive margin: does the facility stop reporting altogether?
# ---------------------------------------------------------------------------
print("\n=== extensive margin — exit rate by base-year level (federal, threshold-bound) ===")
dd = de.sort_values(["FacilityId", "year"]).copy()
last = dd.groupby("FacilityId").year.transform("max")
dd["exits"] = (dd.year == last) & (dd.year < dd.year.max())
e = dd[dd.threshold_bound & (dd.emissions > 0)].copy()
e["r"] = np.log(e.emissions) - RHO
# floor set below the notch on purpose: facilities in the 98.2(i) grace period are
# still reporting from under the line, and they are the ones the off-ramp is about
e = e[(e.r >= -0.30) & (e.r < R_HI)]
e["rbin"] = (np.floor((e.r + 0.30) / 0.20) * 0.20 - 0.30).round(2)
ex = e.groupby("rbin").agg(n=("exits", "size"), exit_rate=("exits", "mean")).round(4)
ex["level"] = (CUT * np.exp(ex.index)).round().astype(int)
print(ex.to_string())
ex.to_csv(OUT / "t25_extensive_margin.csv")

# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.9))

o = res.get("federal, threshold-bound")
if o is not None:
    C, rm, gm = o["C"], o["r_mid"], o["g_mid"]
    im = ax[0].imshow(np.log1p(C), origin="lower", aspect="auto", cmap="Blues",
                      extent=[rm[0] - W_R/2, rm[-1] + W_R/2, gm[0] - W_G/2, gm[-1] + W_G/2])
    ax[0].set_xlim(rm[0] - W_R/2, rm[-1] + W_R/2)
    ax[0].plot(rm, -rm, color="#c0392b", lw=2, label="$g^*(r)$ — lands exactly on 25,000")
    ax[0].set_xlabel("base-year level $r$, log points above 25,000")
    ax[0].set_ylabel("growth $g$")
    ax[0].set_title("(a) The notch is a diagonal ridge, not a point.\n"
                    "That is what identifies the counterfactual.", fontsize=10)
    ax[0].legend(fontsize=8, loc="upper right")
    fig.colorbar(im, ax=ax[0], label="log(1 + facility-years)")

    det = cells["federal, threshold-bound"]
    ax[1].axhline(0, color="k", lw=1, ls=":")
    ax[1].bar(det.g_bin, det.excess, width=W_G * .85, color="#4C72B0", edgecolor="white")
    ax[1].set_xlabel("growth bin $g$")
    ax[1].set_ylabel("excess in the notch-hitting cell")
    ax[1].set_title("(b) Federal, threshold-bound: excess by growth bin\n"
                    f"pooled {o['excess']:+.3f}  95% CI [{o['lo']:+.3f}, {o['hi']:+.3f}]",
                    fontsize=10)

y = np.arange(len(T.dropna(subset=["excess"])))
TT = T.dropna(subset=["excess"]).reset_index(drop=True)
cols = ["#C44E52" if "CA" in s else "#4C72B0" if "federal, threshold" in s
        else "#55A868" if "WA" in s else "#999999" for s in TT["sample"]]
for i in range(len(TT)):
    ax[2].plot([TT.lo[i], TT.hi[i]], [i, i], color=cols[i], lw=2)
ax[2].scatter(TT.excess, y, color=cols, s=70, zorder=3)
ax[2].axvline(0, color="k", lw=1.2, ls=":")
ax[2].set_yticks(y)
ax[2].set_yticklabels([f"{s}\n(n={int(n):,})" for s, n in zip(TT["sample"], TT.n)], fontsize=8)
ax[2].invert_yaxis()
ax[2].set_xlabel("dynamic excess mass at 25,000  (95% CI)")
ax[2].set_title("(c) Same estimator, four samples.\n"
                "always-covered is the placebo: it faces no notch.", fontsize=10)

plt.tight_layout()
plt.savefig(OUT / "f25_marx_dynamic.png", bbox_inches="tight")
print("\nwrote t23, t24, t25 and f25_marx_dynamic.png")
