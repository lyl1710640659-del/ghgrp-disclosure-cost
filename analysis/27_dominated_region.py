"""
27 · The dominated region, and what it does to the reading of the null      (2026-09-13)
================================================================================
TASK B. The threats table asks for a SECOND, independent estimate of the friction width,
to sit against f30's D ~ 5,000 (the excluded-region width at which the CA+WA estimate
peaks and then dies). Kleven & Waseem's dominated region is the standard second route.

THE DOMINATED REGION IN THIS SETTING
------------------------------------
At a notch the dominated region is the interval just ABOVE the threshold in which an
agent is strictly better off moving down to the threshold, whatever its preferences.
Here the notch is a lump sum: cross 25,000 and you pay the annual compliance cost dC.
A facility at 25,000 + D gives up pi * D of marginal profit by scaling back to the line
and saves dC, so it is strictly better off iff

        D  <  dC / pi ,          pi = marginal profit per tCO2e of output

and with linear profit and a pure lump-sum notch the marginal BUNCHER sits at the same
place, so the whole predicted bunching region is dC/pi wide too.

That one line is the entire content of this file, and it is uncomfortable, because dC
for the federal disclosure notch is small: c = $7,654/year (ICR, upper end).

CALIBRATING pi WITHOUT REVENUE DATA
-----------------------------------
pi is not in the GHGRP. Two anchors that are in hand:

(1) A LOWER BOUND FROM ALLOWANCE PRICES. A California covered entity that buys an
    allowance instead of abating reveals pi >= P. CARB/Quebec joint auction #45
    (19 November 2025): current-vintage settlement P = $28.32/tCO2e, floor $25.87.
    (https://ww2.arb.ca.gov/sites/default/files/2025-11/nc-nov_2025_summary_results_report.pdf)

(2) AN INTERNAL CALIBRATION FROM f30. In California the 25,000 line is ALSO the cap-and-
    trade inclusion threshold (17 CCR 95812), and that notch is not a lump sum of a few
    thousand dollars: crossing it means surrendering allowances on the WHOLE 25,000, so
    dC_CA = P * 25,000 ~ $708,000. f30 says the CA+WA bunching region is about 5,000
    wide. Setting 5,000 = dC_CA / pi gives pi ~ 5P ~ $142/tCO2e. Because the bunching
    region can only be WIDER than the dominated region, this is a LOWER bound on pi.

Apply either anchor to the FEDERAL disclosure notch and the predicted response is tiny.
That is the finding, and it changes how the null has to be written.

outputs  t51_dominated_region.csv  t52_width_mde.csv  f39_dominated_region.png
run      python3 27_dominated_region.py           (from analysis/)
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

OUT = Path("output")
CUT = 25_000
C_ANNUAL = 7654.0            # 18_irs_audit_benchmark.py, ICR total / respondents
C_PAPER = 4978.0             # paperwork only, capital and O&M stripped out
P_ALLOW = 28.32              # CARB/Quebec auction #45, 19 Nov 2025, current vintage
D_F30 = 5_000.0              # excluded-region width at which CA+WA peaks (f30)
Z95 = 1.959964


# ================================================================= 1 · arithmetic
print("=" * 78)
print("1 . how wide is the dominated region?   D = dC / pi")
print("=" * 78)
pi_grid = [P_ALLOW, 50, 100, 142, 250, 500, 1000]
rows = []
for pi in pi_grid:
    rows.append(dict(pi=round(pi, 2),
                     D_federal_c7654=C_ANNUAL / pi,
                     D_federal_c4978=C_PAPER / pi,
                     D_CA_captrade=P_ALLOW * CUT / pi))
T1 = pd.DataFrame(rows)
for c in ["D_federal_c7654", "D_federal_c4978", "D_CA_captrade"]:
    T1[c + "_pct"] = (T1[c] / CUT * 100).round(2)
    T1[c] = T1[c].round(1)
print(T1.to_string(index=False))

pi_star = P_ALLOW * CUT / D_F30
print(f"\n  pi that rationalises f30's D = {D_F30:,.0f} under the CA cap-and-trade notch:")
print(f"    pi* = P * 25,000 / D = {P_ALLOW} * 25,000 / {D_F30:,.0f} = ${pi_star:,.2f} /tCO2e"
      f"   (= {pi_star/P_ALLOW:.1f}x the allowance price)")
print(f"  the bunching region is never narrower than the dominated region, so pi >= pi*.")
print(f"\n  >> federal disclosure notch at that pi:")
for c_, lab in [(C_ANNUAL, "c = $7,654 (all)"), (C_PAPER, "c = $4,978 (paperwork)")]:
    print(f"       {lab}:  D = {c_/pi_star:6.1f} tCO2e  = {c_/pi_star/CUT*100:.2f}% of the threshold")


# ================================================================= 2 · fine density
print("\n" + "=" * 78)
print("2 . is there anything to see at that resolution?  CA+WA, 250 tCO2e bins")
print("=" * 78)
ca = pd.read_csv(OUT / "ca_mrr_panel.csv")
wa = pd.read_csv(OUT / "wa_ghg_panel.csv"); wa = wa[wa["Reporter Type"] == "Facility"]
x = np.concatenate([pd.to_numeric(ca.co2e, errors="coerce").dropna().values,
                    pd.to_numeric(wa.co2e, errors="coerce").dropna().values])
x = x[x > 0]
print(f"  CA+WA facility-years: {len(x):,}")
for w in (250, 500, 1000):
    n_dom = int(((x >= CUT) & (x < CUT + C_ANNUAL / pi_star)).sum())
    break
print(f"  facility-years inside the federal dominated region [25,000, "
      f"{CUT + C_ANNUAL/pi_star:,.0f}):  {n_dom}")
print("  -> the region is narrower than the finest bin the data support. There is no")
print("     dominated-region test of the FEDERAL notch to run; the exercise instead")
print("     bounds what any bunching test could ever have seen. Section 4.")

FINE = 250
edges = np.arange(20_000, 30_001, FINE)
h = np.histogram(x, bins=edges)[0]
print(f"\n  density either side of the line ({FINE} tCO2e bins):")
print(f"    [24,000, 25,000): {int(((x>=24000)&(x<25000)).sum()):>4}   "
      f"[25,000, 26,000): {int(((x>=25000)&(x<26000)).sum()):>4}")


# ================================================================= 3 · KW ratio
# Run the dominated-region test anyway, at the width the CA cap-and-trade notch implies
# for a given pi, since THAT notch is big enough to have a visible dominated region.
def kw_ratio(x, cut, dom_hi, fit_lo, fit_hi, excl_lo, binw=500, order=4, B=400, seed=0):
    """Observed / counterfactual density inside [cut, cut+dom_hi).
    Kleven-Waseem read the same ratio as the share of agents who do NOT optimise."""
    e = np.arange(fit_lo, fit_hi + binw, binw)
    c = np.histogram(x[(x >= fit_lo) & (x < fit_hi)], bins=e)[0].astype(float)
    d = e[:-1] + binw / 2 - cut
    z = d / 1000.0
    in_dom = (d >= 0) & (d < dom_hi)
    in_bun = (d >= -excl_lo) & (d < 0)
    in_excl = in_dom | in_bun
    if in_dom.sum() == 0 or (~in_excl).sum() < order + 4:
        return None
    P = np.vander(z, order + 1)
    X = np.hstack([P, np.eye(len(z))[:, in_excl]])

    def stat(counts):
        b, *_ = np.linalg.lstsq(X, counts, rcond=None)
        cf = (P @ b[: order + 1])[in_dom]
        return counts[in_dom].sum() / max(cf.sum(), 1e-9)

    beta, *_ = np.linalg.lstsq(X, c, rcond=None)
    resid = c - X @ beta
    rng = np.random.default_rng(seed)
    draws = [stat(np.maximum(c + rng.choice(resid, len(c), replace=True), 0)) for _ in range(B)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return dict(dom_hi=dom_hi, ratio=stat(c), lo=lo, hi=hi,
                n_dom=int(c[in_dom].sum()), n_bun=int(c[in_bun].sum()))


print("\n" + "=" * 78)
print("3 . dominated-region test for the CA CAP-AND-TRADE notch (the one big enough to")
print("    leave a footprint).  ratio = observed / counterfactual inside [25,000, 25,000+D)")
print("=" * 78)
rows = []
for dom in (500, 1000, 1500, 2000, 3000, 4000, 5000):
    r = kw_ratio(x, CUT, dom, 11_000, 45_000, D_F30)
    if r:
        r["frictions_share_1_minus_a"] = r["ratio"]
        rows.append(r)
T3 = pd.DataFrame(rows)
print(T3.round(3).to_string(index=False))
print("\n  Kleven-Waseem read this ratio as the share of agents who fail to optimise.")
print("  A ratio well below 1 over a wide interval is the signature of a real notch with")
print("  partial frictions; a ratio at 1 says nobody is responding at that width.")


# ================================================================= 4 · width MDE
# The question the arithmetic in section 1 forces: HOW NARROW A BUNCHING REGION COULD
# WE EVER HAVE SEEN?  Inject the model-consistent maximal response -- every facility-year
# in [cut, cut+W) moves into [cut-W, cut), which is what the model says they all strictly
# prefer -- and find the smallest W detected in 80% of replications. Then any c that
# implies a narrower region than that is simply outside the reach of a bunching design,
# and c <= pi * W_min is all the null can support.
def width_power(x, name, cut=CUT, binw=250, fit_lo=11_000, fit_hi=45_000,
                widths=(250, 500, 750, 1000, 1500, 2000, 3000, 5000), R=400, order=4, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for W in widths:
        e = np.arange(fit_lo, fit_hi + binw, binw)
        d = e[:-1] + binw / 2 - cut
        z = d / 1000.0
        in_excl = (d >= -W) & (d < 0)
        if in_excl.sum() == 0 or (~in_excl).sum() < order + 4:
            continue
        P = np.vander(z, order + 1)
        X = np.hstack([P, np.eye(len(z))[:, in_excl]])

        def stat(c):
            b, *_ = np.linalg.lstsq(X, c, rcond=None)
            cf = (P @ b[: order + 1])[in_excl]
            return (c[in_excl] - cf).sum() / max(cf.mean(), 1e-9)

        c0 = np.histogram(x[(x >= fit_lo) & (x < fit_hi)], bins=e)[0].astype(float)
        b0 = stat(c0)
        beta, *_ = np.linalg.lstsq(X, c0, rcond=None)
        resid = c0 - X @ beta
        se = float(np.std([stat(np.maximum(c0 + rng.choice(resid, len(c0), replace=True), 0))
                           for _ in range(R)], ddof=1))
        # the maximal model-consistent response: everyone in [cut, cut+W) moves below
        donors = int(((x >= cut) & (x < cut + W)).sum())
        hit = 0
        for _ in range(R):
            xi = x.copy()
            idx = np.where((xi >= cut) & (xi < cut + W))[0]
            xi[idx] = rng.uniform(cut - W, cut, size=len(idx))
            ci = np.histogram(xi[(xi >= fit_lo) & (xi < fit_hi)], bins=e)[0].astype(float)
            # residual noise, so this is a sampling distribution and not one number
            ci = np.maximum(ci + rng.choice(resid, len(ci), replace=True), 0)
            hit += int(stat(ci) > b0 + Z95 * se)
        out.append(dict(sample=name, width=W, n_donors_moved=donors, baseline=round(b0, 3),
                        se=round(se, 3), power=round(hit / R, 3)))
    T = pd.DataFrame(out)
    w, p = T.width.values.astype(float), T.power.values
    wmin = np.nan
    for i in range(1, len(p)):
        if p[i - 1] < 0.80 <= p[i]:
            wmin = w[i - 1] + (0.80 - p[i - 1]) / max(p[i] - p[i - 1], 1e-9) * (w[i] - w[i - 1])
            break
    T["width_at_80pct"] = round(wmin, 0) if np.isfinite(wmin) else np.nan
    return T


print("\n" + "=" * 78)
print("4 . the narrowest bunching region a design like this could have detected")
print("=" * 78)
wa_only = pd.to_numeric(wa.co2e, errors="coerce").dropna().values
wa_only = wa_only[wa_only > 0]
T4 = pd.concat([width_power(x, "CA+WA"),
                width_power(wa_only, "WA only (disclosure-only notch)")], ignore_index=True)
print(T4.to_string(index=False))
for name, g in T4.groupby("sample", sort=False):
    wmin = g.width_at_80pct.iloc[0]
    if np.isfinite(wmin):
        print(f"\n  {name}: narrowest detectable bunching region W_min = {wmin:,.0f} tCO2e")
        print(f"     -> the null can only support  c <= pi * W_min = "
              f"${pi_star*wmin:,.0f}  (at pi = ${pi_star:,.0f})")
    else:
        print(f"\n  {name}: never reaches 80% power over the widths tried")

T1.to_csv(OUT / "t51_dominated_region.csv", index=False)
T3.to_csv(OUT / "t51b_kw_ratio.csv", index=False)
T4.to_csv(OUT / "t52_width_mde.csv", index=False)
print(f"\n  wrote {OUT/'t51_dominated_region.csv'}, {OUT/'t51b_kw_ratio.csv'}, "
      f"{OUT/'t52_width_mde.csv'}")


# ================================================================= 5 · figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.7))

# (a) the two dominated regions drawn to scale on the real fine-binned density
e = np.arange(20_000, 31_001, 250)
h = np.histogram(x, bins=e)[0]
ax[0].bar(e[:-1], h, width=250, align="edge", color="#bbbbbb")
d_fed = C_ANNUAL / pi_star
ax[0].axvspan(CUT, CUT + 2000, color="#4C72B0", alpha=.18)
ax[0].axvspan(CUT, CUT + d_fed, color="#C44E52", alpha=.95)
ax[0].axvline(CUT, color="k", lw=1.2)
ax[0].annotate(f"federal disclosure notch\ndominated region = {d_fed:.0f} tCO2e\n"
               f"(the red sliver: 0.2% of 25,000)",
               xy=(CUT + d_fed, h.max() * .55), xytext=(CUT + 2600, h.max() * .80),
               fontsize=8, color="#8B2F33",
               arrowprops=dict(arrowstyle="->", color="#8B2F33", lw=1))
ax[0].text(CUT + 2100, h.max() * .33, "CA cap-and-trade notch\ndominated region ~2,000",
           fontsize=8, color="#2F4B7C")
ax[0].set_xlabel("reported emissions (tCO2e), CA + WA, 250-unit bins")
ax[0].set_ylabel("facility-years")
ax[0].set_title("(a) the two notches are four orders of magnitude apart\n"
                "in dollars, and two in the width they predict", fontsize=10)

# (b) the Kleven-Waseem ratio
ax[1].errorbar(T3.dom_hi, T3.ratio, yerr=[T3.ratio - T3.lo, T3.hi - T3.ratio],
               fmt="o-", ms=5, color="#4C72B0", capsize=3)
ax[1].axhline(1.0, ls="--", lw=1.1, color="k")
ax[1].text(4300, 1.01, "no response", fontsize=8.5)
ax[1].set_xlabel("width of the dominated region tested (tCO2e above 25,000)")
ax[1].set_ylabel("observed / counterfactual density")
ax[1].set_title("(b) second friction estimate: 0.80-0.93 out to 2,000, back to 1\n"
                "by 4,000. Only the 1,000 width excludes 1. f30 gave D ~ 5,000", fontsize=10)

# (c) the width MDE
COL = {"CA+WA": "#C44E52", "WA only (disclosure-only notch)": "#55A868"}
for k, g in T4.groupby("sample", sort=False):
    ax[2].plot(g.width, g.power * 100, "o-", ms=5, color=COL[k], label=k)
    w_ = g.width_at_80pct.iloc[0]
    if np.isfinite(w_):
        ax[2].axvline(w_, ls=":", lw=1.2, color=COL[k])
ax[2].axhline(80, ls="--", lw=1.1, color="k")
ax[2].axvline(d_fed, color="#8B2F33", lw=2)
ax[2].text(d_fed * 1.15, 30, f"what c = $7,654\nactually predicts\n({d_fed:.0f} tCO2e)",
           fontsize=8, color="#8B2F33")
ax[2].set_xscale("log")
ax[2].set_xlabel("width of the injected bunching region (tCO2e, log scale)")
ax[2].set_ylabel("% of replications detected")
ax[2].legend(frameon=False, fontsize=8, loc="lower right")
ax[2].set_title("(c) the design bottoms out around 400-600 tCO2e -- an order\n"
                "of magnitude above what the disclosure notch predicts", fontsize=10)

for a_ in ax:
    a_.spines[["top", "right"]].set_visible(False)
fig.suptitle("f39 - the dominated region, and why the federal null cannot be read as "
             "'no response'", fontsize=12, y=1.03)
fig.tight_layout()
fig.savefig(OUT / "f39_dominated_region.png", dpi=150, bbox_inches="tight")
print(f"  wrote {OUT/'f39_dominated_region.png'}")
