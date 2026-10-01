# %% [markdown]
# # 14 · Update since 20 August
#
# **Yiling Long · 3 September 2026**
#
# This notebook covers the work done in response to the 20 August notes. It is an update,
# not a recap: the panel, the 98.2(i) off-ramp, the enforcement record and the cost bound
# are established and are not re-presented here.
#
# Three things it has to say plainly at the top.
#
# | | |
# |---|---|
# | **Two earlier readings are withdrawn** | the growth-rate evidence, and the CA/WA density result |
# | **One earlier section answered the wrong question** | the "window sensitivity" section reported standard deviations when the question was about means |
# | **One design is newly available** | Marx (2018) runs on the federal panel, which the static bunching design cannot |
#
# Everything below loads results already written to `output/`. Sources:
# `09_growth_window_scan.py`, `10_gap_origin.py`, `11_bunching_windows.py`,
# `12_marx_dynamic.py`, `13_windows_and_external_validity.py`.

# %%
import warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
OUT = Path("output")


def t(name):
    return pd.read_csv(OUT / name)


# %% [markdown]
# ---
# # 1 · Growth rates near the threshold
#
# > *"You should investigate the pure year-on-year growth rates of firms near the threshold
# > and away from the threshold in the national sample. Vary the definition of 'near' and
# > 'away' and see what happens — perhaps in 1,000-ton increments from 1,000 to 10,000 tons."*
# > — 20 August
#
# ### What the previous version did, and why it was wrong
#
# The 6 August observation was about the **mean**: facilities within 5,000 of the threshold
# appeared to grow slightly less fast. The follow-up question was also about the mean. The
# August notebook tested the mean **once**, at ±5,000, found it insignificant (t = −0.69,
# p = 0.49), and then reported **only standard-deviation ratios** across four windows in the
# section titled "window sensitivity". No mean difference appears anywhere in that scan.
#
# That substitution was not motivated, and it happened after the mean came back
# insignificant. It is corrected here.
#
# **The standing rule I have adopted:** manipulation predicts a shift in *location* or a
# pile-up of *mass*. It does not predict a narrowing of variance. A standard-deviation
# comparison therefore cannot test manipulation. It can serve as a composition diagnostic,
# and that is the only claim it is allowed to carry.

# %% [markdown]
# ## 1.1 The mean, at every window from 1,000 to 10,000
#
# Windows are defined on **lagged** emissions. Two definitions of "away", because the first
# one turns out to decide the answer:
#
# - `far_all` — every other facility-year (what the earlier version used)
# - `far_band` — outside the window, but with lagged emissions in 10,000–100,000, so the
#   comparison group is of broadly comparable size

# %%
W = t("t17_growth_window_mean.csv")
piv = W[W.cutoff == 25000].pivot(index="window", columns="far_def", values="diff__raw")
piv.columns = ["away = every other facility-year", "away = comparable size (10k–100k)"]
piv.round(5)

# %% [markdown]
# > **The sign of the difference is set by the comparison group.** The near group sits at
# > 20,000–30,000; the unrestricted comparison group contains facilities a hundred times
# > larger. Against facilities of comparable size the difference is **positive**, not
# > negative. Neither version is significant at conventional levels.
#
# This is a direct answer to the question that was asked: the observation is not robust to
# the definition of "away".

# %%
r = W[(W.far_def == "far_band") & (W.cutoff == 25000)]
r[["window", "n_near", "n_far", "diff__raw", "p__raw",
   "diff__facility FE", "p__facility FE"]].round(4).to_string(index=False)

# %% [markdown]
# ## 1.2 The within-facility effect is mean reversion, and truncation causes it
#
# Adding facility fixed effects makes the real cutoff strongly significant. It does the same
# thing, with the opposite sign, at a placebo — which is the signal that something mechanical
# is going on.

# %%
M = t("t18_mean_reversion.csv")
M.groupby("cutoff")[["rel_start", "share_low_draw", "diff_facFE"]].mean().round(4)

# %% [markdown]
# `rel_start` is the mean of log(lagged emissions ÷ the facility's own mean emissions). It is
# how far below its own typical level a facility sits when it enters the window.
#
# | cutoff | rel_start | share entering on a low draw | within-facility growth difference |
# |---|---|---|---|
# | **25,000 real** | **−0.108** | **60.6%** | **+0.022** |
# | 40,000 placebo | −0.025 | 49.8% | −0.002 |
# | 60,000 placebo | +0.000 | 46.8% | −0.018 |
# | *full sample* | −0.078 | 49.7% | — |
#
# Across all 30 (cutoff × window) cells the correlation between the two right-hand columns is
# **−0.945**.
#
# ### Why the 25,000 window is selected on low draws
#
# The window is defined on $E_{t-1}$, which is the denominator of
# $g = \log(E_t) - \log(E_{t-1})$. Selecting on it is selecting on a component of the growth
# rate. And facilities whose typical level is below 25,000 never report at all, so the
# [20k, 30k] window is filled with facilities that normally sit higher and are having a bad
# year. They revert.
#
# > **The truncation that makes the level density unusable is still operating in growth-rate
# > space.** The justification I had been giving for working there — that conditioning on
# > "above 25,000 last year" makes both sides of the decision observable — is true of an
# > individual observation and false of the window's composition.
#
# ![growth window scan](output/f22_growth_window_mean.png)

# %% [markdown]
# ---
# # 2 · The gap figure: where the facility started, not where it moved
#
# The distribution of $\text{gap} = \log(E_t / 25{,}000)$ sits noticeably to the right of zero
# for threshold-bound facilities. It decomposes:
#
# $$\text{gap}_t = \underbrace{\log\!\left(\frac{E_{t-1}}{25{,}000}\right)}_{\text{where it started}} + \underbrace{\log\!\left(\frac{E_t}{E_{t-1}}\right)}_{\text{how far it moved}}$$
#
# The sample condition — lagged emissions in [20,000, 30,000] — is symmetric in tonnes. The
# density inside that band is not: threshold-bound facilities are truncated below 25,000, so
# the band holds far more observations above the line than below it.

# %%
G = t("t19_gap_origin.csv")
G[G.spec == "window scan"][["window", "group", "n", "start_below", "gap_above",
                            "move_median"]].round(4).to_string(index=False)

# %% [markdown]
# ## Two corrections that break the link
#
# | ±5,000 window | started below 25,000 | ended above 25,000 |
# |---|---|---|
# | threshold-bound | 35.2% | **57.4%** |
# | always-covered, raw | 48.4% | **52.0%** |
# | **always-covered, reweighted to the threshold-bound starting distribution** | 36.2% | **57.1%** |
#
# Match the starting distributions and the two groups coincide. Across all windows the
# correlation between "started below" and "ended above" is **−0.822**, and the median
# movement for threshold-bound facilities is **−0.0001**.
#
# > **The right-hand excess is where facilities were, not where they went.**
#
# Restricting the starting point to 24,500–25,500 gives the same answer.

# %%
G[G.spec != "window scan"][["spec", "group", "n", "start_below", "gap_above",
                            "excess", "ci_lo", "ci_hi"]].round(4).to_string(index=False)

# %% [markdown]
# ![gap origin](output/f23_gap_origin.png)

# %% [markdown]
# ---
# # 3 · Bunching in CA and WA: the full window grid
#
# > *"Investigate the potentially observed bunching in CA and WA before the threshold further.
# > Be very clear about what the windows are, and try different windows yourself. They can be
# > symmetric or asymmetric around the threshold … Placebo tests to compare against fake
# > thresholds for both of the above items would also be good."* — 20 August
#
# ## 3.1 Two constraints on the window, both of which had been left to the software
#
# **(a) `rddensity` chooses its own bandwidth, and the two specifications disagree by a factor
# of three.** In California at 25,000: the unrestricted fit uses h ≈ 3,120 (a ±3.1k window) and
# does not reject, p = 0.63; the restricted fit uses h ≈ 10,186 — a ±10.2k window, i.e.
# 15,000–35,000 — and rejects at p = 0.006. The restricted fit also rejects at 15,000 and at
# 35,000. **A specification that rejects at every cutoff is fitting the global shape of the
# size distribution, not a local discontinuity.**
#
# **(b) The fit range must stay entirely above 10,000.** The state data are themselves
# truncated at their own 10,000 reporting threshold: the 5k–10k range holds 742 observations,
# 10k–15k holds 1,588. A polynomial spanning 10,000 fits the truncation edge and the cutoff at
# the same time — which is why the 20,000 and 30,000 placebos initially came out significant.
#
# **A consequence worth stating:** 10,000 therefore **cannot serve as a positive control**, for
# exactly the reason 25,000 cannot in the federal panel. There is no cutoff in these data where
# bunching is known to exist, so the placebos are the only calibration available.
#
# So this section uses the estimator whose windows are chosen by hand and reported: the
# polynomial-counterfactual excess-mass estimator (Chetty et al. 2011; Kleven 2016).

# %% [markdown]
# ## 3.2 The grid, and what the placebos do
#
# Excluded region from 1,000 to 10,000 in 1,000-tonne increments, symmetric and asymmetric,
# at the real cutoff and at every placebo that can get a clean fit range.

# %%
S = t("t26_window_sweep_full.csv")
both = S[S["sample"] == "CA+WA"]
summary = both.groupby("cutoff").agg(cells=("sig", "size"), share_significant=("sig", "mean"))
summary["kind"] = ["REAL" if c == 25000 else "placebo" for c in summary.index]
summary.round(2)

# %% [markdown]
# | cutoff | share of cells significant |
# |---|---|
# | 22,000 placebo | 43% |
# | **25,000 real** | **36%** |
# | 28,000 placebo | 0% |
# | 30,000 placebo | 14% |
# | **35,000 placebo** | **64%** |
# | 40,000 placebo | 50% |
#
# And the estimates swing with the width of the excluded region: at 40,000 from +0.02 to
# **+4.04**; at 35,000 from +0.26 to **−2.03**.
#
# > **The real threshold is not distinguishable from the placebos, and the estimator is not
# > stable at any window in this sample.** An earlier version of this analysis reported a
# > significant California pile-up. That was one point on this grid, and it is withdrawn.
#
# ![window sweep and external validity](output/f26_windows_and_validity.png)

# %%
asym = both[(both.cutoff.isin([25000, 35000, 40000])) &
            (both.excl_shape == "asymmetric (below only)")]
asym.pivot(index="excl_width", columns="cutoff", values="excess").round(3)

# %% [markdown]
# ## 3.3 An institutional confound to record, whatever the estimator says
#
# **In California, 25,000 tCO₂e is also the cap-and-trade covered-entity threshold.** Verified
# against 17 CCR §95812: "Facility Operators — 25,000 MT CO2e per data year", and an entity
# that crosses is "classified as a covered entity as of January 1, 2013, **and for all future
# years**", on 2009–2012 data. That is an obligation to surrender allowances, orders of
# magnitude larger than a reporting or verification burden.
#
# Washington had no comparable programme until the Climate Commitment Act took effect in 2023,
# while facing the same federal reporting threshold and the same 25,000 third-party
# verification threshold (WAC 173-441-085).
#
# The split is recorded below. It inherits the instability documented in §3.2 and cannot carry
# weight on its own, but the institutional point stands: **California cannot be used to bound
# the cost of disclosure while 25,000 also prices carbon there.**

# %%
t("t22_regime_split.csv").round(3).to_string(index=False)

# %% [markdown]
# ---
# # 4 · Marx (2018): a bunching design the federal panel can support
#
# ## 4.1 The idea
#
# Marx conditions on the base-year level instead of comparing "near" against "away". Write
# $r = \log(E_t)$ and $g = \log(E_{t+1}) - r$. A facility lands exactly on the notch when
#
# $$g^*(r) = \log(25{,}000) - r$$
#
# and **$g^*$ is different for every base-year level**: a facility at 30,000 needs $g = -0.18$
# to reach the line; one at 60,000 needs $-0.88$. In the $(r, g)$ plane the notch is a
# **diagonal ridge, not a point**. Within one growth bin, the cell that hits the notch belongs
# to a single base-year level and every other base-year level is an untreated control. His
# equation 3 fits a separate polynomial in base-year level within each growth bin:
#
# $$Y_{i,t+1} = \beta \cdot \text{NearNotch}_{it} + \sum_k \sum_\gamma \alpha_{k\gamma}\, r_{it}^k \cdot \mathbf{1}[\text{gbin}_{it} = \gamma] + e_i$$
#
# ## 4.2 Why it matters here — two reasons, the second one larger
#
# 1. It puts $E_{t-1}$ on the **right-hand side of the regression** rather than using it to
#    select the sample, which is exactly the defect identified in §1.2.
#
# 2. **It never uses the density below the cutoff.** It needs facilities that start above the
#    line and move down, and the year a facility crosses downward *is* reported — and under
#    98.2(i) it keeps reporting for several years after. So the design that truncation rules
#    out in level space is available here.
#
# > **This is the first test of manipulation the federal panel can actually support.**

# %%
t("t23_marx_dynamic.csv").to_string(index=False)

# %% [markdown]
# | sample | n | growth bins | excess | 95% CI | significant |
# |---|---|---|---|---|---|
# | **federal, threshold-bound** | **30,736** | 11 | **−0.051** | **[−0.129, +0.010]** | no |
# | federal, always-covered (placebo) | 10,881 | 11 | +0.052 | [−0.127, +0.167] | no |
# | CA | 2,133 | 3 | +0.038 | [−0.235, +0.284] | no |
# | WA | 734 | 2 | +0.004 | [−0.569, +0.442] | no |
#
# **A specification that fails, reported because the placebo caught it.** Widening the base-year
# range below the cutoff — possible only in the samples that are not truncated there — makes
# the **always-covered placebo reject at −0.121 [−0.205, −0.040]**. A quadratic cannot track
# the density over that range. That specification is biased and is not used; Washington's
# apparent −0.252 under it is the same artefact.
#
# ![Marx dynamic bunching](output/f25_marx_dynamic.png)

# %% [markdown]
# ## 4.3 The extensive margin
#
# Marx's charity application separates a small intensive-margin response (2.6% bunching) from a
# much larger extensive-margin one (8–9% stop filing). The same shape appears here.

# %%
t("t25_extensive_margin.csv").round(4).to_string(index=False)

# %% [markdown]
# Exit rate falls monotonically with base-year level: **6.18%** for facilities reporting from
# inside the 98.2(i) grace period, 3.40% at 22,621, 0.60% at 91,732.
#
# ## 4.4 What is and is not implemented
#
# This implements the **logic of equation 3**, with bin counts as the dependent variable and
# the separate-polynomial-per-growth-bin counterfactual. It does **not** implement his
# section-6 MLE, which estimates the bunching and attrition parameters jointly on a flexible
# latent density — and that is where his headline numbers come from. He provides R code.
# Running it is the next step, and these estimates are directional until then.

# %% [markdown]
# ---
# # 5 · External validity: CA and WA against the rest of the national sample
#
# > *"You should see if there are any systematic differences between firms in California and
# > Washington and the rest of the firms in the threshold-bound sample, as this will speak to
# > the external validity of using CA and WA firms for the rest of the country."* — 20 August
#
# Run inside the federal panel, using the state field, so every covariate is measured
# identically for both groups. This is distinct from the CA-versus-WA comparison.
#
# **CA + WA: 4,007 facility-years, 402 facilities — 6.7% of the national threshold-bound
# sample.** The rest: 55,600 facility-years, 5,549 facilities.

# %%
t("t27_external_validity.csv")[["variable", "ca_wa", "rest", "ND", "flag"]].to_string(index=False)

# %% [markdown]
# Normalised differences, |ND| > 0.25 flagged — the same statistic as the balance table in
# notebook 01.
#
# | variable | CA+WA | rest | ND | |
# |---|---|---|---|---|
# | log emissions | 10.999 | 11.061 | −0.041 | |
# | **methane share** | **0.031** | **0.087** | **−0.330** | *** |
# | CO₂ share | 0.957 | 0.925 | +0.206 | |
# | **more than one subpart** | **0.334** | **0.501** | **−0.344** | *** |
# | CEMS installed | 0.021 | 0.018 | +0.015 | |
# | in the 20–30k band | 0.132 | 0.135 | −0.011 | |
# | panel length (years) | 12.14 | 12.03 | +0.034 | |
# | exits this year | 0.034 | 0.029 | +0.028 | |

# %%
t("t27b_external_validity_sectors.csv").rename(columns={"Unnamed: 0": "naics3"}).head(8).to_string(index=False)

# %% [markdown]
# > **Size, panel structure and position relative to the threshold are balanced** — CA and WA
# > are not an unusually large or small set of facilities. **Gas mix and industry composition
# > are not.** Methane share in CA/WA is about a third of the rest; utilities are over-weighted
# > by 10.9 percentage points and pipelines under-weighted by 9.9.
#
# A result estimated on CA/WA extrapolates to comparably sized facilities, but discounts
# sharply for methane-intensive industries — landfills, pipelines, oil and gas — which are a
# large part of the national sample. This belongs in the limitations.

# %% [markdown]
# ---
# # 6 · Backward induction on the perceived probability
#
# > *"While the low marginal cost of reporting is one potential explanation for compliance,
# > another is that the perception of the probability of being caught is higher than the actual
# > probability. Backwards induction can be used to uncover the perceived perception of being
# > caught."* — 20 August
#
# Forward: a facility reports when $p \cdot F > c$. We measure $c = \$7{,}654$ per year and
# $p \cdot F \le \$12$ per year, and roughly 8,800 facilities report anyway. The forward
# calculation cannot explain that.
#
# Backward: a facility that *chooses to report* reveals
#
# $$\hat{p} \ge \frac{c}{F} = \frac{7{,}654}{F}$$

# %%
c = 7654.0
p_true = 3.2e-5                                    # rule of three, zero cases in 14 years
F = pd.DataFrame({
    "F ($)": [30_510, 84_546, 382_473, 1_000_000],
    "source": ["ECHO median cumulative penalty",
               "smallest ICIS case under this rule",
               "largest ICIS case under this rule",
               "a deliberately inflated ceiling"]})
F["implied p-hat"] = (c / F["F ($)"]).round(4)
F["multiple of the true p"] = (c / F["F ($)"] / p_true).round(0)
F

# %% [markdown]
# | $F$ | implied $\hat p$ | multiple of the true $p$ |
# |---|---|---|
# | \$30,510 | **25.1%** | ≈ 7,800× |
# | \$84,546 | 9.1% | ≈ 2,800× |
# | \$382,473 | **2.0%** | ≈ 625× |
# | \$1,000,000 | 0.77% | ≈ 240× |
#
# > **The behaviour implies a perceived probability three to four orders of magnitude above the
# > actual one, and the conclusion does not depend on which $F$ is chosen.** Closing the gap
# > with $F$ alone would require $F \approx \$240$ million.
#
# **Two caveats stated up front.** "Being caught" costs more than the fine — back-reporting,
# reputation, state-level penalties — so $F$ should be read as the total cost of detection; and
# risk aversion means $\hat p \ge c/F$ is the risk-neutral bound. Neither moves four orders of
# magnitude.
#
# **What would separate the two remaining explanations.** If perception is the mechanism, it
# should correlate with a facility's own enforcement history. ECHO carries
# `FAC_INSPECTION_COUNT`. Do facilities with more inspection history exit less often under
# 98.2(i), or over-comply more, conditional on size and industry? A gradient supports the
# perception story; no gradient supports the alternative — that the calculation is simply never
# made, which is also consistent with the dominated-region result that frictions are large
# relative to the stake.

# %% [markdown]
# ---
# # 7 · Where this leaves the argument
#
# **Unaffected.** Facilities take the statutory exit precisely on schedule — mass points at the
# third and fifth year, placebo waiting periods null. The enforcement record: seven cases in
# fourteen years, six of them fluorinated-gas suppliers, one stationary source at \$0 through a
# voluntary audit disclosure, and zero inspections. Cost heterogeneity by pre-existing air
# permit.
#
# **Narrower than I claimed in August.** The upper bound — that the private cost is smaller than
# abating across the threshold — rested on eight tests. Three of them were measuring something
# other than what they claimed, one answered a different question from the one asked, and the
# state test does not survive its own placebos. What remains is the **dominated-region test**,
# which is immune to truncation and lies entirely above the cutoff, and **Washington**, which is
# clean but small.
#
# **A question I would like to put to you.** The dominated-region result says frictions are
# large relative to the notch. But the upper-bound argument reads "if the cost exceeded the
# abatement cost, facilities would abate; they do not; therefore the cost is smaller." With
# large frictions, "they do not" is also consistent with the frictions rather than the cost —
# so what is really identified is
#
# $$\text{disclosure cost} \; < \; \text{abatement cost} \; + \; \text{adjustment friction}$$
#
# **Should the bound be restated as a joint bound, with the friction reported as a quantity in
# its own right?** That is closer to what Kleven and Waseem do with notches, and it may be more
# informative than a bound that treats the friction as absent.

# %% [markdown]
# ---
# # 8 · Open items
#
# | | | |
# |---|---|---|
# | 1 | **Stata `bunching` / `rfbunch`** | the SSC command asks for `kink()`, `tax0()`, `tax1()` — it estimates an elasticity at a kink in a marginal rate, and our threshold is a notch in a fixed cost, so there is nothing to supply. I have hand-coded the Chetty–Kleven estimator in Stata to cross-check the Python. `rfbunch` cites Kleven–Waseem and may fit; reading its help file next |
# | 2 | **Marx's R code, the section-6 MLE** | the only way to estimate the intensive and extensive margins in one model |
# | 3 | **Firm and sector characteristics** | QCEW, EIA-923/860 for the power sector, GHGRP parent companies to Compustat. In progress |
# | 4 | **Power of the dominated-region test** | its width is \$7,654 ÷ marginal abatement cost, so it is sensitive to a figure that is EPA's own estimate. The minimum detectable depression has never been computed |
# | 5 | **Match rates for ECHO and FRS** | the denominators in the August write-up do not reconcile with the panel and are being recomputed before they are quoted again |
# | 6 | **Institutional text** | 40 CFR 98.2(i) in full, Table A-3 for the always-covered classification, and whether 17 CCR §95812 has an exit provision |
