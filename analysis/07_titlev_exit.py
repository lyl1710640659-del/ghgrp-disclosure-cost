"""
Does pre-existing air-permit infrastructure explain why firms keep reporting?

Two rival explanations for the compliance puzzle (p*F ~ 0 yet ~8,800 facilities report):

  (A) real compliance cost is far below EPA's $7,654 for most facilities, because they
      already hold a Title V permit and already monitor and report emissions;
  (B) something other than deterrence sustains compliance (third-party information,
      norms, contracting) -- the tax-evasion "compliance puzzle" resolution.

These make opposite predictions about the 98.2(i) off-ramp. Under (A), a facility with
no pre-existing air permit pays the full cost and should take the exit MORE often than a
Title V major source, for whom the marginal cost is small. Under (B), permit status is
irrelevant to the exit decision once size is held fixed.

Permit status comes from ECHO's CAA_PERMIT_TYPES, giving an ordered gradient of
pre-existing reporting infrastructure:
    Title V major  >  minor / synthetic minor  >  no CAA permit at all

Size is the obvious confounder (Title V sources are bigger), so everything is also run
within emissions bins.

Outputs
  output/t11_titlev_exit.csv
  output/f21_titlev_exit.png
"""
import warnings
warnings.filterwarnings("ignore")
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
import ghgrp_load as G

OUT = "output"

# ------------------------------------------------------------------ panel
de = G.load_ghgrp(verbose=False)
de = G.add_eligibility(de, verbose=True)

# ------------------------------------------------------------------ permit status
echo = pd.read_csv(f"{OUT}/echo_matched.csv", dtype=str, low_memory=False)
pt = echo.CAA_PERMIT_TYPES.fillna("")
echo["permit"] = np.where(pt.str.contains("Major Emissions"), "Title V major",
                  np.where(pt.str.strip() == "", "no CAA permit",
                                                 "minor / synthetic minor"))
echo = echo[["REGISTRY_ID", "permit"]].drop_duplicates("REGISTRY_ID")

d = de.dropna(subset=["FRSId"]).copy()
d["REGISTRY_ID"] = d.FRSId.astype("int64").astype(str)
d = d.merge(echo, on="REGISTRY_ID", how="inner")

ORDER = ["Title V major", "minor / synthetic minor", "no CAA permit"]
print(f"\nmerged: {len(d):,} facility-years, {d.FacilityId.nunique():,} facilities")
print(d.drop_duplicates("FacilityId").permit.value_counts().reindex(ORDER).to_string())

# ------------------------------------------------------------------ the test
# unit of analysis: eligible facility-years. does the facility leave that year?
E = d[d.eligible].copy()


def rate(g):
    n, k = len(g), int(g.exits.sum())
    p = k / n if n else np.nan
    se = np.sqrt(p * (1 - p) / n) if n else np.nan
    return pd.Series(dict(n=n, exits=k, exit_rate=p, se=se,
                          lo=p - 1.96 * se, hi=p + 1.96 * se,
                          med_emissions=g.emissions.median()))


overall = E.groupby("permit").apply(rate).reindex(ORDER)
print("\n=== exit rate among 98.2(i)-eligible facility-years ===")
print(overall.to_string(float_format=lambda v: f"{v:,.4f}"))

# --- within emissions bins (size is the confounder)
BINS = [0, 5_000, 10_000, 15_000, 20_000, 25_000]
LAB = ["<5k", "5–10k", "10–15k", "15–20k", "20–25k"]
E["bin"] = pd.cut(E.emissions, BINS, labels=LAB, right=False)
byb = E.groupby(["bin", "permit"]).apply(rate).reset_index()
wide = byb.pivot(index="bin", columns="permit", values="exit_rate").reindex(columns=ORDER)
nwide = byb.pivot(index="bin", columns="permit", values="n").reindex(columns=ORDER)
print("\n=== exit rate within emissions bins ===")
print(wide.to_string(float_format=lambda v: f"{v:,.3f}"))
print("\n(n)")
print(nwide.to_string(float_format=lambda v: f"{v:,.0f}"))

# --- size-standardised difference: reweight each group to the pooled bin distribution
w = E.bin.value_counts(normalize=True)
std = {}
for g in ORDER:
    r = byb[byb.permit == g].set_index("bin").exit_rate.reindex(LAB)
    std[g] = float((r * w.reindex(LAB)).sum() / w.reindex(LAB)[r.notna()].sum())
print("\n=== size-standardised exit rate (pooled bin weights) ===")
for g in ORDER:
    print(f"  {g:26s} {std[g]:.3f}")

# --- linear probability model with year FE and log-emissions control
import statsmodels.formula.api as smf
E["y"] = E.exits.astype(int)
E["logE"] = np.log(E.emissions.clip(lower=1))
E["permit"] = pd.Categorical(E.permit, categories=ORDER)
for f in ["y ~ C(permit) + logE + C(year)",
          "y ~ C(permit) + logE + C(year) + C(naics2)",
          "y ~ C(permit)*near + logE + C(year)"]:
    E["naics2"] = E.naics.astype(str).str[:2]
    E["near"] = (E.emissions >= 20_000).astype(int)
    m = smf.ols(f, data=E).fit(cov_type="cluster", cov_kwds={"groups": E.FacilityId})
    keep = [i for i in m.params.index if "permit" in i or i in ("logE", "near")]
    print(f"\n=== LPM: {f}   (SE clustered by facility, n={int(m.nobs):,}) ===")
    print(m.summary2().tables[1].loc[keep].to_string(float_format=lambda v: f"{v:,.4f}"))

byb.to_csv(f"{OUT}/t11_titlev_exit.csv", index=False)

# ------------------------------------------------------------------ placebo
# if permit status simply predicts attrition, it should also predict exit among
# facility-years that are NOT eligible under 98.2(i). It should not.
N = d[~d.eligible].copy()
pl = N.groupby("permit").apply(rate).reindex(ORDER)
print("\n=== PLACEBO: exit rate among NON-eligible facility-years ===")
print(pl[["n", "exits", "exit_rate", "se"]].to_string(float_format=lambda v: f"{v:,.4f}"))
gap_e = overall.exit_rate["no CAA permit"] - overall.exit_rate["Title V major"]
gap_n = pl.exit_rate["no CAA permit"] - pl.exit_rate["Title V major"]
print(f"\n  no-permit minus Title V:   eligible {gap_e:+.3f}   non-eligible {gap_n:+.3f}")

# ------------------------------------------------------------------ difference-in-differences
# permit status predicts some general attrition, so difference it out: the object of
# interest is how much MORE the permit gap widens once the off-ramp becomes available.
P = d.copy()
P["y"] = P.exits.astype(int)
P["logE"] = np.log(P.emissions.clip(lower=1))
P["elig"] = P.eligible.astype(int)
P["permit"] = pd.Categorical(P.permit, categories=ORDER)
dd = smf.ols("y ~ C(permit)*elig + logE + C(year)", data=P).fit(
    cov_type="cluster", cov_kwds={"groups": P.FacilityId})
keep = [i for i in dd.params.index if "permit" in i or i == "elig"]
print(f"\n=== DiD: exit ~ permit x eligible + log(emissions) + year FE  (n={int(dd.nobs):,}) ===")
print(dd.summary2().tables[1].loc[keep].to_string(float_format=lambda v: f"{v:,.4f}"))

# ------------------------------------------------------------------ what I could NOT check
print("\n[note] ECHO's FAC_ACTIVE_FLAG is 'Y' or missing for every matched facility "
      "(no 'N' values), so it cannot be used to ask whether an exit was a real shutdown. "
      "That question still needs establishment-level data (NETS / QCEW).")

# ------------------------------------------------------------------ figure
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
COL = {"Title V major": "#2c3e50", "minor / synthetic minor": "#7f8c8d",
       "no CAA permit": "#c0392b"}

ax = axes[0]
x = np.arange(len(ORDER))
ax.bar(x, overall.exit_rate, color=[COL[g] for g in ORDER], width=.6)
ax.errorbar(x, overall.exit_rate,
            yerr=1.96 * overall.se, fmt="none", ecolor="k", capsize=4)
for i, g in enumerate(ORDER):
    ax.text(i, overall.hi[g] + .006,
            f"{overall.exit_rate[g]:.1%}  (n={overall.n[g]:,.0f})",
            ha="center", fontsize=9)
ax.set_ylim(0, .28)
ax.set_xticks(x)
ax.set_xticklabels([g.replace(" / ", "/\n") for g in ORDER], fontsize=9)
ax.set_ylabel("exit rate | eligible under 98.2(i)")
ax.set_title("Raw exit rate by pre-existing air-permit status", fontsize=10.5)

ax = axes[1]
for g in ORDER:
    s = byb[byb.permit == g].set_index("bin").reindex(LAB)
    ax.plot(LAB, s.exit_rate, "o-", color=COL[g], label=g)
    ax.fill_between(LAB, s.lo, s.hi, color=COL[g], alpha=.12)
ax.set_xlabel("reported emissions in the eligible year")
ax.set_ylabel("exit rate")
ax.legend(fontsize=8.5)
ax.set_title("Within emissions bins — is it permit status or just size?", fontsize=10.5)

fig.suptitle("Does already holding an air permit change who takes the 98.2(i) off-ramp?",
             fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT}/f21_titlev_exit.png", dpi=150)
print("\nwrote t11_titlev_exit.csv, f21_titlev_exit.png")
