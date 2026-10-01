"""
24 · Does enforcement EXPERIENCE change behaviour at the threshold?
================================================================================
John, 3 September 2026:

    "The perceptions of the probability of being caught line up well with the actual
     probabilities of being audited by the IRS, which may BLEED OVER to firms'
     perceptions of being caught for failing to report."

Node 10 showed the implied perceived probability sits in the range of IRS audit rates
and two to four orders of magnitude above GHGRP's own enforcement probability. That is a
CONSISTENCY observation, not a test. This file is the test.

THE IMPLICATION THAT CAN BE CHECKED
-----------------------------------
If firms priced GHGRP enforcement using their own experience of environmental
enforcement, then facilities that have actually been inspected, cited or fined would
hold a different (higher, better-informed) prior than facilities that never have, and
should behave differently at the threshold.

    H1  the prior is imported from elsewhere (tax)  ->  enforcement experience is
        IRRELEVANT: no gradient in threshold behaviour by inspection or penalty history
    H2  the prior is learned from environmental enforcement  ->  inspected and
        penalised facilities show MORE threshold-avoidance: more 98.2(i) exit, more
        mass parked below the line, lower growth near it

ECHO gives the variation: 59.8% of the panel has ever been inspected, 18.2% has ever
had a formal action, 15.3% has ever been penalised.

WHAT THIS CAN AND CANNOT SUPPORT
--------------------------------
ECHO Exporter's counts are LIFETIME, ACROSS ALL STATUTES, and measured as of one
download date -- they are not a time-varying treatment. So this is cross-sectional
variation in enforcement exposure, and a gradient in it is descriptive. It cannot be
read as "being inspected caused the facility to behave differently". What it CAN do is
falsify H1: if enforcement experience were the source of the prior, a gradient would
have to be there. Its absence is informative; its presence would need more work.

The obvious confound is size. Big facilities get inspected more and sit differently
relative to a fixed line, so everything is reported raw and again after removing
log emissions, NAICS3, Title V status and panel length.

outputs  t45_echo_gradient.csv  t46_echo_balance.csv  f36_echo_gradient.png
run      python3 24_echo_inspection_gradient.py        (from analysis/)
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from ghgrp_load import load_cached, add_transitions

OUT = Path("output"); ECHO = Path("data/echo")
CUT = 25_000

# ------------------------------------------------------------------ ECHO merge
# Same route as src/04_enforcement.py: FRS Id, the EPA cross-program key.
KEEP = ["REGISTRY_ID", "FAC_MAJOR_FLAG", "FAC_INSPECTION_COUNT", "FAC_INFORMAL_COUNT",
        "FAC_FORMAL_ACTION_COUNT", "FAC_PENALTY_COUNT", "FAC_TOTAL_PENALTIES"]
de = load_cached()
tgt = set(de.FRSId.dropna().astype("int64").astype(str))
chunks = []
for ch in pd.read_csv(ECHO / "ECHO_EXPORTER.csv", dtype=str, usecols=KEEP,
                      chunksize=200_000):
    chunks.append(ch[ch.REGISTRY_ID.isin(tgt)])
E = pd.concat(chunks, ignore_index=True).drop_duplicates("REGISTRY_ID")
for c in KEEP[2:]:
    E[c] = pd.to_numeric(E[c], errors="coerce").fillna(0)
E["major"] = (E.FAC_MAJOR_FLAG == "Y").astype(float)

de["rid"] = de.FRSId.astype("Int64").astype(str)
d = de.merge(E.rename(columns={"REGISTRY_ID": "rid"}), on="rid", how="left")
matched = d.FAC_INSPECTION_COUNT.notna()
print(f"panel {len(d):,} facility-years | ECHO matched {matched.sum():,} "
      f"({matched.mean():.1%})  facilities {d.loc[matched,'FacilityId'].nunique():,}")

d = d[matched & d.threshold_bound & (d.emissions > 0)].copy()
d["log_em"] = np.log(d.emissions)
d["panel_len"] = d.n_years.astype(float)
d["insp"] = d.FAC_INSPECTION_COUNT
d["insp_per_yr"] = d.insp / d.panel_len.clip(lower=1)
d["ever_penalised"] = (d.FAC_PENALTY_COUNT > 0).astype(float)
d["ever_formal"] = (d.FAC_FORMAL_ACTION_COUNT > 0).astype(float)
d["naics3"] = pd.to_numeric(d.naics3, errors="coerce")
d = d.dropna(subset=["naics3"])
print(f"threshold-bound with ECHO: {len(d):,} facility-years, "
      f"{d.FacilityId.nunique():,} facilities")
print(f"  ever inspected {(d.insp > 0).mean():.1%} | ever formal action "
      f"{d.ever_formal.mean():.1%} | ever penalised {d.ever_penalised.mean():.1%}")
print(f"  inspection count: median {d.insp.median():.0f}, p90 {d.insp.quantile(.9):.0f}, "
      f"max {d.insp.max():.0f}")


# =============================================================================
# outcomes at the threshold margin
# =============================================================================
d["in_band"] = d.emissions.between(20_000, CUT, "left").astype(float)   # parked below
d["exit_f"] = d.exits.astype(float)
d["elig_f"] = d.eligible.astype(float)

de2, tr = add_transitions(de)
tr = tr[tr.lag_em.between(20_000, 30_000) & (tr.emissions > 0)]
# tr already carries naics3 and log_em from the panel, so only the ECHO columns are
# merged in -- taking them from both sides would produce naics3_x / naics3_y.
tr = tr.merge(d[["FacilityId", "year", "insp", "ever_penalised", "ever_formal",
                 "major", "panel_len"]].drop_duplicates(["FacilityId", "year"]),
              on=["FacilityId", "year"], how="inner")
tr["naics3"] = pd.to_numeric(tr.naics3, errors="coerce")
tr["log_em"] = np.log(tr.emissions)
tr = tr.dropna(subset=["naics3"])
tr["above"] = (tr.emissions >= CUT).astype(float)

OUTCOMES = [("98.2(i) exit rate", d, "exit_f"),
            ("98.2(i) eligible", d, "elig_f"),
            ("share parked in 20-25k", d, "in_band"),
            ("ends above the line | started 20-30k", tr, "above"),
            ("growth | started 20-30k", tr, "growth")]


def resid(df, ycol):
    """Residualise on log emissions, NAICS3 fixed effects, Title V major, panel length."""
    y = df[ycol].to_numpy(float)
    dum = pd.get_dummies(df.naics3.astype(int), drop_first=True).to_numpy(float)
    X = np.column_stack([np.ones(len(df)), df.log_em.to_numpy(float),
                         df.major.to_numpy(float), df.panel_len.to_numpy(float), dum])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ b + y.mean()


def gradient(df, ycol, label):
    df = df.dropna(subset=[ycol, "insp", "log_em", "naics3", "major", "panel_len"]).copy()
    df["q"] = pd.cut(df.insp, [-.1, 0, 2, 6, 1e9],
                     labels=["0 (never inspected)", "1-2", "3-6", "7+"])
    df["r"] = resid(df, ycol)
    rows = []
    for q, g in df.groupby("q", observed=True):
        rows.append(dict(outcome=label, group=str(q), n=len(g),
                         raw=round(g[ycol].mean(), 4),
                         adjusted=round(g.r.mean(), 4),
                         se=round(g[ycol].std(ddof=1) / np.sqrt(len(g)), 4)))
    # linear gradient on log(1+inspections), with the same controls
    y = df[ycol].to_numpy(float)
    dum = pd.get_dummies(df.naics3.astype(int), drop_first=True).to_numpy(float)
    X = np.column_stack([np.ones(len(df)), np.log1p(df.insp.to_numpy(float)),
                         df.log_em.to_numpy(float), df.major.to_numpy(float),
                         df.panel_len.to_numpy(float), dum])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    XtXi = np.linalg.pinv(X.T @ X)
    cl = pd.Series(df.FacilityId.values)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for _, idx in cl.groupby(cl).indices.items():
        s = X[idx].T @ e[idx]
        meat += np.outer(s, s)
    G = cl.nunique()
    V = XtXi @ meat @ XtXi * (G / (G - 1)) * ((len(y) - 1) / (len(y) - X.shape[1]))
    se = np.sqrt(np.diag(V))[1]
    # the "ever penalised" contrast, same controls
    yp = df[ycol].to_numpy(float)
    Xp = np.column_stack([np.ones(len(df)), df.ever_penalised.to_numpy(float),
                          df.log_em.to_numpy(float), df.major.to_numpy(float),
                          df.panel_len.to_numpy(float), dum])
    bp, *_ = np.linalg.lstsq(Xp, yp, rcond=None)
    ep = yp - Xp @ bp
    XtXip = np.linalg.pinv(Xp.T @ Xp)
    meatp = np.zeros((Xp.shape[1], Xp.shape[1]))
    for _, idx in cl.groupby(cl).indices.items():
        sp = Xp[idx].T @ ep[idx]
        meatp += np.outer(sp, sp)
    Vp = XtXip @ meatp @ XtXip * (G / (G - 1)) * ((len(yp) - 1) / (len(yp) - Xp.shape[1]))
    sep = np.sqrt(np.diag(Vp))[1]
    T = pd.DataFrame(rows)
    T["beta_log1p_insp"] = round(b[1], 5)
    T["se_beta"] = round(se, 5)
    T["t_beta"] = round(b[1] / se, 2)
    T["beta_ever_penalised"] = round(bp[1], 5)
    T["t_ever_penalised"] = round(bp[1] / sep, 2)
    T["n_clusters"] = G
    return T


res = [gradient(df, c, lab) for lab, df, c in OUTCOMES]
R = pd.concat(res, ignore_index=True)
R.to_csv(OUT / "t45_echo_gradient.csv", index=False)
pd.set_option("display.width", 230)
print("\n=== gradient in threshold behaviour by enforcement experience ===")
for lab, _, _ in OUTCOMES:
    s = R[R.outcome == lab]
    print(f"\n--- {lab} ---")
    print(s[["group", "n", "raw", "adjusted"]].to_string(index=False))
    print(f"    beta on log(1+inspections) = {s.beta_log1p_insp.iloc[0]:+.5f} "
          f"(se {s.se_beta.iloc[0]:.5f}, t {s.t_beta.iloc[0]:+.2f})   |   "
          f"ever penalised = {s.beta_ever_penalised.iloc[0]:+.5f} "
          f"(t {s.t_ever_penalised.iloc[0]:+.2f})")
print("\n 'adjusted' removes log emissions, NAICS3 fixed effects, Title V major status")
print(" and panel length. Standard errors clustered on the facility.")


# =============================================================================
# why the raw gradient exists at all: who gets inspected
# =============================================================================
d["q"] = pd.cut(d.insp, [-.1, 0, 2, 6, 1e9],
                labels=["0 (never inspected)", "1-2", "3-6", "7+"])
BAL = (d.groupby("q", observed=True)
         .agg(n=("FacilityId", "size"),
              facilities=("FacilityId", "nunique"),
              median_emissions=("emissions", "median"),
              mean_log_em=("log_em", "mean"),
              titleV_major=("major", "mean"),
              panel_length=("panel_len", "mean"),
              n_subparts=("n_subparts", "mean"),
              ever_penalised=("ever_penalised", "mean"))
         .round(3).reset_index())
BAL.to_csv(OUT / "t46_echo_balance.csv", index=False)
print("\n=== who gets inspected (this is why the raw gradient exists) ===")
print(BAL.to_string(index=False))
print("\n Median emissions rise only 1.7x across the quartiles (43,760 -> 74,813), so size")
print(" alone is not the story. TITLE V MAJOR STATUS is: 6.9% -> 92.6%. Inspection count")
print(" is, first and foremost, a marker of being a Clean Air Act major source -- which")
print(" is a permitting category, not a GHGRP one.")


# =============================================================================
# figure
# =============================================================================
QL = ["0 (never inspected)", "1-2", "3-6", "7+"]
PANELS = [("98.2(i) exit rate", "%"), ("share parked in 20-25k", "%"),
          ("growth | started 20-30k", "log pts")]
fig, ax = plt.subplots(1, 4, figsize=(18, 4.3))

x = np.arange(4)
b = BAL.set_index("q").loc[QL]
ax[0].bar(x, b.median_emissions / 1000, .55, color="#8C8C8C", alpha=.85)
ax[0].set_xticks(x); ax[0].set_xticklabels(QL, rotation=20, ha="right", fontsize=8)
ax[0].set_ylabel("median emissions (thousand tCO2e)")
ax0b = ax[0].twinx()
ax0b.plot(x, b.titleV_major * 100, "o-", color="#C44E52", ms=6)
ax0b.set_ylabel("% Title V major", color="#C44E52")
ax0b.tick_params(axis="y", labelcolor="#C44E52")
ax0b.spines[["top"]].set_visible(False)
ax[0].set_title("(a) who gets inspected: median emissions rise 1.7x,\n"
                "Title V major status goes 6.9% -> 92.6%", fontsize=9.5)

for i, (lab, unit) in enumerate(PANELS, start=1):
    s = R[R.outcome == lab].set_index("group").loc[QL]
    sc = 100 if unit == "%" else 1
    ax[i].plot(x, s.raw * sc, "o-", ms=6, color="#C44E52", label="raw")
    ax[i].plot(x, s.adjusted * sc, "s--", ms=6, color="#4C72B0",
               label="adjusted for size,\nindustry, Title V, panel length")
    ax[i].set_xticks(x); ax[i].set_xticklabels(QL, rotation=20, ha="right", fontsize=8)
    ax[i].set_ylabel(unit)
    ax[i].legend(frameon=False, fontsize=7.5)
    t = R[R.outcome == lab].t_beta.iloc[0]
    ax[i].set_title(f"({'bcd'[i-1]}) {lab}\nt on log(1+inspections) = {t:+.2f}",
                    fontsize=9.5)

for a_ in ax:
    a_.spines[["top", "right"]].set_visible(False)
fig.suptitle("f36 - enforcement experience and behaviour at the threshold: "
             "the raw gradient is composition", fontsize=12, y=1.03)
fig.tight_layout()
fig.savefig(OUT / "f36_echo_gradient.png", dpi=150, bbox_inches="tight")
print("\nwrote t45, t46 and f36_echo_gradient.png")
