"""
26 · Minimum detectable effect for the MARX estimator                (2026-09-13)
================================================================================
Script 22 put a minimum detectable effect on the DENSITY estimator, which is the one
that runs on the CA/WA state panels. It could not touch the federal threshold-bound
sample, because that sample is truncated at 25,000 and the density estimator does not
run there at all.

But the federal panel is the MAIN sample, and its null rests on Marx (2018) Equation 3:

    federal threshold-bound, Marx:  -0.044  [-0.114, +0.029]

That null currently has no power behind it. This file supplies it, the same way script
22 did: inject a KNOWN amount of bunching into the real data, re-run the estimator, and
record how often it is detected.

HOW BUNCHING IS INJECTED IN THIS SPACE
--------------------------------------
Marx works in (r, g) = (log base-year level relative to the notch, log growth). Landing
exactly on the notch means r + g = 0. A facility that AVOIDS the notch is one that
would have ended just ABOVE the line and instead ends just below it, so the injection
takes facility-years whose NEXT-YEAR level falls in [25,000, 25,000 + donor) and moves
a share beta of them into [25,000 - band, 25,000). The base year is untouched, which is
the point: in Marx's design r is conditioned on, and only the growth rate moves.

The estimator is re-implemented here with a vectorised count matrix so that a few
thousand replications finish in minutes rather than hours; section 1 checks it
reproduces 12_marx_dynamic.py's published estimate before anything else is done with it.

outputs  t50_marx_power.csv  f38_marx_power.png
run      python3 26_marx_power.py          (from analysis/)
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from ghgrp_load import load_cached

OUT = Path("output")
CUT = 25_000
W_R, W_G = 0.10, 0.10
R_HI, G_LO, G_HI = 1.50, -1.30, 0.60
POLY, N_EXCL = 2, 1
Z95 = 1.959964


def marx_fast(lvl0, lvl1, r_lo=0.0, poly=POLY, n_excl=N_EXCL):
    """Marx Equation 3 excess mass. Vectorised; same binning and same fit rule as
    12_marx_dynamic.py (>= poly+4 usable r bins, >= 5 observations in the notch cells)."""
    r = np.log(lvl0 / CUT)
    g = np.log(lvl1 / CUT) - r
    nR = int(np.ceil((R_HI - r_lo) / W_R))
    nG = int(np.ceil((G_HI - G_LO) / W_G))
    r_mid = r_lo + (np.arange(nR) + .5) * W_R
    g_mid = G_LO + (np.arange(nG) + .5) * W_G
    rb = np.floor((r - r_lo) / W_R).astype(int)
    gb = np.floor((g - G_LO) / W_G).astype(int)
    ok = (rb >= 0) & (rb < nR) & (gb >= 0) & (gb < nG)
    C = np.zeros((nG, nR))
    np.add.at(C, (gb[ok], rb[ok]), 1.0)

    obs_tot = cf_tot = 0.0
    cells = 0
    for a in range(nG):
        star = -g_mid[a]
        if star < r_lo or star >= R_HI:
            continue
        hit = int(np.floor((star - r_lo) / W_R))
        excl = list(range(max(hit - n_excl + 1, 0), min(hit + 1, nR)))
        keep = [b for b in range(nR) if b not in excl and C[a, b] > 0]
        if len(keep) < poly + 4 or C[a, excl].sum() < 5:
            continue
        X = np.vander(r_mid[keep], poly + 1)
        beta, *_ = np.linalg.lstsq(X, C[a, keep], rcond=None)
        cf = np.vander(r_mid[excl], poly + 1) @ beta
        obs_tot += C[a, excl].sum()
        cf_tot += max(cf.sum(), 1e-9)
        cells += 1
    if cells == 0 or cf_tot <= 0:
        return np.nan, 0
    return obs_tot / cf_tot - 1, cells


# ---------------------------------------------------------------- samples
de = load_cached()
d = de.sort_values(["FacilityId", "year"]).copy()
d["lead_em"] = d.groupby("FacilityId").emissions.shift(-1)
d["lead_year"] = d.groupby("FacilityId").year.shift(-1)
d = d[(d.lead_year == d.year + 1) & (d.emissions > 0) & (d.lead_em > 0)]
tb = d[d.threshold_bound]
ac = d[d.always_covered]

b_tb, c_tb = marx_fast(tb.emissions.values, tb.lead_em.values, 0.0)
b_ac, c_ac = marx_fast(ac.emissions.values, ac.lead_em.values, 0.0)
print("=== 1 · does the fast implementation reproduce 12_marx_dynamic.py? ===")
print(f"  federal threshold-bound  excess {b_tb:+.3f}  ({c_tb} growth cells)   "
      f"12_marx_dynamic.py published -0.044")
print(f"  federal always-covered   excess {b_ac:+.3f}  ({c_ac} growth cells)   "
      f"published +0.018")


# ---------------------------------------------------------------- 2 · injection
#
# WHERE THE DONORS HAVE TO COME FROM.  Write z1 = log(next-year level / 25,000) = r + g.
# For growth row a the estimator excludes the single r bin that contains -g_mid, so a
# facility-year sits in the excluded cell whenever
#       floor(r / W_R) - floor((r - z1) / W_R) == 0,
# which for |z1| < W_R holds or fails depending only on where r happens to sit inside its
# own bin -- a coin flip on the bin grid. In other words the excluded cell straddles the
# notch: with 0.10 log bins any facility-year within +/-10% of 25,000 (22,600 - 27,600)
# may be counted as a buncher whichever side of the line it is on. Drawing donors from
# [0, 0.10) therefore moves units that the
# estimator was already counting as bunchers, and the injection barely registers -- the
# first version of this file did exactly that and reported an excess of only +0.09 when
# EVERY unit just above the line was moved.
#
# z1 >= W_R guarantees floor(r/W_R) - floor((r-z1)/W_R) >= 1, i.e. the donor is outside
# the excluded cell. So donors are drawn from z1 in [0.10, 0.25): next-year level in
# [27,629, 32,102). They are moved to z1 in [-0.10, 0): [22,621, 25,000).
#
# This is itself a finding worth stating: the Marx design cannot see a move that both
# starts and ends within +/-10% of the threshold. A facility that would have reported
# 27,000 and reports 23,500 instead stays inside the same excluded cell and is invisible
# to it; only the density estimator (script 22) can speak to hiding that small.
DONOR_LO, DONOR_HI = W_R, 0.25
BAND_LO = -W_R          # destination: next-year level in [CUT*exp(-0.10), CUT)


def inject(lvl1, beta, rng):
    """Move a share beta of the facility-years that would have landed just ABOVE the
    notch to just BELOW it. Base-year level is never touched: Marx conditions on it."""
    y = lvl1.copy()
    z = np.log(y / CUT)
    donors = np.where((z >= DONOR_LO) & (z < DONOR_HI))[0]
    if len(donors) == 0:
        return y, 0
    k = rng.binomial(len(donors), beta)
    if k == 0:
        return y, 0
    pick = rng.choice(donors, size=k, replace=False)
    y[pick] = CUT * np.exp(rng.uniform(BAND_LO, 0.0, size=k))
    return y, k


def boot_se(lvl0, lvl1, fid, R=200, seed=0):
    """Facility-clustered bootstrap SE of the Marx excess mass, as in 12_marx_dynamic.py."""
    rng = np.random.default_rng(seed)
    uf = np.unique(fid)
    idx = {f: np.where(fid == f)[0] for f in uf}
    bs = []
    for _ in range(R):
        draw = rng.choice(uf, size=len(uf), replace=True)
        take = np.concatenate([idx[f] for f in draw])
        b, _ = marx_fast(lvl0[take], lvl1[take])
        if np.isfinite(b):
            bs.append(b)
    return float(np.std(bs, ddof=1)), len(bs)


BETAS = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55, 0.75, 1.00)


def mde_marx(df, name, R=200, seed=0):
    lvl0 = df.emissions.values.astype(float)
    lvl1 = df.lead_em.values.astype(float)
    fid = df.FacilityId.values
    b0, ncell = marx_fast(lvl0, lvl1)
    se0, nboot = boot_se(lvl0, lvl1, fid, R=R, seed=seed)
    crit = Z95 * se0

    uf = np.unique(fid)
    idx = {f: np.where(fid == f)[0] for f in uf}
    rng = np.random.default_rng(seed + 1)
    z1 = np.log(lvl1 / CUT)
    n_donor = int(((z1 >= DONOR_LO) & (z1 < DONOR_HI)).sum())

    rows = []
    for beta in BETAS:
        bs, moved, h0, hb = [], [], 0, 0
        for _ in range(R):
            # bootstrap the facilities AND inject, so the spread below is the full
            # sampling distribution of the estimate under that amount of hiding
            draw = rng.choice(uf, size=len(uf), replace=True)
            take = np.concatenate([idx[f] for f in draw])
            y, k = inject(lvl1[take], beta, rng)
            b, _ = marx_fast(lvl0[take], y)
            if not np.isfinite(b):
                continue
            bs.append(b); moved.append(k)
            h0 += int(abs(b) > crit)
            hb += int(b > b0 + crit)
        n = max(len(bs), 1)
        rows.append(dict(sample=name, beta=beta, n_moved=int(np.mean(moved)) if moved else 0,
                         mean_excess=round(float(np.mean(bs)), 3),
                         power_vs_zero=round(h0 / n, 3), power=round(hb / n, 3)))
    T = pd.DataFrame(rows)
    T["baseline_excess"] = round(b0, 3)
    T["se"] = round(se0, 3)
    T["n_obs"] = len(df)
    T["n_facilities"] = len(uf)
    T["n_donor_above"] = n_donor
    T["n_growth_cells"] = ncell
    T["n_boot_ok"] = nboot

    # MDE read off the INCREMENT over the baseline, exactly as in 22_power_mde.py
    p, e = T.power.values, (T.mean_excess.values - b0)
    mde = beta_at = np.nan
    for i in range(1, len(p)):
        if p[i - 1] < 0.80 <= p[i]:
            w = (0.80 - p[i - 1]) / max(p[i] - p[i - 1], 1e-9)
            mde = e[i - 1] + w * (e[i] - e[i - 1])
            beta_at = T.beta.values[i - 1] + w * (T.beta.values[i] - T.beta.values[i - 1])
            break
    T["mde_increment_at_80pct"] = round(mde, 3) if np.isfinite(mde) else np.nan
    T["mde_level_at_80pct"] = round(b0 + mde, 3) if np.isfinite(mde) else np.nan
    T["beta_at_80pct"] = round(beta_at, 3) if np.isfinite(beta_at) else np.nan
    return T


if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    which = sys.argv[2] if len(sys.argv) > 2 else "tb"
    src = {"tb": (tb, "federal threshold-bound"), "ac": (ac, "federal always-covered")}[which]
    print(f"\n=== 2 . power, {src[1]}  (R={R} replications per beta) ===")
    T = mde_marx(src[0], src[1], R=R)
    print(T[["beta", "n_moved", "mean_excess", "power_vs_zero", "power"]].to_string(index=False))
    print(f"  baseline {T.baseline_excess.iloc[0]:+.3f}   se {T.se.iloc[0]:.3f}   "
          f"n {T.n_obs.iloc[0]:,} facility-years / {T.n_facilities.iloc[0]:,} facilities")
    print(f"  >> MDE at 80% power: increment {T.mde_increment_at_80pct.iloc[0]}  "
          f"level {T.mde_level_at_80pct.iloc[0]}  "
          f"(= {T.beta_at_80pct.iloc[0]} of just-above-threshold facility-years hiding)")
    f = OUT / "t50_marx_power.csv"
    if f.exists() and which != "tb":
        old = pd.read_csv(f)
        old = old[old["sample"] != src[1]]
        T = pd.concat([old, T], ignore_index=True)
    T.to_csv(f, index=False)
    print(f"  wrote {f}")

    # ------------------------------------------------------------ 3 · figure
    A = pd.read_csv(f)
    ORD = ["federal threshold-bound", "federal always-covered"]
    if set(ORD).issubset(set(A["sample"])):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        COL = {"federal threshold-bound": "#C44E52", "federal always-covered": "#4C72B0"}
        fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.7))

        for k in ORD:
            s = A[A["sample"] == k].sort_values("mean_excess")
            ax[0].plot(s.mean_excess, s.power * 100, "o-", ms=5, color=COL[k], label=k)
            ax[1].plot(s.beta * 100, s.power * 100, "o-", ms=5, color=COL[k], label=k)
        for a_ in ax[:2]:
            a_.axhline(80, ls="--", lw=1.1, color="k")
            a_.text(0.02, 82, "80% power", fontsize=8.5, transform=a_.get_yaxis_transform())
            a_.set_ylabel("% of replications detected")
            a_.legend(frameon=False, fontsize=8)
        M = A.drop_duplicates("sample").set_index("sample")
        for k in ORD:
            ax[0].axvline(M.loc[k, "mde_level_at_80pct"], ls=":", lw=1.2, color=COL[k])
            ax[1].axvline(M.loc[k, "beta_at_80pct"] * 100, ls=":", lw=1.2, color=COL[k])
        ax[0].set_xlabel("Marx excess mass present in the data")
        ax[0].set_xlim(-0.2, 1.1)
        ax[0].set_title("(a) power curves: bunching injected into the real\n"
                        "federal panel, then Marx Equation 3 re-run", fontsize=10)
        ax[1].set_xlabel("% of just-above-threshold facility-years that hide")
        ax[1].set_xlim(0, 32)
        mb = M.loc["federal threshold-bound"]
        ax[1].set_title(f"(b) the main sample would have caught it if\n"
                        f"{mb.beta_at_80pct*100:.0f}% of them had hidden "
                        f"(excess {mb.mde_level_at_80pct:+.2f})", fontsize=10)

        # (c) what the design is structurally blind to
        z1 = np.log(tb.lead_em.values / CUT)
        ax[2].hist(z1[(z1 > -0.6) & (z1 < 0.6)], bins=60, color="#bbbbbb")
        ax[2].axvspan(-W_R, W_R, color="#C44E52", alpha=.16)
        ax[2].axvspan(DONOR_LO, DONOR_HI, color="#4C72B0", alpha=.20)
        ax[2].axvline(0, color="k", lw=1.2)
        ax[2].text(0, ax[2].get_ylim()[1] * .97, " notch", fontsize=8.5, va="top")
        ax[2].text(-0.335, ax[2].get_ylim()[1] * .62,
                   "one excluded cell:\nwithin +/-10% of\n25,000 the estimator\ncannot tell above\nfrom below",
                   fontsize=8, color="#8B2F33")
        ax[2].text(DONOR_LO + .005, ax[2].get_ylim()[1] * .40, "donor\nwindow",
                   fontsize=8, color="#2F4B7C")
        ax[2].set_xlabel("log(next-year emissions / 25,000)")
        ax[2].set_ylabel("facility-years")
        ax[2].set_title("(c) the blind spot the MDE is conditional on: a move\n"
                        "from 27,000 down to 23,500 stays inside one cell", fontsize=10)

        for a_ in ax:
            a_.spines[["top", "right"]].set_visible(False)
        fig.suptitle("f38 - minimum detectable effect for the Marx dynamic estimator "
                     "(federal panel)", fontsize=12, y=1.03)
        fig.tight_layout()
        fig.savefig(OUT / "f38_marx_power.png", dpi=150, bbox_inches="tight")
        print(f"  wrote {OUT/'f38_marx_power.png'}")
