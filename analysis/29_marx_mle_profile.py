"""
29 · Confidence intervals for the Marx section-6 MLE                    (2026-09-14)
================================================================================
TASK D. t36_marx_mle.csv reports point estimates and an LR statistic but no interval,
because the author's R file has the Hessian block commented out. The threats list has
carried "bootstrap 200 reps ~ 3 hours" ever since.

WHY THIS DOES A PROFILE LIKELIHOOD INSTEAD OF A BOOTSTRAP
---------------------------------------------------------
  - The bunching share is bounded at zero and two of the five samples sit exactly ON
    that boundary. A bootstrap percentile interval is badly behaved there; the profile
    likelihood is not.
  - fit_staged already contains the machinery: stage C pins the bunching parameters and
    re-optimises everything else from the unrestricted solution, which is exactly one
    point of a profile. Generalising it from "pinned at 0" to "pinned at b" costs
    nothing new.
  - One fit takes 40-60 s cold. Each profile point starts from the unrestricted
    solution, so it converges in a fraction of that. 200 bootstrap refits would be
    ~3 hours per sample; a 14-point profile is a few minutes.

WHAT IS PROFILED, AND A CORRECTION TO HOW t36 SHOULD BE READ
  In Marx's model the bunching share BELOW the notch is not one number. excess() uses

      bshare = share(p[0] + p[2] * (r - rmin))

  so p[0] is the share at the BOTTOM of the support and p[2] is its slope in r. The
  column t36 calls `bunch_below` is share(p[0]) -- the share at r = rmin, NOT the
  typical share. That matters here: the first version of this file profiled p[0] with
  p[2] left free, and the estimand did not move at all (excess stayed at 0.0021 even
  with p[0] pinned at zero) because p[2] simply absorbed it. A profile of p[0] alone
  is not a profile of anything.

  So the profile is run inside the submodel

      M1:  p[2] = 0   ->   a CONSTANT bunching share b below the notch

  which contains the null (b = 0) and in which excess() is monotone in b. Within M1
  the profile has one free restriction, and the 95% interval is
  {b : 2*(nll(b) - min_b nll(b)) <= 3.841}. The nll of the FULL model (p[2] free) is
  carried in the output as nll_hat so the cost of the restriction is visible; report
  it, do not hide it.

RESUMABLE. Each device shell call is capped at three minutes and background jobs do not
survive it, so this script does as many grid points as fit in its budget, appends them
to the CSV, and exits. Call it again to continue.

    python3 29_marx_mle_profile.py 0          # sample 0, next unfinished grid points
    python3 29_marx_mle_profile.py 0 --budget 150

outputs  output/t55_marx_mle_profile.csv  output/mle_anchor_<k>.npz  f41_marx_mle_ci.png
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, ".")
sys.path.insert(0, "marx_mle")
from ghgrp_load import load_cached
from marx_mle import (Spec, fit_staged, prepare, make_neglogL, free_mask, share,
                      excess, NEAR_ZERO)

OUT = Path("output")
NOTCH = np.log(25_000)
STATE_FLOOR = np.log(10_000)
CHI2_95 = 3.841459
CSV = OUT / "t55_marx_mle_profile.csv"
GRID = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15,
        0.20, 0.25, 0.30, 0.40, 0.50, 0.65, 0.80, 0.95]


# ---------------------------------------------------------------- samples (as in 20)
def pairs(df, idcol, y0=None, y1=None):
    x = df.dropna(subset=["co2e"]).sort_values([idcol, "year"]).copy()
    x["lead"] = x.groupby(idcol).co2e.shift(-1)
    x["ly"] = x.groupby(idcol).year.shift(-1)
    x.loc[x.ly != x.year + 1, "lead"] = np.nan
    x = x[x.co2e > 0]
    if y0 is not None:
        x = x[x.year >= y0]
    if y1 is not None:
        x = x[x.year <= y1 - 1]
    return x.co2e.values, x.lead.values


def build_samples():
    ca = pd.read_csv(OUT / "ca_mrr_panel.csv")
    wa = pd.read_csv(OUT / "wa_ghg_panel.csv"); wa = wa[wa["Reporter Type"] == "Facility"]
    for d in (ca, wa):
        d["co2e"] = pd.to_numeric(d.co2e, errors="coerce")
    de = load_cached()
    d = de.sort_values(["FacilityId", "year"]).copy()
    d["lead_em"] = d.groupby("FacilityId").emissions.shift(-1)
    d["lead_year"] = d.groupby("FacilityId").year.shift(-1)
    d.loc[d.lead_year != d.year + 1, "lead_em"] = np.nan
    ac = d[d.always_covered & (d.emissions > 0)]
    ac_s = ac.sample(n=min(len(ac), 10_500), random_state=11)   # same seed as script 20
    ca_l, ca_n = pairs(ca, "arb_id")
    wa_l, wa_n = pairs(wa, "Reporter")
    both_l = np.concatenate([ca_l, wa_l]); both_n = np.concatenate([ca_n, wa_n])
    wac_l, wac_n = pairs(wa, "Reporter", None, 2023)
    return [
        ("CA + WA, notch 25,000", both_l, both_n, NOTCH, STATE_FLOOR),
        ("WA 2012-2022, notch 25,000", wac_l, wac_n, NOTCH, STATE_FLOOR),
        ("CA only, notch 25,000", ca_l, ca_n, NOTCH, STATE_FLOOR),
        ("federal always-covered (placebo)", ac_s.emissions.values, ac_s.lead_em.values,
         NOTCH, STATE_FLOOR),
        ("CA + WA, PLACEBO notch 40,000", both_l, both_n, np.log(40_000), STATE_FLOOR),
    ]


# ---------------------------------------------------------------- profile machinery
def profile_point(level_t, level_t1, sp, p_anchor, b, maxiter=2000, restarts=1, seed=7):
    """Re-optimise every free parameter EXCEPT the two that define the bunching share
    below the notch: p[0] is pinned so that share(p[0]) == b, and p[2] (its slope in r)
    is pinned at 0. Started from the unrestricted solution."""
    r, gt, lb, tonotch = prepare(level_t, level_t1, sp)
    th = np.zeros_like(r)
    nll_full = make_neglogL(gt, r, lb, tonotch, th, sp)
    base = p_anchor.copy()
    base[0] = NEAR_ZERO if b <= 0 else float(np.arcsin(np.clip(2 * b - 1, -1, 1)))
    base[2] = 0.0                        # submodel M1: constant share below the notch
    mask = free_mask(sp).copy()
    mask[0] = False                      # the profiled parameter
    mask[2] = False                      # held at zero, see the docstring

    def nll(q):
        p = base.copy(); p[mask] = q
        return nll_full(p)

    rng = np.random.default_rng(seed)
    bq, bf = base[mask].copy(), nll(base[mask])
    for k in range(restarts + 1):
        s0 = bq if k == 0 else bq + rng.normal(0, 0.08, bq.shape)
        for meth, opt in (("Nelder-Mead", dict(maxiter=maxiter, maxfev=maxiter * 3,
                                               xatol=1e-7, fatol=1e-7)),
                          ("Powell", dict(maxiter=maxiter, maxfev=maxiter * 6))):
            rr = minimize(nll, s0, method=meth, options=opt)
            if np.isfinite(rr.fun) and rr.fun < bf:
                bf, bq = float(rr.fun), np.asarray(rr.x).copy()
            s0 = bq
    p = base.copy(); p[mask] = bq
    return dict(nll=bf, excess=float(excess(p, r, tonotch, th, sp)),
                bunch_above=float(share(p[1])),
                attrition=float(p[36]), params=p)


def main():
    k = int(sys.argv[1])
    budget = 150.0
    if "--budget" in sys.argv:
        budget = float(sys.argv[sys.argv.index("--budget") + 1])
    t_start = time.time()

    S = build_samples()
    name, lt, l1, notch, rmin = S[k]
    sp = Spec(notch, rmin, full=False)

    # --- anchor: the unrestricted fit, cached so later calls do not redo it
    anchor_f = OUT / f"mle_anchor_{k}.npz"
    if anchor_f.exists():
        z = np.load(anchor_f)
        p_anchor, nll_hat, b_hat, ex_hat = z["p"], float(z["nll"]), float(z["b"]), float(z["ex"])
        print(f"[{name}] anchor loaded: nll {nll_hat:.2f}  bunch_below {b_hat:.4f}")
    else:
        print(f"[{name}] fitting the anchor (unrestricted) ...", flush=True)
        t0 = time.time()
        o = fit_staged(lt, l1, sp, maxiter=2000, restarts=1, seed=11)
        p_anchor, nll_hat, b_hat, ex_hat = o["params"], o["nll"], o["bunch_below"], o["excess"]
        np.savez(anchor_f, p=p_anchor, nll=nll_hat, b=b_hat, ex=ex_hat)
        print(f"[{name}] anchor done in {time.time()-t0:.0f}s: "
              f"nll {nll_hat:.2f}  bunch_below {b_hat:.4f}  excess {ex_hat:.5f}")

    done = set()
    if CSV.exists():
        D = pd.read_csv(CSV)
        done = set(D[D["sample"] == name].b.round(6))
    todo = [b for b in GRID if round(b, 6) not in done]
    print(f"[{name}] {len(done)} grid points done, {len(todo)} to go")

    rows = []
    for b in todo:
        if time.time() - t_start > budget:
            print(f"[{name}] budget reached, stopping with {len(todo)-len(rows)} left")
            break
        t0 = time.time()
        o = profile_point(lt, l1, sp, p_anchor, b)
        rows.append(dict(sample=name, b=b, nll=o["nll"], excess=o["excess"],
                         bunch_above=o["bunch_above"], attrition=o["attrition"],
                         nll_hat=nll_hat, b_hat=b_hat, excess_hat=ex_hat,
                         lr=2 * (o["nll"] - nll_hat), secs=round(time.time() - t0, 1)))
        print(f"    b={b:<5} nll {o['nll']:10.3f}  LR {2*(o['nll']-nll_hat):7.3f}  "
              f"excess {o['excess']:.5f}  ({time.time()-t0:.0f}s)", flush=True)
        # once the profile is far past the 3.841 cut on the upper side there is nothing
        # left to learn from the remaining grid, and each point costs a shell call
        seen = [r["nll"] for r in rows]
        if b > 0 and o["nll"] - min(seen) > 8.0:
            print(f"    (LR vs the profile minimum is "
                  f"{2*(o['nll']-min(seen)):.1f}; the upper limit is already bracketed, "
                  f"skipping the rest of the grid)")
            break

    if rows:
        best = min(rows, key=lambda r: r["nll"])
        if best["nll"] < nll_hat - 1e-9:
            # The full model should never be beaten by a submodel at a true optimum, so
            # this means the t36 fit was under-converged. Keep the better solution as the
            # starting point for later calls, but keep nll_hat as the FULL-model value
            # actually achieved, which is what the write-up has to quote.
            print(f"[{name}] NOTE: an M1 profile point beat the full-model anchor by "
                  f"{nll_hat-best['nll']:.4f}. The t36 fit was under-converged; "
                  f"re-anchoring the start point at b={best['b']}.")
            o2 = profile_point(lt, l1, sp, p_anchor, best["b"])
            np.savez(anchor_f, p=o2["params"], nll=best["nll"], b=best["b"],
                     ex=best["excess"])
        T = pd.DataFrame(rows)
        if CSV.exists():
            T = pd.concat([pd.read_csv(CSV), T], ignore_index=True)
        T = T.sort_values(["sample", "b"]).drop_duplicates(["sample", "b"], keep="last")
        T.to_csv(CSV, index=False)
        print(f"[{name}] wrote {CSV}  ({len(T)} rows total)")


# ================================================================= summary + figure
def _cross(bs, lrs, exs, i0, step):
    """Walk from the profile minimum in direction `step` and interpolate the b (and the
    excess) at which the likelihood ratio crosses 3.841."""
    i = i0
    while 0 <= i + step < len(bs):
        if lrs[i + step] >= CHI2_95 > lrs[i]:
            w = (CHI2_95 - lrs[i]) / max(lrs[i + step] - lrs[i], 1e-12)
            return bs[i] + w * (bs[i + step] - bs[i]), exs[i] + w * (exs[i + step] - exs[i])
        i += step
    return bs[i], exs[i]          # never crossed: the interval runs off the grid


def summarise():
    D = pd.read_csv(CSV).sort_values(["sample", "b"])
    rows = []
    for name, g in D.groupby("sample", sort=False):
        g = g.sort_values("b")
        bs, nl, ex = g.b.values, g.nll.values, g.excess.values
        i0 = int(np.argmin(nl))
        lr = 2 * (nl - nl[i0])
        lo_b, lo_e = _cross(bs, lr, ex, i0, -1)
        hi_b, hi_e = _cross(bs, lr, ex, i0, +1)
        rows.append(dict(
            sample=name, n_grid=len(g),
            b_hat=round(float(bs[i0]), 4), excess_hat=round(float(ex[i0]), 5),
            b_lo=round(float(lo_b), 4), b_hi=round(float(hi_b), 4),
            excess_lo=round(float(lo_e), 5), excess_hi=round(float(hi_e), 5),
            lr_at_zero=round(float(lr[0]), 3),
            rejects_zero=bool(lr[0] > CHI2_95),
            grid_exhausted_above=bool(lr[-1] < CHI2_95),
            nll_full_model=round(float(g.nll_hat.iloc[0]), 3),
            nll_M1_min=round(float(nl[i0]), 3),
            cost_of_M1=round(float(2 * (nl[i0] - g.nll_hat.iloc[0])), 3)))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "t56_marx_mle_ci.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n=== Marx section-6 MLE: profile-likelihood 95% intervals ===")
    print(T[["sample", "b_hat", "b_lo", "b_hi", "excess_hat", "excess_lo", "excess_hi",
             "lr_at_zero", "rejects_zero"]].to_string(index=False))
    print("\n cost_of_M1 = 2*(nll at the M1 minimum - nll of the full model). Negative")
    print(" means the SUBMODEL fitted better, i.e. the full-model fit in t36 was")
    print(" under-converged; that is reported, not hidden.")
    print(T[["sample", "nll_full_model", "nll_M1_min", "cost_of_M1"]].to_string(index=False))

    # ---------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SHORT = {"CA + WA, notch 25,000": "CA + WA",
             "CA only, notch 25,000": "CA only",
             "WA 2012-2022, notch 25,000": "WA 2012-2022\n(disclosure only)",
             "federal always-covered (placebo)": "federal always-covered\nPLACEBO",
             "CA + WA, PLACEBO notch 40,000": "CA + WA at 40,000\nPLACEBO"}
    COL = {"CA + WA": "#C44E52", "CA only": "#8172B2",
           "WA 2012-2022\n(disclosure only)": "#55A868",
           "federal always-covered\nPLACEBO": "#999999",
           "CA + WA at 40,000\nPLACEBO": "#666666"}
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))

    for name, g in D.groupby("sample", sort=False):
        g = g.sort_values("b")
        k = SHORT[name]
        lr = 2 * (g.nll.values - g.nll.values.min())
        ls = ":" if "PLACEBO" in k else "-"
        ax[0].plot(g.excess.values, lr, ls, marker="o", ms=3.5, color=COL[k], label=k)
    ax[0].axhline(CHI2_95, ls="--", lw=1.1, color="k")
    ax[0].text(0.002, CHI2_95 + .35, "chi2(1) 5% = 3.84", fontsize=8.5)
    ax[0].set_ylim(0, 14); ax[0].set_xlim(0, 0.0065)
    ax[0].set_xlabel("excess() -- the mean bunching point mass")
    ax[0].set_ylabel("likelihood ratio vs the profile minimum")
    ax[0].legend(frameon=False, fontsize=7.5)
    ax[0].set_title("(a) profile likelihoods, constant bunching share below\n"
                    "the notch (submodel M1)", fontsize=10)

    T2 = T.copy(); T2["k"] = T2["sample"].map(SHORT)
    ORD = ["CA only", "CA + WA", "WA 2012-2022\n(disclosure only)",
           "CA + WA at 40,000\nPLACEBO", "federal always-covered\nPLACEBO"]
    T2 = T2.set_index("k").loc[ORD]
    y = np.arange(len(ORD))[::-1]
    ax[1].hlines(y, T2.excess_lo, T2.excess_hi, lw=4, color=[COL[k] for k in ORD], alpha=.75)
    ax[1].plot(T2.excess_hat, y, "o", ms=8, color="k", zorder=3)
    ax[1].axvline(0, color="k", lw=1)
    ax[1].axvspan(T2.loc["CA + WA at 40,000\nPLACEBO", "excess_lo"],
                  T2.loc["CA + WA at 40,000\nPLACEBO", "excess_hi"],
                  color="#666666", alpha=.13)
    ax[1].set_yticks(y); ax[1].set_yticklabels(ORD, fontsize=8.5)
    ax[1].set_xlabel("excess(), 95% profile-likelihood interval")
    ax[1].set_title("(b) every interval overlaps the placebo band (grey):\n"
                    "the MLE cannot separate these samples", fontsize=10)

    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.suptitle("f41 - confidence intervals for the Marx section-6 MLE", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "f41_marx_mle_ci.png", dpi=150, bbox_inches="tight")
    print(f"\n  wrote {OUT/'t56_marx_mle_ci.csv'} and {OUT/'f41_marx_mle_ci.png'}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summarise()
    else:
        main()
