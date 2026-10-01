"""
marx_mle.py -- Marx (2018) section-6 MLE, translated from the author's R code.

SOURCE
  Benjamin M. Marx, "Dynamic Bunching Estimation with Panel Data", MPRA 88647.
  Author's R file: dynamic_bunching_estimation_R.R (linked from his research page).
  Everything below is a line-by-line translation of poly_upper / poly_lower /
  laplace_* / censored / fobserved / excess. Deviations are marked  >> OURS <<.

THE MODEL, IN ONE SCREEN
  For each observation: r = log(base-year level), g = log growth to the next year.
      tonotch = log(notch) - r      growth that lands exactly on the notch
      lb      = rmin      - r       growth that lands on the bottom of the support
  Latent growth has a FLEXIBLE LAPLACE distribution conditional on r:
      g >= theta :  F*(g) = 1 - exp(Pu(g,r))
      g <  theta :  F*(g) =     exp(Pl(g,r))
  with Pu and Pl each a sum of {constant, linear, exponential, Gaussian, arctangent}
  terms whose coefficients are themselves quadratics in (r - rmin). Pl is built so
  that F* is continuous at theta by construction.

  The OBSERVED distribution differs from F* in three ways:
    1. bottom censoring: everything at or below lb is a point mass (this is where
       attritors are placed -- the fill-in value);
    2. a point mass AT the notch, fed by the share `bunching` of those whose latent
       growth would have put them in [tonotch - bunchrangewidth, tonotch);
    3. a point mass at tonotch + missingrangewidth, the top of the omitted region,
       plus extra attrition for those who would have crossed.
  `missing` and `bunching` are mapped to [0,1] by 0.5*(sin(.)+1), so the optimiser
  is unconstrained.

  ESTIMAND: excess() = mean over observations of the bunching point mass.

>> OURS <<  what had to change, and why
  * notch = log(25,000).
  * rmin is the bottom of the OBSERVED support, and it MUST be strictly below the
    notch or the model degenerates: tonotch and lb coincide, and the bunching point
    mass is not separable from the censoring point mass. For the federal
    threshold-bound panel rmin IS the notch, so the MLE cannot be run on it at all
    -- the same wall as node 4. It runs on CA+WA (floor 10,000) and on the federal
    always-covered placebo (no floor).
  * theta = 0 throughout, as the paper states for both of his applications. (His
    code reads per-year theta from files that are not distributed.)
  * quadratic-in-r terms can be switched off (`full=False`) to fit 23 parameters
    instead of 43; with our sample sizes the full model is not identified.
"""
import numpy as np
from scipy.optimize import minimize

# ----------------------------------------------------------------- primitives
class Spec:
    def __init__(self, notch, rmin, bunchrangewidth=np.log(100 / 95),
                 missingrangewidth=np.log(130 / 100), full=False):
        self.notch = notch
        self.rmin = rmin
        self.bw = bunchrangewidth
        self.mw = missingrangewidth
        self.full = full            # include the (r-rmin)^2 terms
        if rmin >= notch:
            raise ValueError("rmin must be strictly below the notch; see the docstring")


def _sq(x, on):
    return x ** 2 if on else 0.0


# ---- Pu: 18 coefficients (R paramup[1..18] -> au[0..17])
def poly_upper(g, r, au, th, sp):
    x = r - sp.rmin; z = g - th; s = _sq(x, sp.full)
    return ((au[0] + au[1] * x + au[12] * s)
            + (au[2] + au[3] * x + au[13] * s) * z
            + (au[4] + au[5] * x + au[14] * s) * (np.exp(-z) - 1)
            + (au[6] + au[7] * x + au[15] * s) * (np.exp(-z ** 2) - 1)
            + (au[8] + au[9] * x + au[16] * s)
            * np.arctan((au[10] + au[11] * x + au[17] * s) * z))


def dpoly_upper(g, r, au, th, sp):
    x = r - sp.rmin; z = g - th; s = _sq(x, sp.full)
    a = au[10] + au[11] * x + au[17] * s
    return ((au[2] + au[3] * x + au[13] * s)
            + (au[4] + au[5] * x + au[14] * s) * (-np.exp(-z))
            + (au[6] + au[7] * x + au[15] * s) * (-2 * z) * np.exp(-z ** 2)
            + (au[8] + au[9] * x + au[16] * s) * a / (1 + (a * z) ** 2))


# ---- Pl: 15 coefficients (R paramlow[1..15] -> al[0..14]); first term ties Pl to Pu
#      at theta so that F* is continuous there.
def poly_lower(g, r, al, au, th, sp):
    x = r - sp.rmin; z = g - th; s = _sq(x, sp.full)
    base = au[0] + au[1] * x + au[12] * s
    return (np.log1p(-np.exp(np.minimum(base, -1e-12)))
            + (al[0] + al[1] * x + al[10] * s) * z
            + (al[2] + al[3] * x + al[11] * s) * (np.exp(z) - 1)
            + (al[4] + al[5] * x + al[12] * s) * (np.exp(-z ** 2) - 1)
            + (al[6] + al[7] * x + al[13] * s)
            * np.arctan((al[8] + al[9] * x + al[14] * s) * z))


def dpoly_lower(g, r, al, th, sp):
    x = r - sp.rmin; z = g - th; s = _sq(x, sp.full)
    a = al[8] + al[9] * x + al[14] * s
    return ((al[0] + al[1] * x + al[10] * s)
            + (al[2] + al[3] * x + al[11] * s) * np.exp(z)
            + (al[4] + al[5] * x + al[12] * s) * (-2 * z) * np.exp(-z ** 2)
            + (al[6] + al[7] * x + al[13] * s) * a / (1 + (a * z) ** 2))


# ---- the flexible Laplace itself
def cdfstar(g, r, al, au, th, sp):
    out = np.empty_like(g, dtype=float)
    lo = g < th; hi = ~lo
    if lo.any():
        out[lo] = np.exp(poly_lower(g[lo], r[lo], al, au, th[lo] if np.ndim(th) else th, sp))
    if hi.any():
        out[hi] = 1 - np.exp(poly_upper(g[hi], r[hi], au, th[hi] if np.ndim(th) else th, sp))
    return out


def fstar(g, r, al, au, th, sp):
    out = np.zeros_like(g, dtype=float)
    lo = g < th; hi = ~lo
    if lo.any():
        t = th[lo] if np.ndim(th) else th
        out[lo] = dpoly_lower(g[lo], r[lo], al, t, sp) * np.exp(
            poly_lower(g[lo], r[lo], al, au, t, sp))
    if hi.any():
        t = th[hi] if np.ndim(th) else th
        out[hi] = -dpoly_upper(g[hi], r[hi], au, t, sp) * np.exp(
            poly_upper(g[hi], r[hi], au, t, sp))
    return out


def share(p):
    """R's 0.5*(sin(p)+1): an unconstrained real mapped into [0,1]."""
    return 0.5 * (np.sin(p) + 1.0)


# --------------------------------------------------------------- observed law
# Direct translation of censored() and fobserved(). `r < notch` and `r >= notch`
# are kept as separate branches exactly as in the R, because the parameters that
# govern attrition differ between them.
def _miss_below(m, x, sp):
    return m[0] + m[3] * x + m[5] * _sq(x, sp.full)


def _miss_above(m, x, sp):
    return m[0] + m[4] * x


def censored(lb, tonotch, r, al, au, m, th, sp):
    """Probability mass at or below the bottom-censoring point (attritors live here)."""
    out = np.zeros_like(r, dtype=float)
    x = r - sp.rmin
    below = r < sp.notch
    above = ~below
    if below.any():
        mb = _miss_below(m, x[below], sp)
        F_lb = cdfstar(lb[below], r[below], al, au, th[below], sp)
        F_tn = cdfstar(tonotch[below], r[below], al, au, th[below], sp)
        F_tm = cdfstar(tonotch[below] + sp.mw, r[below], al, au, th[below], sp)
        out[below] = (mb + m[1]
                      + (1 - mb - m[1]) * F_lb
                      + share(m[2]) * (F_tm - F_tn)
                      + share(m[6]) * (1 - F_tm))
    if above.any():
        ma = _miss_above(m, x[above], sp)
        F_lb = cdfstar(lb[above], r[above], al, au, th[above], sp)
        out[above] = ma + (1 - ma) * F_lb
    return out


def fobserved(lb, tonotch, g, r, al, au, m, b, th, sp):
    n = len(g)
    x = r - sp.rmin
    fo = np.zeros(n)
    below = r < sp.notch
    above = ~below

    mb = np.zeros(n); ma = np.zeros(n)
    mb[below] = _miss_below(m, x[below], sp)
    ma[above] = _miss_above(m, x[above], sp)
    keep = np.where(below, 1 - mb - m[1], 1 - ma)          # mass left for the latent law

    F_tn = cdfstar(tonotch, r, al, au, th, sp)
    F_tm = cdfstar(tonotch + sp.mw, r, al, au, th, sp)
    F_bw = cdfstar(tonotch - sp.bw, r, al, au, th, sp)

    bshare = np.where(below, share(b[0] + b[2] * x), share(b[1]))
    # mass available to be pulled into the notch
    avail = np.where(below, 1 - mb - m[1] - share(m[2]), 1 - ma)
    bmass = bshare * avail * (F_tm - F_tn)

    # 1 - bottom-censored (includes every attritor)
    at_lb = g <= lb + 1e-12
    if at_lb.any():
        fo[at_lb] = censored(lb[at_lb], tonotch[at_lb], r[at_lb], al, au, m,
                             th[at_lb], sp)
    # 2 - interior, strictly below the bunching region
    mid = (~at_lb) & (g < tonotch - sp.bw)
    if mid.any():
        fo[mid] = keep[mid] * fstar(g[mid], r[mid], al, au, th[mid], sp)
    # 3 - the notch point mass
    at_n = (~at_lb) & (g >= tonotch - sp.bw) & (g <= tonotch + 1e-12)
    if at_n.any():
        fo[at_n] = bmass[at_n] + keep[at_n] * (F_tn[at_n] - F_bw[at_n])
    # 4 - the top of the omitted region
    at_m = (g > tonotch + 1e-12) & (g <= tonotch + sp.mw + 1e-12)
    if at_m.any():
        fo[at_m] = (1 - bshare[at_m]) * avail[at_m] * (F_tm[at_m] - F_tn[at_m])
    # 5 - above the omitted region
    hi = g > tonotch + sp.mw + 1e-12
    if hi.any():
        extra = np.where(below[hi], share(m[6]), 0.0)
        fo[hi] = (keep[hi] - extra) * fstar(g[hi], r[hi], al, au, th[hi], sp)
    return fo


def split(p):
    return p[0:3], p[3:18], p[18:36], p[36:43]


def make_neglogL(g, r, lb, tonotch, th, sp):
    def nll(p):
        b, al, au, m = split(p)
        with np.errstate(all="ignore"):
            f = fobserved(lb, tonotch, g, r, al, au, m, b, th, sp)
        f = np.where(np.isfinite(f) & (f > 0), f, 1e-300)
        return -np.sum(np.log(f))
    return nll


def excess(p, r, tonotch, th, sp):
    """Marx's excess(): the mean bunching point mass. This is the estimand."""
    b, al, au, m = split(p)
    x = r - sp.rmin
    below = r < sp.notch
    mb = np.where(below, _miss_below(m, x, sp), 0.0)
    ma = np.where(below, 0.0, _miss_above(m, x, sp))
    F_tn = cdfstar(tonotch, r, al, au, th, sp)
    F_tm = cdfstar(tonotch + sp.mw, r, al, au, th, sp)
    bshare = np.where(below, share(b[0] + b[2] * x), share(b[1]))
    avail = np.where(below, 1 - mb - m[1] - share(m[2]), 1 - ma)
    return float(np.nanmean(bshare * avail * (F_tm - F_tn)))


# ------------------------------------------------------------------- fitting
def prepare(level_t, level_t1, sp):
    """Build r, g and the censoring/notch landmarks, and discretise g the way the
    R code does: bottom-censor, collapse the bunching region onto the notch, and
    collapse the omitted region onto its upper edge."""
    r = np.log(level_t)
    ok = (r >= sp.rmin) & np.isfinite(r)
    r = r[ok]
    nxt = np.asarray(level_t1, float)[ok]
    g = np.where(nxt > 0, np.log(np.where(nxt > 0, nxt, 1)) - r, np.nan)
    tonotch = sp.notch - r
    lb = sp.rmin - r
    gt = g.copy()
    gt[~np.isfinite(gt)] = lb[~np.isfinite(gt)]          # attritors -> the fill-in
    gt[gt < lb] = lb[gt < lb]
    sel = (gt >= tonotch) & (gt < tonotch + sp.mw)
    gt[sel] = (tonotch + sp.mw)[sel]
    sel = (gt >= tonotch - sp.bw) & (gt < tonotch)
    gt[sel] = tonotch[sel]
    return r, gt, lb, tonotch


def start_values(sp, bunch0=0.05, miss0=0.05):
    p = np.zeros(43)
    p[0] = np.arcsin(2 * bunch0 - 1)      # bunching below notch
    p[1] = np.arcsin(2 * bunch0 - 1)      # bunching above notch
    p[2] = 0.0
    # Pl: linear term only -> exponential lower tail
    p[3] = 1.0
    # Pu: intercept -> F*(theta); slope -> exponential upper tail
    p[18] = np.log(0.5)
    p[20] = -1.0
    p[28] = 1.0                            # arctan scale, keeps the term finite
    p[36] = miss0                          # missing[1]
    p[39] = 0.0                            # missing[4]
    p[40] = 0.0                            # missing[5]
    p[38] = np.arcsin(2 * 0.02 - 1)        # missing[3]
    p[42] = np.arcsin(2 * 0.02 - 1)        # missing[7]
    return p


# indices of the (r-rmin)^2 coefficients, inert when full=False
SQ_IDX = list(range(13, 18)) + list(range(30, 36)) + [41]


def free_mask(sp):
    m = np.ones(43, dtype=bool)
    if not sp.full:
        m[SQ_IDX] = False
    return m


def fit(level_t, level_t1, sp, p0=None, maxiter=6000, restarts=3, seed=0,
        verbose=False):
    # Nelder-Mead then Powell, on the FREE parameters only. When full=False the
    # quadratic-in-r coefficients multiply zero, so leaving them in the search would
    # hand the optimiser twelve exactly flat directions. They are held out instead.
    r, gt, lb, tonotch = prepare(level_t, level_t1, sp)
    th = np.zeros_like(r)
    nll_full = make_neglogL(gt, r, lb, tonotch, th, sp)
    mask = free_mask(sp)
    base = start_values(sp) if p0 is None else np.asarray(p0, float).copy()

    def nll(q):
        p = base.copy()
        p[mask] = q
        return nll_full(p)

    rng = np.random.default_rng(seed)
    best_q = base[mask].copy()
    best_f = nll(best_q)
    for k in range(restarts):
        q0 = best_q if k == 0 else best_q + rng.normal(0, 0.15, best_q.shape)
        r1 = minimize(nll, q0, method="Nelder-Mead",
                      options=dict(maxiter=maxiter, maxfev=maxiter * 3,
                                   xatol=1e-7, fatol=1e-7, disp=verbose))
        r2 = minimize(nll, r1.x, method="Powell",
                      options=dict(maxiter=maxiter, maxfev=maxiter * 6))
        for rr in (r1, r2):
            if np.isfinite(rr.fun) and rr.fun < best_f:
                best_f, best_q = float(rr.fun), np.asarray(rr.x).copy()

    p = base.copy()
    p[mask] = best_q
    return dict(params=p, nll=best_f, n=len(r), n_free=int(mask.sum()),
                excess=excess(p, r, tonotch, th, sp),
                bunch_below=float(share(p[0])),
                bunch_above=float(share(p[1])),
                miss_instead_of_crossing=float(share(p[38])),
                attrition_at_rmin=float(p[36]),
                r=r, gt=gt, lb=lb, tonotch=tonotch)


NEAR_ZERO = np.arcsin(2 * 1e-6 - 1)          # the share transform evaluated at ~0


def fit_staged(level_t, level_t1, sp, maxiter=6000, restarts=2, seed=0):
    """Two-stage fit.

    Optimising all 31 free parameters from a cold start lands in a local optimum where
    the latent distribution stretches to swallow the notch mass and the bunching share
    collapses to zero. Fitting the latent law FIRST with bunching pinned at zero, then
    releasing it, avoids that: the profile over the bunching share is well behaved once
    the latent parameters are roughly right (verified on simulated data).

      stage A   bunching pinned at ~0, latent law + attrition free
      stage B   everything free, started from stage A
    """
    # ---- stage A
    p0 = start_values(sp)
    p0[0] = p0[1] = NEAR_ZERO
    mask_a = free_mask(sp).copy()
    mask_a[[0, 1, 2]] = False                 # hold the bunching parameters
    r, gt, lb, tonotch = prepare(level_t, level_t1, sp)
    th = np.zeros_like(r)
    nll_full = make_neglogL(gt, r, lb, tonotch, th, sp)

    def _run(base, mask, q0, iters, nrestart, rng):
        def nll(q):
            p = base.copy(); p[mask] = q
            return nll_full(p)
        bq, bf = np.asarray(q0, float).copy(), nll(q0)
        for k in range(nrestart):
            s0 = bq if k == 0 else bq + rng.normal(0, 0.12, bq.shape)
            for meth, opt in (("Nelder-Mead", dict(maxiter=iters, maxfev=iters * 3,
                                                   xatol=1e-7, fatol=1e-7)),
                              ("Powell", dict(maxiter=iters, maxfev=iters * 6))):
                rr = minimize(nll, s0, method=meth, options=opt)
                if np.isfinite(rr.fun) and rr.fun < bf:
                    bf, bq = float(rr.fun), np.asarray(rr.x).copy()
                s0 = bq
        p = base.copy(); p[mask] = bq
        return p, bf

    rng = np.random.default_rng(seed)
    pA, fA = _run(p0, mask_a, p0[mask_a], maxiter, restarts, rng)

    # ---- stage B
    mask_b = free_mask(sp)
    pB, fB = _run(pA, mask_b, pA[mask_b], maxiter, restarts, rng)

    # ---- stage C: the PROPER restricted fit for the likelihood-ratio test.
    # Comparing stage B against stage A would conflate "bunching helps" with "the
    # optimiser simply had another pass at everything else". Stage C starts from the
    # unrestricted solution and re-optimises with the bunching shares pinned at zero,
    # so the only thing that differs is the restriction.
    pC0 = pB.copy(); pC0[0] = pC0[1] = NEAR_ZERO; pC0[2] = 0.0
    mask_c = mask_b.copy(); mask_c[[0, 1, 2]] = False
    pC, fC = _run(pC0, mask_c, pC0[mask_c], maxiter, restarts, rng)

    # ---- stage D: the restricted fit got an extra optimisation pass, so if it beat
    # the unrestricted one (a negative LR, which is impossible at a true optimum) the
    # unrestricted fit was simply under-converged. Give it the same extra pass from
    # the restricted solution and keep whichever is better. Iterate until the LR is
    # non-negative or the budget runs out.
    for _ in range(3):
        if fB <= fC:
            break
        pD, fD = _run(pC, mask_b, pC[mask_b], maxiter, 1, rng)
        if fD < fB:
            pB, fB = pD, fD
        pC0 = pB.copy(); pC0[0] = pC0[1] = NEAR_ZERO; pC0[2] = 0.0
        pC, fC = _run(pC0, mask_c, pC0[mask_c], maxiter, 1, rng)

    return dict(params=pB, nll=fB, nll_stageA=fA, nll_restricted=fC,
                n=len(r), n_free=int(mask_b.sum()),
                excess=excess(pB, r, tonotch, th, sp),
                bunch_below=float(share(pB[0])),
                bunch_above=float(share(pB[1])),
                miss_instead_of_crossing=float(share(pB[38])),
                attrition_at_rmin=float(pB[36]),
                lr_vs_no_bunching=2 * (fC - fB),
                r=r, gt=gt, lb=lb, tonotch=tonotch)
