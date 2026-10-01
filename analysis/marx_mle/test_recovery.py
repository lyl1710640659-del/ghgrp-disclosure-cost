"""Simulation recovery test: does this implementation get back a KNOWN bunching share?

This is the check that makes the translation trustworthy. If it cannot recover a
share it planted itself, nothing it says about the GHGRP panel means anything.
"""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from marx_mle import Spec, fit, fit_staged, prepare

RMIN, NOTCH = np.log(10_000), np.log(25_000)
TRUE_BUNCH = float(sys.argv[1]) if len(sys.argv) > 1 else 0.20
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
TRUE_ATTR = 0.05
SCALE = 0.40

sp = Spec(NOTCH, RMIN, full=False)
rng = np.random.default_rng(7)

# r is drawn over a band that actually straddles the notch on BOTH sides at
# a distance the growth distribution can cover. Drawn too wide, no facility
# far above the notch ever falls to it, and the above-notch bunching
# parameter is simply not identified -- which is itself worth knowing.
r = rng.uniform(RMIN, RMIN + 1.6, N)
g = rng.laplace(0.0, SCALE, N)                 # latent growth
tonotch = NOTCH - r

# bunching: those who would land in the omitted region [notch, 1.3*notch) move to the notch
inmiss = (g >= tonotch) & (g < tonotch + sp.mw)
move = inmiss & (rng.random(N) < TRUE_BUNCH)
# a buncher lands just BELOW the notch, inside [tonotch-bw, tonotch)
g[move] = tonotch[move] - sp.bw / 2

lvl_t = np.exp(r)
lvl_t1 = np.exp(r + g)
attr = rng.random(N) < TRUE_ATTR
lvl_t1[attr] = np.nan                           # attritors

print(f"simulated: N={N:,}  true bunching share={TRUE_BUNCH:.3f}  "
      f"true attrition={TRUE_ATTR:.3f}")
print(f"  facilities whose latent landing is in the omitted region: {inmiss.sum():,}")
print(f"  of which moved to the notch: {move.sum():,}")
_, gt, lb, tn = prepare(lvl_t, lvl_t1, sp)
print(f"  observed at the notch point: {(np.abs(gt - tn) < 1e-9).sum():,}"
      f"   bottom-censored: {(gt <= lb + 1e-12).sum():,}")

t0 = time.time()
out = fit_staged(lvl_t, lvl_t1, sp, restarts=int(sys.argv[3]) if len(sys.argv)>3 else 2,
                 maxiter=int(sys.argv[4]) if len(sys.argv)>4 else 5000)
print(f"\nfitted in {time.time()-t0:.0f}s   -logL = {out['nll']:,.1f}   "
      f"free params = {out['n_free']}")
print(f"  bunching share, r below notch : {out['bunch_below']:.3f}   (true {TRUE_BUNCH:.3f})")
print(f"  bunching share, r above notch : {out['bunch_above']:.3f}   (true {TRUE_BUNCH:.3f})")
print(f"  attrition intercept           : {out['attrition_at_rmin']:.3f}   (true {TRUE_ATTR:.3f})")
print(f"  excess (mean notch point mass): {out['excess']:.4f}")
print(f"  LR vs no-bunching (stage A)   : {out['lr_vs_no_bunching']:.1f}  (chi2_1 5% = 3.84)")
print(f"  raw share sitting at the notch: {(np.abs(out['gt']-out['tonotch'])<1e-9).mean():.4f}")
