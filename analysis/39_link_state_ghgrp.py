"""
39 · Matching the CA/WA state panels back to GHGRP FacilityId          (John ① pending)
================================================================================
The gap Elaine flagged in the 0928 note:

    "The reweighting was done on the federal panel (covariates on one definition). The bunching
     estimate runs on the state panel, which carries no covariates at all -- that number has never
     been adjusted. Adjusting it requires matching the state facilities back to the GHGRP panel."

That is exactly right and it is the one thing John ① asked for that is still open. The
federal panel carries methane share, subpart count, NAICS -- everything the propensity
reweighting in working note 07 used. The CA MRR panel carries four columns (arb_id, name,
co2e, year) and the WA panel not many more. So the headline bunching estimate (0.863)
has never been composition-adjusted.

THE MATCHING SIGNAL THAT MAKES THIS EASY
  Above 25,000 a facility appears in BOTH datasets, and both report ITS OWN annual
  emissions. So a CA facility-year and a GHGRP facility-year in California that agree
  on the emissions number to within a few percent, in the same year, are almost
  certainly the same plant. Name similarity is a second, independent signal. Requiring
  both is far stronger than either alone -- and unlike the EIA match (script 31) there
  is no co-location problem, because two different plants do not emit the same amount.

  ⚠️ This works only ABOVE the federal threshold. Below 25,000 the facility is not in
  the GHGRP at all, so the part of the state panel that carries the bunching -- the mass
  just BELOW the line -- is unmatchable by construction. Section 4 says what that means.

outputs  output/link_state_ghgrp.csv  output/t67_state_match.csv
run      python3 39_link_state_ghgrp.py            (from analysis/)
"""
import re, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from ghgrp_load import load_cached

OUT = Path("output")
STOP = {"llc", "inc", "lp", "co", "company", "corp", "corporation", "ltd", "plant",
        "the", "and", "of", "l", "p", "lc", "partners", "holdings", "us", "usa",
        "facility", "station", "works", "operations", "energy", "power"}


def toks(s):
    w = re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split()
    return {t for t in w if t not in STOP and len(t) > 1}


def name_score(a, b):
    ta, tb = toks(a), toks(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


# ---------------------------------------------------------------- 1 · inputs
de = load_cached()
de = de[de.emissions > 0]
ca = pd.read_csv(OUT / "ca_mrr_panel.csv")
ca["co2e"] = pd.to_numeric(ca.co2e, errors="coerce")
ca = ca.dropna(subset=["co2e"])
ca = ca[ca.co2e > 0].rename(columns={"arb_id": "sid", "name": "sname"})
ca["state"] = "CA"

wa = pd.read_csv(OUT / "wa_ghg_panel.csv")
wa = wa[wa["Reporter Type"] == "Facility"].copy()
wa["co2e"] = pd.to_numeric(wa.co2e, errors="coerce")
wa = wa.dropna(subset=["co2e"])
wa = wa[wa.co2e > 0].rename(columns={"Reporter": "sname", "Year": "syear"})
wa["sid"] = wa.sname
wa["state"] = "WA"
wa["year"] = pd.to_numeric(wa.syear, errors="coerce")
S = pd.concat([ca[["sid", "sname", "co2e", "year", "state"]],
               wa[["sid", "sname", "co2e", "year", "state"]]], ignore_index=True)
S["year"] = S.year.astype(int)
print(f"state panel: {len(S):,} facility-years "
      f"({S[S.state=='CA'].sid.nunique():,} CA + {S[S.state=='WA'].sid.nunique():,} WA facilities)")

G = de[de.state.isin(["CA", "WA"])][["FacilityId", "facility_name", "state", "year",
                                     "emissions"]].copy()
print(f"GHGRP CA+WA:  {len(G):,} facility-years ({G.FacilityId.nunique():,} facilities)")

# ---------------------------------------------------------------- 2 · pair scoring
# ⚠️ FIRST ATTEMPT WAS BACKWARDS. Matching on emissions agreement and using the name as
# support produced 136,651 candidate pairs from 7,184 GHGRP facility-years: with several
# hundred facilities spread over orders of magnitude, coincidental agreement is common.
# At |log ratio| < 0.001 the list still contained pairs like
#     "ACE Cogeneration"           -> "CalPortland Company Oro Grande Plant"
#     "Aera Energy Coastal Basins" -> "Greenleaf Energy Unit 2 LLC"
# while the genuine ones sit at 5e-5 with obviously matching names ("AES Alamitos, LLC"
# -> "AES Alamitos"). So the NAME is the primary signal and emissions agreement is the
# confirmation, not the other way round.
def best_name(sn, gnames, gids):
    sc = [name_score(sn, g) for g in gnames]
    if not sc:
        return None, None, 0.0, 0.0
    o = np.argsort(sc)[::-1]
    second = sc[o[1]] if len(o) > 1 else 0.0
    return gids[o[0]], gnames[o[0]], sc[o[0]], second


rows = []
for st in ("CA", "WA"):
    s = S[S.state == st].drop_duplicates("sid")
    g = G[G.state == st].drop_duplicates("FacilityId")
    gn, gi = list(g.facility_name), list(g.FacilityId)
    for _, r in s.iterrows():
        fid, gname, sc, sc2 = best_name(r.sname, gn, gi)
        rows.append(dict(state=st, sid=r.sid, sname=r.sname, FacilityId=fid,
                         gname=gname, name_score=sc, name_margin=sc - sc2))
L = pd.DataFrame(rows)
print(f"\nname-first candidates: {len(L):,} state facilities scored")

# confirmation: do the two series agree on emissions in the years they overlap?
sy = S.set_index(["state", "sid", "year"]).co2e
gy = G.set_index(["FacilityId", "year"]).emissions
conf = []
for _, r in L.iterrows():
    if pd.isna(r.FacilityId):
        conf.append((0, np.nan)); continue
    a = S[(S.state == r.state) & (S.sid == r.sid)][["year", "co2e"]]
    b = G[G.FacilityId == r.FacilityId][["year", "emissions"]]
    m = a.merge(b, on="year")
    if not len(m):
        conf.append((0, np.nan)); continue
    lr = np.abs(np.log(m.co2e / m.emissions))
    conf.append((len(m), float(np.median(lr))))
L["n_overlap"] = [c[0] for c in conf]
L["med_logratio"] = [c[1] for c in conf]

L["accept"] = ((L.name_score >= 0.50) |
               ((L.name_score >= 0.25) & (L.med_logratio < 0.02)) |
               ((L.name_score >= 0.15) & (L.med_logratio < 0.002)))
L = L.sort_values(["name_score", "name_margin"], ascending=False)
L.loc[L.accept, "accept"] = ~L[L.accept].duplicated("FacilityId")   # keep 1-to-1
A = L[L.accept]
print(f"accepted 1-to-1 links: {len(A):,} "
      f"(CA {int((A.state=='CA').sum())}, WA {int((A.state=='WA').sum())})")
print(f"  median name score {A.name_score.median():.2f}, "
      f"median |log emissions ratio| {A.med_logratio.median():.4f}")
print(f"  of the accepted, {int((A.med_logratio < 0.02).sum())} also agree on emissions "
      f"to within 2%")
L.to_csv(OUT / "link_state_ghgrp.csv", index=False)


# ---------------------------------------------------------------- 3 · what it covers
print("\n" + "=" * 78)
print("3 . what fraction of the state panel is now matched?")
print("=" * 78)
SM = S.merge(A[["state", "sid", "FacilityId"]], on=["state", "sid"], how="left")
SM["matched"] = SM.FacilityId.notna()
rows = []
for lab, sub in [("all state facility-years", SM),
                 ("in the bunching fit range [11k, 45k]",
                  SM[SM.co2e.between(11_000, 45_000)]),
                 ("ABOVE the line [25k, 45k]", SM[SM.co2e.between(25_000, 45_000)]),
                 ("BELOW the line [20k, 25k)  <- the excess mass",
                  SM[SM.co2e.between(20_000, 25_000, inclusive="left")])]:
    rows.append(dict(sample=lab, n=len(sub), matched=int(sub.matched.sum()),
                     pct=round(sub.matched.mean() * 100, 1),
                     pct_emis=round(sub.loc[sub.matched, "co2e"].sum()
                                    / max(sub.co2e.sum(), 1) * 100, 1)))
T = pd.DataFrame(rows)
T.to_csv(OUT / "t67_state_match.csv", index=False)
print(T.to_string(index=False))


# ---------------------------------------------------------------- 4 · adjust
print("\n" + "=" * 78)
print("4 . now the thing this was for: is the bunching estimate composition-sensitive?")
print("=" * 78)
COV = ["ch4_share", "n_subparts", "multi_subpart", "log_em"]
d = de.copy()
d["ch4_share"] = (pd.to_numeric(d["Methane (CH4) emissions"], errors="coerce")
                  / d.emissions).clip(0, 1).fillna(0)
d["multi_subpart"] = (d.n_subparts > 1).astype(float)
FC = (d.sort_values(["FacilityId", "year"]).groupby("FacilityId")[COV].mean()
        .reset_index())
SM2 = SM.merge(FC, on="FacilityId", how="left")
nat = d[d.threshold_bound].groupby("FacilityId")[COV].mean()

print("\n  normalised differences, MATCHED state facilities vs national threshold-bound")
print("  (|ND| > 0.25 is the usual 'unbalanced' flag)")
mm = SM2[SM2.matched].drop_duplicates("FacilityId")
nd_rows = []
for c in COV:
    a, b = mm[c].dropna(), nat[c].dropna()
    nd = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
    nd_rows.append(dict(covariate=c, state_mean=round(a.mean(), 3),
                        national_mean=round(b.mean(), 3), ND=round(nd, 3),
                        unbalanced=abs(nd) > .25))
    print(f"    {c:<16s} state {a.mean():8.3f}   national {b.mean():8.3f}   "
          f"ND {nd:+.3f} {'⚠️' if abs(nd) > .25 else ''}")
pd.DataFrame(nd_rows).to_csv(OUT / "t68_state_balance.csv", index=False)

# --- propensity weights toward the national composition, on the matched facilities
X = mm[COV].dropna()
ids = mm.loc[X.index, "FacilityId"]
Z = pd.concat([X.assign(state_side=1),
               nat.dropna().assign(state_side=0)], ignore_index=True)
Xm = sm_add = np.column_stack([np.ones(len(Z))] + [Z[c].values for c in COV])
y = Z.state_side.values.astype(float)
beta = np.zeros(Xm.shape[1])
for _ in range(60):                       # plain IRLS logit, as in script 21
    p = 1 / (1 + np.exp(-Xm @ beta))
    W = np.clip(p * (1 - p), 1e-6, None)
    z = Xm @ beta + (y - p) / W
    beta = np.linalg.solve(Xm.T @ (Xm * W[:, None]), Xm.T @ (W * z))
ps = 1 / (1 + np.exp(-(Xm[:len(X)] @ beta)))
w = np.clip((1 - ps) / np.clip(ps, 1e-3, None), 0, 20)
WT = pd.DataFrame({"FacilityId": ids.values, "wt": w})
print(f"\n  propensity weights built on {len(WT):,} matched facilities "
      f"(median {np.median(w):.2f}, p95 {np.percentile(w, 95):.2f})")


def bunch_w(x, wts, cut=25_000, bin_w=1000, fit_hw=10_000, excl_lo=5000, order=4,
            B=300, seed=5):
    """The f30 estimator, with optional facility weights."""
    edges = np.arange(cut - fit_hw, cut + fit_hw + bin_w, bin_w)
    keep = (x >= cut - fit_hw) & (x < cut + fit_hw)
    c = np.histogram(x[keep], bins=edges, weights=wts[keep])[0].astype(float)
    dd = edges[:-1] + bin_w / 2 - cut
    z = dd / 1000.0
    in_excl = (dd >= -excl_lo) & (dd < 0)
    if c.sum() < 60 or (~in_excl).sum() < order + 4:
        return None
    P_ = np.vander(z, order + 1)
    Xd = np.hstack([P_, np.eye(len(z))[:, in_excl]])

    def stat(cc):
        b, *_ = np.linalg.lstsq(Xd, cc, rcond=None)
        cf = (P_ @ b[: order + 1])[in_excl]
        return (cc[in_excl] - cf).sum() / max(cf.mean(), 1e-9)

    bb, *_ = np.linalg.lstsq(Xd, c, rcond=None)
    res = c - Xd @ bb
    rng = np.random.default_rng(seed)
    dr = [stat(np.maximum(c + rng.choice(res, len(c), replace=True), 0)) for _ in range(B)]
    lo, hi = np.percentile(dr, [2.5, 97.5])
    return dict(excess=stat(c), lo=lo, hi=hi, sig=bool(lo > 0 or hi < 0), n=int(c.sum()))


# ⚠️ Read the three rows below as a DEMONSTRATION, not as three estimates. Restricting
# to matched facilities removes most of the below-line mass (26.5% matched there against
# 61.0% above), and that mass IS the object being estimated. The sign flip in row (2) is
# the selection, not a composition effect. Row (3) is worse: 412 state facilities against
# ~5,400 national ones drives the propensity score near zero, so (1-p)/p piles up at the
# cap (median weight 12.2, p95 at the cap of 20). Neither row is a corrected estimate.
# Geometry note: this is the SYMMETRIC [15k, 35k] window with w = 5,000, not f30's
# asymmetric [11k, 45k]. That is why row (1) is +1.06 and not the published 0.863.
SM3 = SM.merge(WT, on="FacilityId", how="left")
runs = []
for lab, sub, wcol in [
        ("(1) full state panel      [symmetric window, cf. f30's 0.863]", SM3, None),
        ("(2) matched subsample     [selection, NOT a corrected estimate]",
         SM3[SM3.matched], None),
        ("(3) matched + reweighted  [degenerate weights, NOT usable]",
         SM3[SM3.matched], "wt")]:
    x = sub.co2e.values
    wts = np.ones(len(sub)) if wcol is None else sub[wcol].fillna(1.0).values
    r = bunch_w(x, wts)
    if r:
        runs.append(dict(spec=lab, **r))
        print(f"\n  {lab}")
        print(f"      excess {r['excess']:+.3f}  [{r['lo']:+.3f}, {r['hi']:+.3f}]  "
              f"{'significant' if r['sig'] else 'NOT significant'}   n(fit) = {r['n']:,}")
pd.DataFrame(runs).to_csv(OUT / "t69_state_reweighted_bunching.csv", index=False)

print("""
==============================================================================
5 . the answer to John (1)'s pending item: this adjustment CANNOT be made, and
    the reason is structural rather than a matching failure
==============================================================================
  The covariates live in the federal panel. A state facility BELOW 25,000 is not in the
  federal panel -- that is what the threshold does. So the facilities carrying the excess
  mass are exactly the ones for which no covariates can ever exist. The 26.5% of the
  below-line band that did match are facilities that cross the line in OTHER years, i.e.
  the crossers -- a selected subset of the below-line mass, not a sample of it.

  Rows (2) and (3) above show what happens if you try anyway: the estimate flips sign
  because the restriction deletes the mass being measured. That is a demonstration of
  the obstacle, not a corrected number. DO NOT quote -1.148 or -0.874 anywhere.

  WHAT THE MATCH IS GOOD FOR, and it is not nothing:
    - a balance check on the part that CAN be observed. Above the line, the matched
      California and Washington facilities look like the national threshold-bound
      sample on every covariate tested: all four |ND| < 0.25, the largest being
      multi-subpart at -0.225. That is not proof about the unmatched part, but it is
      the only evidence available, and it points the reassuring way.
    - covariates for any state-panel analysis restricted to above-threshold facilities.

  HOW TO WRITE IT: "The state-panel bunching estimate is not composition-adjusted,
  because adjustment would require covariates for facilities below the federal
  reporting threshold, which by construction do not appear in the federal data. Above
  the threshold, where both panels observe the same facilities, the state sample is
  balanced against the national threshold-bound sample on emissions level, methane
  share and subpart count (all |ND| < 0.25)."
""")
print(f"wrote link_state_ghgrp.csv, t67, t68, t69")
