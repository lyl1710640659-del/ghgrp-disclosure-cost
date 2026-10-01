"""
Formal manipulation test at the 25,000 tCO2e cutoff, state panels.

Replaces my hand-rolled log-density scan with the Cattaneo-Jansson-Ma (2020, JASA)
local-polynomial density estimator (`rddensity`): data-driven bandwidths, bias
correction, jackknife SEs. This is the standard estimator in the RD literature.

Why the state panels and not the federal one: the federal panel is truncated at
25,000 (facilities below the cutoff are simply absent), so a density test there is
meaningless. CA and WA require reporting from 10,000, so both sides of the federal
cutoff are fully observed.

Two specifications are reported because they disagree in the small WA sample:
  unrestricted  - separate polynomials each side (rddensity default)
  restricted    - common density imposed, more stable in small samples

Outputs
  output/t10_rddensity.csv   full table
  output/f20_rddensity.png   density plots with the test result annotated
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rddensity import rddensity

OUT = "output"
LO, HI = 5_000, 100_000
CUTS = [25_000, 15_000, 20_000, 30_000, 35_000, 40_000]


def prep(s):
    x = pd.to_numeric(s, errors="coerce").dropna()
    return x[(x >= LO) & (x <= HI)].values


ca = pd.read_csv(f"{OUT}/ca_mrr_panel.csv")
wa = pd.read_csv(f"{OUT}/wa_ghg_panel.csv")
wa = wa[wa["Reporter Type"] == "Facility"]

S = {"CA": prep(ca.co2e), "WA": prep(wa.co2e)}
S["CA+WA"] = np.concatenate([S["CA"], S["WA"]])

rows = []
for name, x in S.items():
    for c in CUTS:
        for spec in ["unrestricted", "restricted"]:
            r = rddensity(X=x, c=c, fitselect=spec)
            fl, fr = r.hat["left"], r.hat["right"]
            se = r.sd_jk["diff"]
            rows.append(dict(
                sample=name, cutoff=c, spec=spec, n=len(x),
                n_left=int(((x >= c - 5000) & (x < c)).sum()),
                n_right=int(((x >= c) & (x < c + 5000)).sum()),
                h_left=r.h["left"], h_right=r.h["right"],
                f_left=fl, f_right=fr,
                rel_jump=(fr - fl) / fl if fl > 0 else np.nan,
                t=r.test["t_jk"], p=r.test["p_jk"],
                sig=r.test["p_jk"] < 0.05,
                mde_rel=2.8 * se / fl if fl > 0 else np.nan,   # 80% power, 5% level
                bad_fit=fl <= 0 or fr <= 0,                    # negative density = unusable
            ))

T = pd.DataFrame(rows)
T.to_csv(f"{OUT}/t10_rddensity.csv", index=False)

pd.set_option("display.width", 220)
show = ["sample", "cutoff", "spec", "n_left", "n_right", "rel_jump", "t", "p", "sig",
        "mde_rel", "bad_fit"]
print(T[show].to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
grid = np.arange(10_000, 46_000, 1_000)
for ax, (name, x) in zip(axes, S.items()):
    ax.hist(x[(x >= grid[0]) & (x < grid[-1])], bins=grid,
            color="#b8c4d4", edgecolor="white")
    ax.axvline(25_000, color="#c0392b", lw=2)
    u = T[(T["sample"] == name) & (T.cutoff == 25_000) & (T.spec == "unrestricted")].iloc[0]
    q = T[(T["sample"] == name) & (T.cutoff == 25_000) & (T.spec == "restricted")].iloc[0]
    note = f"unrestricted  t={u.t:+.2f}, p={u.p:.2f}"
    if u.bad_fit:
        note += "  (negative density — unusable)"
    note += f"\nrestricted    t={q.t:+.2f}, p={q.p:.2f}   MDE {q.mde_rel*100:.0f}%"
    ax.set_title(f"{name}   n={len(x):,} in [5k,100k]\n{note}", fontsize=9.5)
    ax.set_xlabel("reported emissions (tCO$_2$e)")
    ax.set_ylabel("facility-years")
fig.suptitle("Formal manipulation test (Cattaneo–Jansson–Ma) at 25,000.  The default "
             "specification cannot reject; the restricted one rejects here but also at "
             "clean placebos (35,000), so its rejections are not informative.",
             fontsize=10.5)
fig.tight_layout()
fig.savefig(f"{OUT}/f20_rddensity.png", dpi=150)
print("\nwrote t10_rddensity.csv, f20_rddensity.png")
