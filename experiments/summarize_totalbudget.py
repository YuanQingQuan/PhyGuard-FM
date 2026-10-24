from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

TH = 0.05
D = ROOT / "results" / "crb_cutoff_totalbudget"


def crossing(logx, y, thresh=TH):
    order = np.argsort(logx)
    x, y = logx[order], np.log10(np.maximum(y[order], 1e-12))
    lt = np.log10(thresh)
    for i in range(len(x) - 1):
        if (y[i] - lt) * (y[i + 1] - lt) <= 0 and y[i] != y[i + 1]:
            return float(x[i] + (lt - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    return None


def permanent_crossing(logx, y, thresh=TH):

    order = np.argsort(logx)
    x, y = logx[order], np.asarray(y[order], dtype=float)
    lt = np.log10(thresh)
    ly = np.log10(np.maximum(y, 1e-12))
    above = np.where(ly > lt)[0]
    if len(above) == 0:
        return None
    i = above[-1]
    if i == len(x) - 1:
        return float("inf")
    if ly[i] == ly[i + 1]:
        return float(x[i])
    return float(x[i] + (lt - ly[i]) * (x[i + 1] - x[i]) / (ly[i + 1] - ly[i]))


def main():
    fis = pd.read_csv(D / "fisher_full.csv")
    emp = pd.read_csv(D / "empirical.csv")
    low = D / "empirical_lowend.csv"
    if low.exists():
        emp = pd.concat([emp, pd.read_csv(low)], ignore_index=True)
        emp = emp.sort_values("total_photons")
        emp.to_csv(D / "empirical_all.csv", index=False)


    rows = []
    for (s, p, m), g in fis.groupby(["structure", "protocol", "mode"]):
        g = g.dropna(subset=["sqrt_crb"]).sort_values("total_photons")
        cx = crossing(np.log10(g["total_photons"].values), g["sqrt_crb"].values)
        rows.append({"structure": s, "protocol": p, "mode": m,
                     "photons_needed": np.nan if cx is None else 10 ** cx})
    need = pd.DataFrame(rows)
    need.to_csv(ROOT / "results" / "paper_figures" / "photons_needed_total.csv", index=False)
    print("=== theory: total photons for sqrt(CRB)=0.05 ===")
    print(need.pivot_table(index=["structure", "mode"], columns="protocol",
                           values="photons_needed").round(3))


    rng = np.random.default_rng(0)
    print("\n=== empirical crossings (median [95% CI]) vs CRB ===")
    out = []
    for (s, p, m), g in emp.groupby(["structure", "protocol", "mode"]):
        g = g.sort_values("total_photons")
        x = g["total_photons"].values
        mu = g["rmse"].values
        se = (g["rmse_sd"] / np.sqrt(g["n_seeds"])).values
        cx_emp = permanent_crossing(np.log10(x), mu)
        crb_row = need.query("structure==@s and protocol==@p and mode==@m")
        v = crb_row["photons_needed"].iloc[0] if len(crb_row) else np.nan
        cx_crb = float(v) if v is not None and np.isfinite(v) else np.nan
        reps = []
        for _ in range(5000):
            pert = mu * rng.lognormal(0, np.log1p((se / np.maximum(mu, 1e-12)) ** 2) ** 0.5)
            c = permanent_crossing(np.log10(x), np.maximum(pert, 1e-6))
            if c is not None and np.isfinite(c):
                reps.append(10 ** c)
        if cx_emp is None and not reps:
            out.append({"structure": s, "protocol": p, "mode": m,
                        "emp_cross": None, "ci_lo": None, "ci_hi": None,
                        "crb_cross": cx_crb, "ratio": None})
            continue
        med = float(np.median(reps)) if reps else cx_emp
        lo, hi = (float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))) if reps \
            else (None, None)
        ratio = None if (med is None or not np.isfinite(med) or not np.isfinite(cx_crb)) \
            else med / cx_crb
        out.append({"structure": s, "protocol": p, "mode": m,
                    "emp_cross": med, "ci_lo": lo, "ci_hi": hi,
                    "crb_cross": cx_crb, "ratio": ratio})
    oc = pd.DataFrame(out)
    oc.to_csv(D / "crossings_with_ci.csv", index=False)
    print(oc.round(3).to_string(index=False))
    ratios = oc["ratio"].dropna()
    if len(ratios):
        print(f"\nempirical/CRB crossing ratio: min={ratios.min():.2f} max={ratios.max():.2f}")


if __name__ == "__main__":
    main()
