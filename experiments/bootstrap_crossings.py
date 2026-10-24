from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

TH = 0.05
EVEN = {"defocus", "astig_x", "spherical"}
D = ROOT / "results" / "crb_cutoff_totalbudget"


def permanent_crossing(x, y, thresh=TH):
    order = np.argsort(x)
    x, y = np.log10(x[order]), np.log10(np.maximum(y[order], 1e-12))
    lt = np.log10(thresh)
    above = np.where(y > lt)[0]
    if len(above) == 0:
        return None
    i = above[-1]
    if i == len(x) - 1:
        return np.inf
    return float(10 ** (x[i] + (lt - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])))


def main():
    ps = pd.read_csv(D / "perseed.csv")
    low = D / "perseed_lowend.csv"
    if low.exists():
        ps = pd.concat([ps, pd.read_csv(low)], ignore_index=True)
    rng = np.random.default_rng(0)
    rows = []
    for (s, p, m), g in ps.groupby(["structure", "protocol", "mode"]):
        levels = np.sort(g["total_photons"].unique())
        seeds = np.sort(g["seed"].unique())

        E = np.full((len(seeds), len(levels)), np.nan)
        for i, sd in enumerate(seeds):
            for j, lv in enumerate(levels):
                q = g[(g.seed == sd) & (g.total_photons == lv)]
                if len(q):
                    e = q["estimate"].iloc[0] - q["true"].iloc[0]
                    if m in EVEN:
                        e = min(abs(e), abs(q["estimate"].iloc[0] + q["true"].iloc[0]))
                    E[i, j] = e**2
        reps = []
        for _ in range(5000):
            idx = rng.integers(0, len(seeds), len(seeds))
            rmse = np.sqrt(np.nanmean(E[idx], axis=0))
            c = permanent_crossing(levels, rmse)
            if c is not None and np.isfinite(c):
                reps.append(c)
        if reps:
            rows.append({"structure": s, "protocol": p, "mode": m,
                         "cross_median": float(np.median(reps)),
                         "ci_lo": float(np.percentile(reps, 2.5)),
                         "ci_hi": float(np.percentile(reps, 97.5)),
                         "n_finite": len(reps)})
        else:
            rows.append({"structure": s, "protocol": p, "mode": m,
                         "cross_median": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                         "n_finite": 0})
    out = pd.DataFrame(rows)
    out.to_csv(D / "crossings_bootstrap.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
