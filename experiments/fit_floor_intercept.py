from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "vectorial_minimal"
AMAX = 0.1
EVEN = {"defocus", "astig_x", "astig_y", "spherical"}


def fit_mode(q: pd.DataFrame):
    q = q[q["alpha"] <= AMAX].sort_values("alpha")
    a2 = q["alpha"].to_numpy() ** 2
    F = q["F_ii"].to_numpy()
    X = np.column_stack([np.ones_like(a2), a2])
    beta, res, *_ = np.linalg.lstsq(X, F, rcond=None)
    F0, c = beta
    n = len(F)
    dof = max(n - 2, 1)
    s2 = float(((F - X @ beta) ** 2).sum() / dof)
    cov = s2 * np.linalg.inv(X.T @ X)
    se0 = float(np.sqrt(cov[0, 0]))
    fmax = float(q["F_ii"].iloc[-1])
    return F0, se0, c, fmax


def main():
    sources = {
        "scalar": ROOT / "results" / "information_budget" / "parity_law_8mode.csv",
        "vec_iso": OUT / "slopes_iso.csv",
        "vec_x": OUT / "slopes_x.csv",
    }
    rows = []
    for name, path in sources.items():
        df = pd.read_csv(path)
        for m in df["mode"].unique():
            F0, se0, c, fmax = fit_mode(df[df["mode"] == m])
            rows.append({
                "model": name, "mode": m,
                "parity": "even" if m in EVEN else "odd",
                "F0": F0, "F0_se": se0,
                "F0_over_F_at_amax": F0 / fmax,
                "c_quad": c,
                "z_F0_eq_0": F0 / se0 if se0 > 0 else np.nan,
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "floor_intercept.csv", index=False)
    print(out.to_string(index=False,
                        float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
