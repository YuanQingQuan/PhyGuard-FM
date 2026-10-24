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
        return float("inf")
    return float(10 ** (x[i] + (lt - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])))


def main():
    ps = pd.read_csv(D / "perseed.csv")
    low = D / "perseed_lowend.csv"
    if low.exists():
        ps = pd.concat([ps, pd.read_csv(low)], ignore_index=True)

    rows = []
    for (s, p, m, t), g in ps.groupby(["structure", "protocol", "mode", "total_photons"]):
        e = g["estimate"].values
        true = g["true"].iloc[0]
        signed = np.sqrt(np.mean((e - true) ** 2))
        if m in EVEN:
            sa = np.sqrt(np.mean(np.minimum((e - true) ** 2, (e + true) ** 2)))
        else:
            sa = signed
        rows.append({"structure": s, "protocol": p, "mode": m, "total_photons": t,
                     "rmse_signed": signed, "rmse_signaware": sa,
                     "signflip_frac": float((np.abs(e + true) < np.abs(e - true)).mean())})
    df = pd.DataFrame(rows)

    out = []
    for (s, p, m), g in df.groupby(["structure", "protocol", "mode"]):
        g = g.sort_values("total_photons")
        cs = permanent_crossing(g["total_photons"].values, g["rmse_signed"].values)
        ca = permanent_crossing(g["total_photons"].values, g["rmse_signaware"].values)
        out.append({"structure": s, "protocol": p, "mode": m,
                    "cross_signed": cs, "cross_signaware": ca})
    oc = pd.DataFrame(out)
    oc.to_csv(D / "crossings_signed_vs_signaware.csv", index=False)
    print(oc.to_string(index=False))

    sf = df.groupby(["structure", "protocol", "mode"])["signflip_frac"].mean()
    print("\nmean sign-flip fractions (even modes; 5f should be ~0 if diversity works):")
    print(sf.round(3).to_string())


if __name__ == "__main__":
    main()
