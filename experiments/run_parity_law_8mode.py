from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from phyguard.estimation.info import parameter_fisher
from experiments.run_biosr_identifiability_sweep import load_gt

MODES8 = ["defocus", "astig_x", "astig_y", "coma_x", "coma_y",
          "trefoil_x", "trefoil_y", "spherical"]
EVEN8 = {"defocus", "astig_x", "astig_y", "spherical"}


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gt = torch.tensor(load_gt("ER", 1, 1))[None, None].to(device)
    alphas = np.logspace(-3, np.log10(0.3), 12)
    rows = []
    for m in MODES8:
        for a in alphas:
            coeffs = {k: torch.tensor(float(a) if k == m else 0.0,
                                      requires_grad=False, device=device)
                      for k in MODES8}
            F = parameter_fisher(coeffs, gt, wavelength_nm=520.0, na=1.2,
                                 pixel_size_nm=62.6, signal_photons=50.0,
                                 background_photons=max(0.2, 0.15 * 50.0),
                                 read_noise_std=0.15)
            i = MODES8.index(m)
            rows.append({"mode": m, "parity": "even" if m in EVEN8 else "odd",
                         "alpha": float(a), "F_ii": float(F[i, i])})
        print(m, "done", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "information_budget"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "parity_law_8mode.csv", index=False)
    print("\n=== slope of log F_ii vs log alpha (full range) ===")
    for m in MODES8:
        q = df[df["mode"] == m]
        a, f = np.log10(q["alpha"]), np.log10(np.maximum(q["F_ii"], 1e-30))
        slope, intercept = np.polyfit(a, f, 1)
        print(f"  {m:10s} ({'even' if m in EVEN8 else 'odd '}): slope {slope:+.2f}")


if __name__ == "__main__":
    main()
