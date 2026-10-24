from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from phyguard.estimation.info import information_budget
from experiments.run_biosr_identifiability_sweep import load_gt
from phyguard.estimation.calibration import ABERRATION_MODES as MODES


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--structures", nargs="+", default=["ER", "MT"])
    ap.add_argument("--photons", type=float, nargs="+", default=[1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    outdir = ROOT / "results" / "information_budget"
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rows = []
    for sname in args.structures:
        structure = torch.tensor(load_gt(sname, 1, 1))[None, None].to(device)
        for photons in args.photons:
            for seed in args.seeds:
                rng = np.random.default_rng(seed)
                true = {m: float(rng.uniform(-0.2, 0.2)) for m in MODES}
                coeffs = {m: torch.tensor(true[m], requires_grad=True, device=device) for m in MODES}
                bg = max(0.2, 0.15 * photons)
                info = information_budget(
                    coeffs, structure,
                    wavelength_nm=520.0, na=1.2, pixel_size_nm=62.6,
                    signal_photons=photons, background_photons=bg,
                    read_noise_std=0.15, gamma=0.05,
                )
                row = {"structure": sname, "photons_per_pixel": photons, "seed": seed,
                       "n_degenerate": info.n_degenerate_directions,
                       "c_gamma": info.content_c_gamma,
                       "eff_rank_frac": info.content_eff_rank_fraction,
                       "content_fisher_diag_mean": info.content_fisher_diag_mean,
                       "budget_ratio": info.budget_ratio_param_vs_content}
                for m in MODES:
                    row[f"crb_{m}"] = info.param_crb.get(m, float("nan"))
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "budget.csv", index=False)
    g = df.groupby(["structure", "photons_per_pixel"]).agg(
        n_degenerate=("n_degenerate", "mean"),
        c_gamma=("c_gamma", "mean"),
        eff_rank_frac=("eff_rank_frac", "mean"),
        crb_defocus=("crb_defocus", "mean"),
        crb_coma=("crb_coma_x", "mean"),
        content_fisher=("content_fisher_diag_mean", "mean"),
    ).round(6)
    print(g.to_string())
    print("saved:", outdir)


if __name__ == "__main__":
    main()
