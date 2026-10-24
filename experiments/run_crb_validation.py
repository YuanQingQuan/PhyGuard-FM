from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from phyguard.estimation.calibration import estimate_aberrations, parameterized_psf
from phyguard.estimation.info import parameter_fisher, crb_identifiable
from phyguard.optics.forward import PhotonMicroscope
from phyguard.optics.psf import MicroscopyPSFConfig, make_psf

from experiments.run_biosr_identifiability_sweep import load_gt

MODES = ["defocus", "astig_x", "coma_x", "spherical"]


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--structures", nargs="+", default=["ER", "MT"])
    ap.add_argument("--photons", type=float, nargs="+", default=[5.0, 10.0, 50.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    outdir = ROOT / "results" / "crb_validation"
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rows = []
    for sname in args.structures:
        structure = torch.tensor(load_gt(sname, 1, 1))[None, None].to(device)
        for photons in args.photons:
            bg = max(0.2, 0.15 * photons)

            rng = np.random.default_rng(7)
            true = {m: float(rng.uniform(-0.2, 0.2)) for m in MODES}
            coeffs = {m: torch.tensor(true[m], requires_grad=True, device=device) for m in MODES}
            F = parameter_fisher(
                coeffs, structure, wavelength_nm=520.0, na=1.2, pixel_size_nm=62.6,
                signal_photons=photons, background_photons=bg, read_noise_std=0.15,
            )
            crb, n_degen = crb_identifiable(F, MODES)

            ests = {m: [] for m in MODES}
            for seed in args.seeds:
                psf_true = make_psf(MicroscopyPSFConfig(
                    wavelength_nm=520.0, na=1.2, pixel_size_nm=62.6,
                    kernel_size=41, aberrations_waves=true,
                )).to(device)
                mic = PhotonMicroscope(psf=psf_true, signal_photons=photons,
                                       background_photons=bg, read_noise_std=0.15)
                torch.manual_seed(100 + seed)
                obs = mic.sample(structure)
                res = estimate_aberrations(
                    obs["counts"], structure, modes=MODES,
                    wavelength_nm=520.0, na=1.2, pixel_size_nm=62.6,
                    signal_photons=photons, background_photons=bg,
                    read_noise_std=0.15, n_steps=500, n_restarts=4, seed=seed,
                )
                for m in MODES:
                    ests[m].append(res.estimated_coeffs[m])

            for m in MODES:
                e = np.array(ests[m])

                sign = np.sign(true[m]) if true[m] != 0 else 1.0
                rmse = float(np.sqrt(np.mean(np.minimum((e - true[m]) ** 2, (e + true[m]) ** 2))))
                rows.append({
                    "structure": sname, "photons_per_pixel": photons, "mode": m,
                    "true": true[m],
                    "sqrt_crb": float(np.sqrt(max(crb.get(m, np.nan), 0.0))),
                    "empirical_rmse": rmse,
                    "n_degenerate_directions": n_degen,
                    "rmse_over_sqrt_crb": rmse / max(np.sqrt(max(crb.get(m, 1e-12), 1e-12)), 1e-12),
                })

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "crb_validation.csv", index=False)
    print(df.to_string(index=False))
    print("saved:", outdir)


if __name__ == "__main__":
    main()
