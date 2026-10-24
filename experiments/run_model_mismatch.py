from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from phyguard.estimation.calibration import estimate_aberrations_multiframe
from phyguard.optics.forward import PhotonMicroscope
from phyguard.optics.psf import MicroscopyPSFConfig, make_psf

from experiments.run_biosr_identifiability_sweep import load_gt

MODES = ["defocus", "astig_x", "coma_x", "spherical"]
OFFSETS = [-1.0, -0.5, 0.0, 0.5, 1.0]
TRUE = {"wavelength_nm": 520.0, "na": 1.2, "pixel_size_nm": 62.6}
FACTORS = [0.95, 0.98, 0.99, 1.00, 1.01, 1.02, 1.05]
PHOTONS = 50.0


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--structure", default="ER")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    outdir = ROOT / "results" / "model_mismatch"
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    structure = torch.tensor(load_gt(args.structure, 1, 1))[None, None].to(device)

    rng = np.random.default_rng(7)
    true = {m: float(rng.uniform(-0.2, 0.2)) for m in MODES}
    bg = max(0.2, 0.15 * PHOTONS)


    counts = []
    for off in OFFSETS:
        th = dict(true); th["defocus"] = th["defocus"] + off
        psf = make_psf(MicroscopyPSFConfig(
            wavelength_nm=TRUE["wavelength_nm"], na=TRUE["na"],
            pixel_size_nm=TRUE["pixel_size_nm"], kernel_size=41, aberrations_waves=th,
        )).to(device)
        mic = PhotonMicroscope(psf=psf, signal_photons=PHOTONS,
                               background_photons=bg, read_noise_std=0.15)
        torch.manual_seed(500)
        counts.append(mic.sample(structure)["counts"])
    counts = torch.cat(counts, dim=0)

    rows = []
    for param in ["na", "pixel_size_nm", "wavelength_nm"]:
        for f in FACTORS:
            assumed = dict(TRUE)
            assumed[param] = TRUE[param] * f
            for seed in args.seeds:
                res = estimate_aberrations_multiframe(
                    counts, structure, modes=MODES, defocus_offsets=OFFSETS,
                    wavelength_nm=assumed["wavelength_nm"], na=assumed["na"],
                    pixel_size_nm=assumed["pixel_size_nm"],
                    signal_photons=PHOTONS, background_photons=bg,
                    read_noise_std=0.15, n_steps=500, n_restarts=4, seed=seed,
                )
                row = {"param": param, "factor": f, "seed": seed,
                       "assumed": assumed[param], "nll": res.loss_history[-1]}
                for m in MODES:
                    row[f"bias_{m}"] = res.estimated_coeffs[m] - true[m]
                rows.append(row)
                print(row, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "model_mismatch.csv", index=False)
    g = df.groupby(["param", "factor"]).agg(
        **{f"bias_{m}": (f"bias_{m}", "mean") for m in MODES})
    g["rms_total"] = np.sqrt((g[[f"bias_{m}" for m in MODES]] ** 2).sum(axis=1))
    print(g.round(4).to_string())
    g.to_csv(outdir / "model_mismatch_summary.csv")
    print("saved:", outdir)


if __name__ == "__main__":
    main()
