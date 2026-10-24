from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio
from skimage.restoration import richardson_lucy

from phyguard.calibration import estimate_aberrations, fisher_information, ABERRATION_MODES
from phyguard.optics.forward import PhotonMicroscope
from phyguard.optics.psf import MicroscopyPSFConfig, make_psf

from experiments.run_real_optical_pilot import normalize
from experiments.run_biosr_identifiability_sweep import load_gt

MODES = ["defocus", "astig_x", "coma_x", "spherical"]


def load_structure(name: str, device: torch.device) -> torch.Tensor:
    if name == "ER":
        x = load_gt("ER", 1, 1)
    elif name == "MT":
        x = load_gt("MT", 1, 1)
    else:
        raise ValueError(name)
    return torch.tensor(x)[None, None].to(device)


def random_aberrations(rng: np.random.Generator) -> dict:
    return {m: float(rng.uniform(-0.3, 0.3)) for m in MODES}


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--structures", nargs="+", default=["ER", "MT"])
    ap.add_argument("--photons", type=float, nargs="+", default=[5.0, 10.0, 50.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-steps", type=int, default=350)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    outdir = ROOT / "results" / "aberration_calibration"
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    rows = []
    for sname in args.structures:
        structure = load_structure(sname, device)
        for photons in args.photons:
            for seed in args.seeds:
                rng = np.random.default_rng(seed)
                true = random_aberrations(rng)

                psf_true = make_psf(MicroscopyPSFConfig(
                    wavelength_nm=520.0, na=1.2, pixel_size_nm=62.6,
                    kernel_size=41, aberrations_waves=true,
                )).to(device)
                bg = max(0.2, 0.15 * photons)
                mic = PhotonMicroscope(
                    psf=psf_true, signal_photons=photons,
                    background_photons=bg, read_noise_std=0.15,
                )
                torch.manual_seed(100 + seed)
                obs = mic.sample(structure)
                counts = obs["counts"]

                res = estimate_aberrations(
                    counts, structure, modes=MODES,
                    wavelength_nm=520.0, na=1.2, pixel_size_nm=62.6,
                    signal_photons=photons, background_photons=bg,
                    read_noise_std=0.15, n_steps=args.n_steps, lr=0.05,
                    init={m: 0.0 for m in MODES},
                )

                for m in MODES:
                    est = res.estimated_coeffs[m]
                    rows.append({
                        "structure": sname,
                        "photons_per_pixel": photons,
                        "seed": seed,
                        "mode": m,
                        "true": true[m],
                        "estimated": est,
                        "abs_error": abs(est - true[m]),
                        "sign_aware_error": min(abs(est - true[m]), abs(est + true[m])),
                        "magnitude_error": abs(abs(est) - abs(true[m])),
                        "sign_hit": int(np.sign(est) == np.sign(true[m])),
                        "crb": res.crb.get(m, float("nan")),
                        "fisher_cond": float(np.linalg.cond(res.fisher_info)) if res.fisher_info is not None else float("nan"),
                        "n_degenerate_directions": res.n_degenerate_directions,
                    })

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "per_run.csv", index=False)


    summary = df.groupby(["structure", "photons_per_pixel", "mode"], as_index=False).agg(
        mean_abs_error=("abs_error", "mean"),
        mean_sign_aware_error=("sign_aware_error", "mean"),
        mean_magnitude_error=("magnitude_error", "mean"),
        sign_hit_rate=("sign_hit", "mean"),
        empirical_std=("estimated", "std"),
        mean_crb=("crb", "mean"),
        crb_std=("crb", "std"),
    )
    summary["crb_ratio"] = summary["mean_crb"] / np.maximum(summary["empirical_std"] ** 2, 1e-12)
    summary.to_csv(outdir / "summary.csv", index=False)
    print(summary.to_string(index=False))

    cond_summary = df.groupby(["structure", "photons_per_pixel"], as_index=False).agg(
        fisher_cond_mean=("fisher_cond", "mean"),
        fisher_cond_std=("fisher_cond", "std"),
    )
    cond_summary.to_csv(outdir / "fisher_condition.csv", index=False)
    print(cond_summary.to_string(index=False))


    rec_rows = []
    for sname in ["ER"]:
        structure = load_structure(sname, device)
        gt_np = structure[0, 0].detach().cpu().numpy()
        for photons in [5.0, 50.0]:
            rng = np.random.default_rng(7)
            true = random_aberrations(rng)
            psf_true = make_psf(MicroscopyPSFConfig(
                wavelength_nm=520.0, na=1.2, pixel_size_nm=62.6,
                kernel_size=41, aberrations_waves=true,
            )).to(device)
            bg = max(0.2, 0.15 * photons)
            mic = PhotonMicroscope(psf=psf_true, signal_photons=photons,
                                   background_photons=bg, read_noise_std=0.15)
            torch.manual_seed(107)
            obs = mic.sample(structure)
            obs_img = np.clip(obs["normalized"][0, 0].detach().cpu().numpy(), 0, 1)
            res = estimate_aberrations(
                obs["counts"], structure, modes=MODES,
                wavelength_nm=520.0, na=1.2, pixel_size_nm=62.6,
                signal_photons=photons, background_photons=bg,
                read_noise_std=0.15, n_steps=args.n_steps,
                init={m: 0.0 for m in MODES},
            )
            est_psf = make_psf(MicroscopyPSFConfig(
                wavelength_nm=520.0, na=1.2, pixel_size_nm=62.6,
                kernel_size=41, aberrations_waves=res.estimated_coeffs,
            )).to(device)
            nominal_psf = make_psf(MicroscopyPSFConfig(
                wavelength_nm=520.0, na=1.2, pixel_size_nm=62.6, kernel_size=41,
            )).to(device)

            def rl_psnr(psf_np):
                r = richardson_lucy(obs_img, psf_np, num_iter=15, clip=False)
                r = normalize(np.clip(r, 0, None), p_low=0.1, p_high=99.7)
                return peak_signal_noise_ratio(gt_np, r, data_range=1.0)

            rec_rows.append({
                "structure": sname, "photons_per_pixel": photons,
                "obs_psnr": peak_signal_noise_ratio(gt_np, obs_img, data_range=1.0),
                "rl_true_psnr": rl_psnr(psf_true.detach().cpu().numpy()),
                "rl_calibrated_psnr": rl_psnr(est_psf.detach().cpu().numpy()),
                "rl_nominal_psnr": rl_psnr(nominal_psf.detach().cpu().numpy()),
                "est_error_mean": float(np.mean([abs(res.estimated_coeffs[m] - true[m]) for m in MODES])),
            })
    pd.DataFrame(rec_rows).to_csv(outdir / "reconstruction.csv", index=False)
    print(pd.DataFrame(rec_rows).to_string(index=False))
    print("saved:", outdir)


if __name__ == "__main__":
    main()
