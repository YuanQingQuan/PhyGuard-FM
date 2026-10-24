from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio

from phyguard.optics.psf import MicroscopyPSFConfig, make_psf
from phyguard.optics.forward import PhotonMicroscope
from phyguard.estimation.dip import dip_blind_estimate
from experiments.run_blind_calibration import blind_estimate, MODES
from experiments.run_real_optical_pilot import normalize
from experiments.run_biosr_identifiability_sweep import load_gt
from skimage.restoration import richardson_lucy


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--structure", default="ER")
    ap.add_argument("--photons", type=float, default=20.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--multi", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    outdir = ROOT / "results" / "dip_blind"
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    gt = torch.tensor(load_gt(args.structure, 1, 1))[None, None].to(device)
    offsets = [-0.3, 0.0, 0.3] if args.multi else [0.0]
    import pandas as pd
    rows = []
    for seed in args.seeds:
        rng = np.random.default_rng(100 + seed)
        true = {m: float(rng.uniform(-0.25, 0.25)) for m in MODES}
        photons = args.photons
        bg = max(0.2, 0.15 * photons)
        counts_all = []

        torch.manual_seed(200 + seed)
        for off in offsets:
            th = dict(true); th["defocus"] = th["defocus"] + off
            psf = make_psf(MicroscopyPSFConfig(
                wavelength_nm=520.0, na=1.2, pixel_size_nm=62.6,
                kernel_size=41, aberrations_waves=th,
            )).to(device)
            mic = PhotonMicroscope(psf=psf, signal_photons=photons,
                                   background_photons=bg, read_noise_std=0.15)
            counts_all.append(mic.sample(gt)["counts"])
        counts = torch.stack(counts_all, dim=0) if len(counts_all) > 1 else counts_all[0]

        mid = len(offsets) // 2
        c = counts[mid] if len(offsets) > 1 else counts
        obs_img = torch.clamp((c - bg) / photons, 0, 1)[0, 0].cpu().numpy().astype(np.float32)
        nom = make_psf(MicroscopyPSFConfig(
            wavelength_nm=520.0, na=1.2, pixel_size_nm=62.6, kernel_size=41,
        )).detach().cpu().numpy()
        rl = richardson_lucy(np.clip(obs_img, 0, 1), nom, num_iter=15, clip=False)
        x_init = torch.tensor(normalize(np.clip(rl, 0, None), p_low=0.1, p_high=99.7))[None, None].to(device)


        res_tv = blind_estimate(
            counts, x_init, modes=MODES, wavelength_nm=520.0, na=1.2,
            pixel_size_nm=62.6, signal_photons=photons, background_photons=bg,
            read_noise_std=0.15, tv_weight=0.10, n_outer=15, n_inner=6,
            lr_x=0.01, lr_theta=0.2, seed=seed,
            defocus_offsets=offsets if len(offsets) > 1 else None,
        )

        res_dip = dip_blind_estimate(
            counts, modes=MODES, wavelength_nm=520.0, na=1.2,
            pixel_size_nm=62.6, signal_photons=photons, background_photons=bg,
            read_noise_std=0.15, n_steps=600, lr_net=0.01, lr_theta=0.1,
            seed=seed, n_restarts=3, defocus_offsets=offsets if len(offsets) > 1 else None,
        )
        gt_np = gt[0, 0].cpu().numpy()
        row = {"seed": seed, "multi": args.multi,
               "tv_recon_psnr": peak_signal_noise_ratio(gt_np, res_tv["x"][0, 0].cpu().numpy(), data_range=1.0),
               "dip_recon_psnr": peak_signal_noise_ratio(gt_np, res_dip["x"], data_range=1.0),
               "obs_psnr": peak_signal_noise_ratio(gt_np, obs_img, data_range=1.0)}
        for m in MODES:
            row[f"tv_err_{m}"] = min(abs(res_tv["theta"][m] - true[m]), abs(res_tv["theta"][m] + true[m]))
            row[f"dip_err_{m}"] = min(abs(res_dip["theta"][m] - true[m]), abs(res_dip["theta"][m] + true[m]))
        rows.append(row)
        print(row, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / f"dip_vs_tv_{args.structure}_multi{int(args.multi)}.csv", index=False)
    print(df.mean(numeric_only=True).round(4).to_string())
    print("saved:", outdir)


if __name__ == "__main__":
    main()
