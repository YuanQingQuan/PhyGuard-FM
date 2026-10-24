from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import tifffile
import torch
from skimage.restoration import richardson_lucy

from phyguard.optics.psf import MicroscopyPSFConfig, make_psf
from phyguard.optics.forward import PhotonMicroscope
from experiments.run_blind_calibration import blind_estimate, MODES
from experiments.run_real_optical_pilot import normalize

NA, WAVELENGTH_NM, PIXEL_NM = 1.1, 515.0, 86.0


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--magnitudes", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    ap.add_argument("--photons", type=float, default=30.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-outer", type=int, default=20)
    ap.add_argument("--n-inner", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    outdir = ROOT / "results" / "real_beads_multilevel"
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    vol = tifffile.imread(ROOT / "data" / "cocoa_beads" / "ref_no_aberration.tif").astype(np.float32)
    zi = vol.shape[0] // 2
    h, w = vol.shape[-2:]
    y0, x0 = (h - args.crop) // 2, (w - args.crop) // 2
    frame = normalize(vol[zi][y0:y0+args.crop, x0:x0+args.crop], p_low=0.5, p_high=99.5)
    gt_t = torch.tensor(frame)[None, None].to(device)

    nom_psf = make_psf(MicroscopyPSFConfig(
        wavelength_nm=WAVELENGTH_NM, na=NA, pixel_size_nm=PIXEL_NM, kernel_size=41,
    )).detach().cpu().numpy()

    rows = []
    for mag in args.magnitudes:
        for seed in args.seeds:
            rng = np.random.default_rng(50 + seed)
            true = {m: float(rng.uniform(-mag, mag)) for m in MODES}
            psf_true = make_psf(MicroscopyPSFConfig(
                wavelength_nm=WAVELENGTH_NM, na=NA, pixel_size_nm=PIXEL_NM,
                kernel_size=41, aberrations_waves=true,
            )).to(device)
            mic = PhotonMicroscope(psf=psf_true, signal_photons=args.photons,
                                   background_photons=max(0.2, 0.15*args.photons), read_noise_std=0.15)
            torch.manual_seed(60 + seed)
            obs = mic.sample(gt_t)
            counts = obs["counts"]
            obs_img = torch.clamp((counts - mic.background_photons) / args.photons, 0, 1)[0, 0].cpu().numpy().astype(np.float32)
            rl0 = richardson_lucy(np.clip(obs_img, 0, 1), nom_psf, num_iter=15, clip=False)
            x_init = torch.tensor(normalize(np.clip(rl0, 0, None), p_low=0.1, p_high=99.7))[None, None].to(device)

            res = blind_estimate(
                counts, x_init, modes=MODES, wavelength_nm=WAVELENGTH_NM, na=NA,
                pixel_size_nm=PIXEL_NM, signal_photons=args.photons,
                background_photons=mic.background_photons, read_noise_std=0.15,
                tv_weight=0.08, n_outer=args.n_outer, n_inner=args.n_inner,
                lr_x=0.01, lr_theta=0.15, seed=seed,
            )
            est = res["theta"]
            errs = {m: min(abs(est[m]-true[m]), abs(est[m]+true[m])) for m in MODES}
            row = {"magnitude": mag, "seed": seed,
                   "mean_err": float(np.mean(list(errs.values()))),
                   "max_err": float(np.max(list(errs.values())))}
            for m in MODES:
                row[f"err_{m}"] = errs[m]
            rows.append(row)
            print(row, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "multilevel.csv", index=False)
    g = df.groupby("magnitude").agg(mean_err=("mean_err","mean"), max_err=("max_err","mean"),
                                    err_coma=("err_coma_x","mean"), err_sph=("err_spherical","mean"),
                                    err_defocus=("err_defocus","mean"), err_astig=("err_astig_x","mean")).round(4)
    print(g.to_string())
    print("saved:", outdir)


if __name__ == "__main__":
    main()
