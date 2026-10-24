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
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from phyguard.adapters.unimie import UniMIEPhysicsSampler
from phyguard.optics.forward import PhotonMicroscope
from phyguard.optics.psf import MicroscopyPSFConfig, make_psf
from phyguard.io import read_image
from experiments.run_fmd_real_benchmark import PAIRS
from experiments.run_real_optical_pilot import normalize


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--denoise-steps", type=int, default=25)
    ap.add_argument("--guidance-scale", type=float, default=3.0)
    ap.add_argument("--photons", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    outdir = ROOT / "results" / "unimie_baseline"
    outdir.mkdir(parents=True, exist_ok=True)
    sampler = UniMIEPhysicsSampler(
        repo=ROOT / "third_party" / "UniMIE",
        model_path=ROOT / "third_party" / "weights" / "256x256_diffusion_uncond.pt",
        device="cuda:0",
        denoise_steps=args.denoise_steps,
        guidance_scale=args.guidance_scale,
    )

    rows = []
    for pair_id, cfg in PAIRS.items():
        ref = read_image(ROOT / "data" / "fmd_minipairs" / pair_id / "reference.png")
        ref = normalize(ref, p_low=0.5, p_high=99.5)
        y0, x0 = (ref.shape[0] - 256) // 2, (ref.shape[1] - 256) // 2
        ref = ref[y0:y0 + 256, x0:x0 + 256]
        ref_t = torch.tensor(ref)[None, None]

        psf_cfg = MicroscopyPSFConfig(
            wavelength_nm=cfg["wavelength_nm"], na=cfg["na"],
            pixel_size_nm=cfg["pixel_size_nm"], kernel_size=41,
            aberrations_waves={"defocus": 0.12},
        )
        psf = make_psf(psf_cfg)
        torch.manual_seed(args.seed)
        bg = max(0.2, 0.15 * args.photons)
        mic = PhotonMicroscope(psf=psf, signal_photons=args.photons,
                               background_photons=bg, read_noise_std=0.15)
        obs = mic.sample(ref_t)
        out = sampler.run(obs["normalized"], obs["counts"], mic)
        est = out[0, 0].detach().cpu().numpy().astype(np.float32)
        obs_img = obs["normalized"][0, 0].detach().numpy().astype(np.float32)

        row = {
            "pair": pair_id,
            "photons_per_pixel": args.photons,
            "seed": args.seed,
            "denoise_steps": args.denoise_steps,
            "guidance_scale": args.guidance_scale,
            "obs_psnr": peak_signal_noise_ratio(ref, obs_img, data_range=1.0),
            "obs_ssim": structural_similarity(ref, obs_img, data_range=1.0),
            "unimie_psnr": peak_signal_noise_ratio(ref, est, data_range=1.0),
            "unimie_ssim": structural_similarity(ref, est, data_range=1.0),
        }
        rows.append(row)
        np.save(outdir / f"unimie_{pair_id}_photons{args.photons:g}.npy", est)
        print(row, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / f"baseline_photons{args.photons:g}.csv", index=False)
    print("saved:", outdir)


if __name__ == "__main__":
    main()
