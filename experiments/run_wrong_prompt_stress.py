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
from scipy.ndimage import maximum_filter
from skimage.filters import unsharp_mask
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import richardson_lucy, denoise_tv_chambolle

from phyguard.adapters.fluoresfm import FluoResFMInference
from phyguard.optics.forward import PhotonMicroscope
from phyguard.optics.psf import MicroscopyPSFConfig, make_psf
from phyguard.verification import measurement_support_map, verify_and_fuse

from experiments.run_real_optical_pilot import normalize
from experiments.run_biosr_identifiability_sweep import load_gt, STRUCTURES

STRUCTURE_PROMPTS = {
    "ER": {
        "sample": "COS-7 cell", "structure": "ER",
        "indicator": "mEmerald (GFP)",
        "microscope_text": (
            "wide-field microscope with excitation numerical aperture of 1.41, "
            "detection numerical aperture of 1.3"
        ),
        "target_text": (
            "linear structured illumination microscope with excitation numerical "
            "aperture of 1.41, detection numerical aperture of 1.3"
        ),
        "input_pixel": "62.6 x 62.6 nm", "target_pixel": "31.3 x 31.3 nm",
        "wavelength_nm": 520.0,
    },
    "CCP": {
        "sample": "COS-7 cell", "structure": "CCP",
        "indicator": "mEmerald (GFP)",
        "microscope_text": (
            "wide-field microscope with excitation numerical aperture of 1.41, "
            "detection numerical aperture of 1.3"
        ),
        "target_text": (
            "linear structured illumination microscope with excitation numerical "
            "aperture of 1.41, detection numerical aperture of 1.3"
        ),
        "input_pixel": "62.6 x 62.6 nm", "target_pixel": "31.3 x 31.3 nm",
        "wavelength_nm": 520.0,
    },
    "F-actin": {
        "sample": "COS-7 cell", "structure": "F-actin",
        "indicator": "mEmerald (GFP)",
        "microscope_text": (
            "wide-field microscope with excitation numerical aperture of 1.41, "
            "detection numerical aperture of 1.3"
        ),
        "target_text": (
            "linear structured illumination microscope with excitation numerical "
            "aperture of 1.41, detection numerical aperture of 1.3"
        ),
        "input_pixel": "62.6 x 62.6 nm", "target_pixel": "31.3 x 31.3 nm",
        "wavelength_nm": 520.0,
    },
    "MT": {
        "sample": "COS-7 cell", "structure": "Microtubules",
        "indicator": "mEmerald (GFP)",
        "microscope_text": (
            "wide-field microscope with excitation numerical aperture of 1.41, "
            "detection numerical aperture of 1.3"
        ),
        "target_text": (
            "linear structured illumination microscope with excitation numerical "
            "aperture of 1.41, detection numerical aperture of 1.3"
        ),
        "input_pixel": "62.6 x 62.6 nm", "target_pixel": "31.3 x 31.3 nm",
        "wavelength_nm": 520.0,
    },
}


def make_prompt(cfg: dict, photons: float, bg: float) -> str:
    base = (
        f"Task: sr; sample: {cfg['sample']}; structure: {cfg['structure']}; "
        f"fluorescence indicator: {cfg['indicator']}; "
        f"input microscope: {cfg['microscope_text']}; "
        f"input pixel size: {cfg['input_pixel']}; "
        f"target microscope: {cfg['target_text']}; "
        f"target pixel size: {cfg['target_pixel']}."
    )

    return base


def punctum_density(x: np.ndarray, size: int = 3) -> float:

    x = np.asarray(x, dtype=np.float32)
    mx = maximum_filter(x, size=size)
    local_max = (x == mx) & (x > 0.15)
    return float(local_max.mean())


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--structure", default="ER", choices=["ER", "MT"])
    ap.add_argument("--cell", type=int, default=1)
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--photons", type=float, nargs="+", default=[1.0, 2.0, 5.0, 10.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[3, 11, 29])
    args = ap.parse_args()

    outdir = ROOT / "results" / "wrong_prompt_stress"
    outdir.mkdir(parents=True, exist_ok=True)
    gd = ROOT / "third_party" / "weights" / "fluoresfm_gdrive"
    inference = FluoResFMInference(
        repo=ROOT / "third_party" / "fluoresfm",
        checkpoint=gd / "checkpoints" / "fluoresfm" / "epoch_0_iter_700000.pt",
        biomedclip_json=gd / "checkpoints" / "biomedclip" / "open_clip_config.json",
        biomedclip_bin=gd / "checkpoints" / "biomedclip" / "open_clip_pytorch_model.bin",
        device="cuda:0",
    )

    gt = load_gt(args.structure, args.cell, args.level)
    psf = make_psf(MicroscopyPSFConfig(
        wavelength_nm=520.0, na=1.2, pixel_size_nm=62.6,
        kernel_size=41, aberrations_waves={"defocus": 0.12},
    ))
    psf_np = psf.detach().cpu().numpy()
    gt_density = punctum_density(gt)

    rows = []
    for photons in args.photons:
        for seed in args.seeds:
            torch.manual_seed(seed)
            bg = max(0.2, 0.15 * photons)
            mic = PhotonMicroscope(psf=psf, signal_photons=photons,
                                   background_photons=bg, read_noise_std=0.15)
            obs = mic.sample(torch.tensor(gt)[None, None])
            counts = obs["counts"]
            obs_img = obs["normalized"][0, 0].detach().numpy().astype(np.float32)

            rl = richardson_lucy(np.clip(obs_img, 0, 1), psf_np, num_iter=12, clip=False)
            rl = normalize(np.clip(rl, 0, None), p_low=0.1, p_high=99.7)
            physics_candidate = denoise_tv_chambolle(rl, weight=0.08, channel_axis=None).astype(np.float32)

            for pname, cfg in STRUCTURE_PROMPTS.items():
                prompt = make_prompt(cfg, photons, bg)
                out = inference(obs_img, prompt=prompt)
                out = np.clip(out, 0, 1).astype(np.float32)

                support = np.clip(measurement_support_map(
                    counts, torch.tensor(out)[None, None], mic,
                    local_window=9, temperature=0.70,
                )[0, 0].detach().numpy(), 0, 1)
                low_support_frac = float((support < 0.5).mean())
                low_support_energy = float(((1 - support) ** 2).mean())

                ver = verify_and_fuse(
                    counts,
                    torch.tensor(out)[None, None],
                    torch.tensor(physics_candidate)[None, None],
                    mic, local_window=9, temperature=0.70, conservative_bias=0.05,
                )
                fused = np.clip(ver.fused[0, 0].detach().numpy(), 0, 1)

                rows.append({
                    "gt_structure": args.structure,
                    "prompt": pname,
                    "prompt_is_correct": pname == args.structure,
                    "photons_per_pixel": photons,
                    "seed": seed,
                    "obs_psnr": peak_signal_noise_ratio(gt, obs_img, data_range=1.0),
                    "out_psnr": peak_signal_noise_ratio(gt, out, data_range=1.0),
                    "out_ssim": structural_similarity(gt, out, data_range=1.0),
                    "fused_psnr": peak_signal_noise_ratio(gt, fused, data_range=1.0),
                    "fused_ssim": structural_similarity(gt, fused, data_range=1.0),
                    "out_punctum_density": punctum_density(out),
                    "gt_punctum_density": gt_density,
                    "morphology_shift": punctum_density(out) - gt_density,
                    "low_support_frac": low_support_frac,
                    "low_support_energy": low_support_energy,
                    "psnr_gain_from_fusion": peak_signal_noise_ratio(gt, fused, data_range=1.0) - peak_signal_noise_ratio(gt, out, data_range=1.0),
                    "ssim_gain_from_fusion": structural_similarity(gt, fused, data_range=1.0) - structural_similarity(gt, out, data_range=1.0),
                })
                print(rows[-1], flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / f"stress_{args.structure}.csv", index=False)
    print(df.to_string(index=False))
    print("saved:", outdir)


if __name__ == "__main__":
    main()
