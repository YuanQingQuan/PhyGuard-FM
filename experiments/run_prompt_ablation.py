from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from phyguard.adapters.fluoresfm import FluoResFMInference
from phyguard.optics.forward import PhotonMicroscope
from phyguard.optics.psf import MicroscopyPSFConfig, make_psf

from experiments.run_biosr_identifiability_sweep import load_gt, STRUCTURES

BASE = (
    "Task: sr; sample: COS-7 cell; structure: ER; fluorescence indicator: mEmerald (GFP); "
    "input microscope: wide-field microscope with excitation numerical aperture of 1.41, "
    "detection numerical aperture of 1.3; input pixel size: 62.6 x 62.6 nm; "
    "target microscope: linear structured illumination microscope with excitation numerical "
    "aperture of 1.41, detection numerical aperture of 1.3; target pixel size: 31.3 x 31.3 nm."
)


def physics_tail(photons, bg, na=1.2, wavelength=520.0, pixel=62.6):


    parts = [f"wavelength: {wavelength:g}nm", f"NA: {na:g}",
             f"pixel size: {pixel:g}nm", f"signal photons per pixel: {photons:g}",
             f"background photons per pixel: {bg:g}"]
    return " Optical physics: " + "; ".join(parts) + "."


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--photons", type=float, nargs="+", default=[2.0, 5.0, 10.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[3, 11, 29])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    outdir = ROOT / "results" / "prompt_ablation"
    outdir.mkdir(parents=True, exist_ok=True)
    gd = ROOT / "third_party" / "weights" / "fluoresfm_gdrive"
    inference = FluoResFMInference(
        repo=ROOT / "third_party" / "fluoresfm",
        checkpoint=gd / "checkpoints" / "fluoresfm" / "epoch_0_iter_700000.pt",
        biomedclip_json=gd / "checkpoints" / "biomedclip" / "open_clip_config.json",
        biomedclip_bin=gd / "checkpoints" / "biomedclip" / "open_clip_pytorch_model.bin",
        device=args.device,
    )

    gt = load_gt("ER", 1, 1)
    psf = make_psf(MicroscopyPSFConfig(
        wavelength_nm=520.0, na=1.2, pixel_size_nm=62.6,
        kernel_size=41, aberrations_waves={"defocus": 0.12},
    ))
    rows = []
    for photons in args.photons:
        bg = max(0.2, 0.15 * photons)
        for seed in args.seeds:
            torch.manual_seed(seed)
            mic = PhotonMicroscope(psf=psf, signal_photons=photons,
                                   background_photons=bg, read_noise_std=0.15)
            obs = mic.sample(torch.tensor(gt)[None, None])
            obs_img = obs["normalized"][0, 0].detach().numpy().astype(np.float32)

            prompts = {
                "A0_no_physics": BASE,
                "A1_correct_physics": BASE + physics_tail(photons, bg, na=1.2, wavelength=520.0),
                "A1_wrong_physics": BASE + physics_tail(50.0, bg, na=0.55, wavelength=520.0),
            }
            for pname, prompt in prompts.items():
                out = np.clip(inference(obs_img, prompt=prompt), 0, 1)
                rows.append({
                    "photons_per_pixel": photons, "seed": seed, "prompt": pname,
                    "psnr": peak_signal_noise_ratio(gt, out, data_range=1.0),
                    "ssim": structural_similarity(gt, out, data_range=1.0),
                })
            print(f"photons={photons} seed={seed} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "ablation.csv", index=False)
    g = df.groupby(["photons_per_pixel", "prompt"]).agg(
        psnr=("psnr", "mean"), psnr_std=("psnr", "std"),
        ssim=("ssim", "mean"), ssim_std=("ssim", "std"),
    ).round(4)
    print(g.to_string())


    effects = []
    for photons in args.photons:
        sub = df[df.photons_per_pixel == photons].pivot(index="seed", columns="prompt", values=["psnr", "ssim"])
        a0p = sub[("psnr", "A0_no_physics")].mean()
        a1p = sub[("psnr", "A1_correct_physics")].mean()
        awp = sub[("psnr", "A1_wrong_physics")].mean()
        effects.append({"photons": photons,
                        "d_psnr_A1_vs_A0": a1p - a0p,
                        "d_psnr_wrong_vs_A1": awp - a1p})
    eff = pd.DataFrame(effects)
    eff.to_csv(outdir / "effects.csv", index=False)
    print(eff.to_string(index=False))
    print("saved:", outdir)


if __name__ == "__main__":
    main()
