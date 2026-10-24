from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import mrcfile
from skimage.filters import unsharp_mask
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import richardson_lucy, denoise_tv_chambolle

from phyguard.adapters.fluoresfm import FluoResFMInference
from phyguard.optics.forward import PhotonMicroscope
from phyguard.optics.psf import MicroscopyPSFConfig, make_psf
from phyguard.verification import verify_and_fuse

from experiments.run_real_optical_pilot import (
    inject_plausible_duplicate,
    normalize,
    auc_rank,
    masked_mse,
)


def load_gt_frame(cell: int, frame: int) -> np.ndarray:
    path = ROOT / "data" / "BioTISR" / "BioTISR_CCPs" / f"Cell_{cell:03d}" / "SIM_gt.mrc"
    with mrcfile.open(path, permissive=True) as m:
        vol = np.asarray(m.data, dtype=np.float32)
    x = vol[frame]
    y0, x0 = (x.shape[0] - 256) // 2, (x.shape[1] - 256) // 2
    x = x[y0:y0 + 256, x0:x0 + 256]


    return normalize(x, p_low=0.5, p_high=100.0)


def make_prompt(photons: float, bg: float) -> str:
    base = (
        "Task: sr; sample: live SUM-159 cell; structure: CCP; "
        "fluorescence indicator: mEmerald (GFP); "
        "input microscope: wide-field microscope with excitation numerical "
        "aperture of 1.41, excitation wavelength is 488 nm; "
        "input pixel size: 60.4 x 60.4 nm. Nearest interpolation with a factor of 2; "
        "target microscope: linear structured illumination microscope (SIM) "
        "with excitation numerical aperture of 1.41; "
        "target pixel size: 30.2 x 30.2 nm."
    )

    return base


def inject_ccp_duplicate(x: np.ndarray, patch: int = 28, gain: float = 3.0):

    from scipy.ndimage import uniform_filter
    from experiments.run_real_optical_pilot import best_patch_locations
    src, dst = best_patch_locations(x, patch=patch, margin=12)
    sy, sx = src; dy, dx = dst
    source = x[sy:sy + patch, sx:sx + patch].copy()
    t = np.hanning(patch).astype(np.float32)
    win = np.outer(t, t)
    candidate = x.copy()
    before = candidate[dy:dy + patch, dx:dx + patch].copy()
    proposal = np.maximum(before, np.clip(gain * source * win, 0, 1))
    candidate[dy:dy + patch, dx:dx + patch] = proposal
    mask = np.zeros_like(x, bool)
    delta = proposal - before
    mask[dy:dy + patch, dx:dx + patch] = delta > 0.06
    return np.clip(candidate, 0, 1).astype(np.float32), mask, {"source_yx": src, "dest_yx": dst, "patch": patch, "gain": gain}


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--cell", type=int, default=1)
    ap.add_argument("--frame", type=int, default=10)
    ap.add_argument("--photons", type=float, nargs="+", default=[2.0, 5.0, 10.0])
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    outdir = ROOT / "results" / "biotisr_external"
    outdir.mkdir(parents=True, exist_ok=True)
    gd = ROOT / "third_party" / "weights" / "fluoresfm_gdrive"
    inference = FluoResFMInference(
        repo=ROOT / "third_party" / "fluoresfm",
        checkpoint=gd / "checkpoints" / "fluoresfm" / "epoch_0_iter_700000.pt",
        biomedclip_json=gd / "checkpoints" / "biomedclip" / "open_clip_config.json",
        biomedclip_bin=gd / "checkpoints" / "biomedclip" / "open_clip_pytorch_model.bin",
        device="cuda:0",
    )

    gt = load_gt_frame(args.cell, args.frame)


    foundation_hall, halluc_mask, inj = inject_ccp_duplicate(gt, patch=28, gain=3.0)

    psf = make_psf(MicroscopyPSFConfig(
        wavelength_nm=488.0, na=1.2, pixel_size_nm=60.4,
        kernel_size=41, aberrations_waves={"defocus": 0.12},
    ))
    psf_np = psf.detach().cpu().numpy()
    rows = []
    for photons in args.photons:
        torch.manual_seed(args.seed)
        bg = max(0.2, 0.15 * photons)
        mic = PhotonMicroscope(psf=psf, signal_photons=photons,
                               background_photons=bg, read_noise_std=0.15)
        obs = mic.sample(torch.tensor(gt)[None, None])
        counts = obs["counts"]
        obs_img = obs["normalized"][0, 0].detach().numpy().astype(np.float32)

        rl = richardson_lucy(np.clip(obs_img, 0, 1), psf_np, num_iter=12, clip=False)
        rl = normalize(np.clip(rl, 0, None), p_low=0.1, p_high=99.7)
        physics_candidate = denoise_tv_chambolle(rl, weight=0.08, channel_axis=None).astype(np.float32)

        foundation = inference(obs_img, prompt=make_prompt(photons, bg))
        foundation = np.clip(foundation, 0, 1).astype(np.float32)

        ver = verify_and_fuse(
            counts,
            torch.tensor(foundation_hall)[None, None],
            torch.tensor(physics_candidate)[None, None],
            mic, local_window=9, temperature=0.70, conservative_bias=0.05,
        )
        support = np.clip(ver.support_foundation[0, 0].detach().numpy(), 0, 1)
        tau = float(np.quantile(support, 0.005))
        keep = 1.0 / (1.0 + np.exp(-(support - tau) / 0.02))
        fused = np.clip(keep * foundation_hall + (1.0 - keep) * physics_candidate, 0, 1)

        rows.append({
            "cell": args.cell, "frame": args.frame,
            "photons_per_pixel": photons, "seed": args.seed,
            "obs_psnr": peak_signal_noise_ratio(gt, obs_img, data_range=1.0),
            "fluoresfm_psnr": peak_signal_noise_ratio(gt, foundation, data_range=1.0),
            "fluoresfm_ssim": structural_similarity(gt, foundation, data_range=1.0),
            "physics_psnr": peak_signal_noise_ratio(gt, physics_candidate, data_range=1.0),
            "fused_psnr": peak_signal_noise_ratio(gt, fused, data_range=1.0),
            "hallucination_auc": auc_rank(halluc_mask, 1.0 - support),
            "support_drop": float(support[~halluc_mask].mean() - support[halluc_mask].mean()),
            "mse_reduction_factor": float(masked_mse(gt, foundation_hall, halluc_mask) / max(masked_mse(gt, fused, halluc_mask), 1e-12)),
        })
        print(rows[-1], flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / f"external_cell{args.cell}_frame{args.frame}.csv", index=False)
    np.save(outdir / f"external_cell{args.cell}_frame{args.frame}.npy",
            {"gt": gt, "obs": obs_img, "foundation": foundation, "fused": fused, "support": support}, allow_pickle=True)
    print("saved:", outdir)


if __name__ == "__main__":
    main()
