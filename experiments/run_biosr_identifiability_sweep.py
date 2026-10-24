from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import mrcfile
from scipy.stats import spearmanr
from skimage.filters import unsharp_mask
from skimage.restoration import richardson_lucy, denoise_tv_chambolle

from phyguard.adapters.fluoresfm import FluoResFMInference
from phyguard.identifiability import convolution_identifiability
from phyguard.optics.forward import PhotonMicroscope
from phyguard.optics.psf import MicroscopyPSFConfig, make_psf
from phyguard.verification import verify_and_fuse

from experiments.run_real_optical_pilot import (
    inject_plausible_duplicate,
    normalize,
    auc_rank,
    masked_mse,
)

STRUCTURES = {
    "ER": {
        "sample": "COS-7 cell",
        "structure": "ER",
        "indicator": "mEmerald (GFP)",
        "microscope_text": (
            "wide-field microscope with excitation numerical aperture of 1.41, "
            "detection numerical aperture of 1.3"
        ),
        "target_text": (
            "linear structured illumination microscope with excitation numerical "
            "aperture of 1.41, detection numerical aperture of 1.3"
        ),
        "input_pixel": "62.6 x 62.6 nm",
        "target_pixel": "31.3 x 31.3 nm",
        "wavelength_nm": 520.0,
    },
    "MT": {
        "sample": "COS-7 cell",
        "structure": "Microtubules",
        "indicator": "mEmerald (GFP)",
        "microscope_text": (
            "wide-field microscope with excitation numerical aperture of 1.41, "
            "detection numerical aperture of 1.3"
        ),
        "target_text": (
            "linear structured illumination microscope with excitation numerical "
            "aperture of 1.41, detection numerical aperture of 1.3"
        ),
        "input_pixel": "62.6 x 62.6 nm",
        "target_pixel": "31.3 x 31.3 nm",
        "wavelength_nm": 520.0,
    },
}


STRUCTURE_DIRS = {"ER": "ER", "MT": "Microtubules", "F-actin": "F-actin_Nonlinear"}


def load_gt(structure: str, cell: int, level: int) -> np.ndarray:

    root = ROOT / "data" / "BioSR" / STRUCTURE_DIRS[structure] / f"Cell_{cell:03d}"
    if structure == "MT":
        path = root / "RawSIMData_gt.mrc"
        with mrcfile.open(path, permissive=True) as m:
            x = np.asarray(m.data, dtype=np.float32)
        x = x[0] if x.ndim == 3 else x
    else:
        path = root / "GTSIM" / f"GTSIM_level_{level:02d}.mrc"
        with mrcfile.open(path, permissive=True) as m:
            x = np.asarray(m.data, dtype=np.float32)
    h, w = x.shape
    y0, x0 = (h - 256) // 2, (w - 256) // 2
    x = x[y0:y0 + 256, x0:x0 + 256]
    return normalize(x, p_low=0.5, p_high=99.5)


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


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--structure", default="ER", choices=["ER", "MT"])
    ap.add_argument("--cell", type=int, default=1)
    ap.add_argument("--level", type=int, default=1)
    args = ap.parse_args()

    outdir = ROOT / "results" / f"biosr_sweep_{args.structure}"
    outdir.mkdir(parents=True, exist_ok=True)
    gd = ROOT / "third_party" / "weights" / "fluoresfm_gdrive"
    inference = FluoResFMInference(
        repo=ROOT / "third_party" / "fluoresfm",
        checkpoint=gd / "checkpoints" / "fluoresfm" / "epoch_0_iter_700000.pt",
        biomedclip_json=gd / "checkpoints" / "biomedclip" / "open_clip_config.json",
        biomedclip_bin=gd / "checkpoints" / "biomedclip" / "open_clip_pytorch_model.bin",
        device="cuda:0",
    )

    cfg = STRUCTURES[args.structure]
    gt = load_gt(args.structure, args.cell, args.level)
    foundation_clean = unsharp_mask(gt, radius=0.8, amount=0.35, preserve_range=True).astype(np.float32)
    foundation_hall, halluc_mask, inj = inject_plausible_duplicate(foundation_clean, patch=28, strength=0.95)

    nas = [0.55, 0.70, 0.85, 1.00, 1.20, 1.40]
    defocuses = [0.0, 0.10, 0.20, 0.35]
    photons_list = [2.0, 5.0, 10.0]
    seeds = [3, 11, 29, 47, 71]
    rows = []

    for na in nas:
        for defocus in defocuses:
            cfg_psf = MicroscopyPSFConfig(
                wavelength_nm=cfg["wavelength_nm"], na=float(na),
                pixel_size_nm=62.6, kernel_size=41,
                aberrations_waves={"defocus": float(defocus)},
            )
            psf = make_psf(cfg_psf)
            ident = convolution_identifiability(psf, gt.shape, gamma=0.05).to_dict()
            psf_np = psf.detach().cpu().numpy()
            for photons in photons_list:
                for seed in seeds:
                    torch.manual_seed(seed)
                    mic = PhotonMicroscope(
                        psf=psf, signal_photons=photons,
                        background_photons=max(0.2, 0.15 * photons),
                        read_noise_std=0.15,
                    )
                    obs = mic.sample(torch.tensor(gt)[None, None])
                    counts = obs["counts"]
                    obs_img = obs["normalized"][0, 0].detach().numpy().astype(np.float32)

                    rl = richardson_lucy(np.clip(obs_img, 0, 1), psf_np, num_iter=12, clip=False)
                    rl = normalize(np.clip(rl, 0, None), p_low=0.1, p_high=99.7)
                    physics_candidate = denoise_tv_chambolle(rl, weight=0.08, channel_axis=None).astype(np.float32)

                    prompt = make_prompt(cfg, photons, mic.background_photons)
                    foundation = inference(obs_img, prompt=prompt)
                    foundation = np.clip(foundation, 0, 1).astype(np.float32)


                    ver = verify_and_fuse(
                        counts,
                        torch.tensor(foundation_hall)[None, None],
                        torch.tensor(physics_candidate)[None, None],
                        mic,
                        local_window=9, temperature=0.70, conservative_bias=0.05,
                    )
                    support = np.clip(ver.support_foundation[0, 0].detach().numpy(), 0, 1)

                    tail_quantile = 0.005
                    tau = float(np.quantile(support, tail_quantile))
                    keep = 1.0 / (1.0 + np.exp(-(support - tau) / 0.02))
                    fused = np.clip(keep * foundation_hall + (1.0 - keep) * physics_candidate, 0, 1)

                    f_mse = masked_mse(gt, foundation_hall, halluc_mask)
                    fused_mse = masked_mse(gt, fused, halluc_mask)
                    rows.append({
                        "structure": args.structure,
                        "cell": args.cell,
                        "level": args.level,
                        "na": na, "defocus_waves": defocus,
                        "photons_per_pixel": photons, "seed": seed,
                        "c_gamma": ident["c_gamma"],
                        "effective_rank_fraction": ident["effective_rank_fraction"],
                        "null_fraction": ident["null_fraction"],
                        "condition_proxy": ident["condition_proxy"],
                        "obs_psnr": float(np.nan),
                        "fluoresfm_psnr": float(np.nan),
                        "fused_psnr": float(np.nan),
                        "hallucination_auc": auc_rank(halluc_mask, 1.0 - support),
                        "support_drop": float(support[~halluc_mask].mean() - support[halluc_mask].mean()),
                        "mse_reduction_factor": float(f_mse / max(fused_mse, 1e-12)),
                    })

                    print(f"done na={na} defocus={defocus} photons={photons} seed={seed}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "per_run_metrics.csv", index=False)
    summary = df.groupby(["na", "defocus_waves", "photons_per_pixel"], as_index=False).agg(
        c_gamma=("c_gamma", "first"),
        effective_rank_fraction=("effective_rank_fraction", "first"),
        hallucination_auc_mean=("hallucination_auc", "mean"),
        hallucination_auc_std=("hallucination_auc", "std"),
        support_drop_mean=("support_drop", "mean"),
        mse_reduction_factor_mean=("mse_reduction_factor", "mean"),
    )
    summary.to_csv(outdir / "summary_by_operator.csv", index=False)

    corr = {}
    for photons, g in summary.groupby("photons_per_pixel"):
        r_auc, p_auc = spearmanr(g["effective_rank_fraction"], g["hallucination_auc_mean"])
        r_cg, p_cg = spearmanr(g["c_gamma"], g["hallucination_auc_mean"])
        corr[f"photons_{int(photons)}"] = {
            "spearman_effrank_vs_auc": float(r_auc), "p": float(p_auc),
            "spearman_cgamma_vs_auc": float(r_cg), "p_cgamma": float(p_cg),
        }
    with open(outdir / "correlations.json", "w") as f:
        json.dump({"n_operators": len(summary) // len(photons_list), "seeds_per_operator": len(seeds), "spearman": corr}, f, indent=2)

    for photons in photons_list:
        g = summary[summary["photons_per_pixel"] == photons]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(g["effective_rank_fraction"], g["hallucination_auc_mean"])
        ax.set_xlabel("OTF effective-rank fraction")
        ax.set_ylabel("Unsupported-structure detection AUC")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"BioSR {args.structure}, {photons:g} photons/pixel")
        fig.tight_layout()
        fig.savefig(outdir / f"effrank_vs_auc_{photons:g}photons.png", dpi=180)
        plt.close(fig)
    print(json.dumps(corr, indent=2))
    print("saved:", outdir)


if __name__ == "__main__":
    main()
