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
from scipy.stats import spearmanr
from skimage.filters import unsharp_mask
from skimage.restoration import richardson_lucy, denoise_tv_chambolle
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from phyguard.adapters.fluoresfm import FluoResFMInference
from phyguard.identifiability import convolution_identifiability
from phyguard.io import read_image
from phyguard.optics.forward import PhotonMicroscope
from phyguard.optics.psf import MicroscopyPSFConfig, make_psf
from phyguard.physics_stats import estimate_blind_stats
from phyguard.verification import verify_and_fuse

from experiments.run_real_optical_pilot import (
    bootstrap_ci,
    inject_plausible_duplicate,
    normalize,
    auc_rank,
    masked_mse,
)

PAIRS = {
    "Confocal_FISH_3": {
        "modality": "confocal",
        "task": "dn",
        "structure": "NCC",
        "sample": "fixed zebrafish embryos",
        "indicator": "eGFP (GFP)",
        "microscope_text": (
            "Nikon A1R-MP laser scanning confocal microscope equipped with a "
            "Nikon Apo LWD 40x, 1.15 NA water immersion objective. "
            "Excitation wavelength 488 nm, 10% laser power, 140 PMT gain"
        ),
        "pixel_size_nm": 300.0,
        "wavelength_nm": 488.0,
        "na": 1.15,
    },
    "WideField_BPAE": {
        "modality": "wide-field",
        "task": "dn",
        "structure": "BPAE cell",
        "sample": "fixed bovine pulmonary artery endothelial (BPAE) cell",
        "indicator": "DAPI / FITC / TRITC",
        "microscope_text": (
            "wide-field fluorescence microscope with excitation numerical "
            "aperture of 1.41, detection numerical aperture of 1.3"
        ),
        "pixel_size_nm": 100.0,
        "wavelength_nm": 520.0,
        "na": 1.3,
    },
}


def make_prompt(cfg: dict, photons: float | None, bg: float | None) -> str:

    base = (
        f"Task: {cfg['task']}; sample: {cfg['sample']}; structure: {cfg['structure']}; "
        f"fluorescence indicator: {cfg['indicator']}; "
        f"input microscope: {cfg['microscope_text']}; "
        f"input pixel size: {cfg['pixel_size_nm']:g} x {cfg['pixel_size_nm']:g} nm; "
        f"target microscope: {cfg['microscope_text']}; "
        f"target pixel size: {cfg['pixel_size_nm']:g} x {cfg['pixel_size_nm']:g} nm."
    )


    return base


def track1(inference, outdir: Path) -> pd.DataFrame:

    rows = []
    datadir = ROOT / "data" / "fmd_minipairs"
    for pair_id, cfg in PAIRS.items():
        pair_dir = datadir / pair_id
        raw = read_image(pair_dir / "raw.png")
        ref = read_image(pair_dir / "reference.png")

        h = min(raw.shape[0], ref.shape[0])
        w = min(raw.shape[1], ref.shape[1])
        raw, ref = raw[:h, :w], ref[:h, :w]
        prompt = make_prompt(cfg, None, None)
        est = inference(raw, prompt=prompt)
        est = np.clip(est, 0, 1).astype(np.float32)

        upstream_denoised = read_image(pair_dir / "upstream_denoised.png")
        upstream_denoised = upstream_denoised[:h, :w]

        stats = estimate_blind_stats(torch.tensor(raw)[None, None])
        rows.append({
            "pair": pair_id,
            "modality": cfg["modality"],
            "raw_psnr": peak_signal_noise_ratio(ref, raw, data_range=1.0),
            "raw_ssim": structural_similarity(ref, raw, data_range=1.0),
            "fluoresfm_psnr": peak_signal_noise_ratio(ref, est, data_range=1.0),
            "fluoresfm_ssim": structural_similarity(ref, est, data_range=1.0),
            "upstream_denoised_psnr": peak_signal_noise_ratio(ref, upstream_denoised, data_range=1.0),
            "upstream_denoised_ssim": structural_similarity(ref, upstream_denoised, data_range=1.0),
            "blind_snr_proxy_db": stats.snr_proxy_db,
            "blind_hf_noise_std": stats.high_frequency_noise_std,
        })

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, (im, t) in zip(axes, [
            (raw, "FMD raw (real noise)"),
            (est, "FluoResFM (frozen)"),
            (upstream_denoised, "Upstream denoised"),
            (ref, "FMD reference (50-frame avg)"),
        ]):
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
            ax.set_title(t)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(outdir / f"track1_{pair_id}.png", dpi=160)
        plt.close(fig)
        np.save(outdir / f"track1_{pair_id}_fluoresfm.npy", est)
    return pd.DataFrame(rows)


def track2(inference, outdir: Path) -> pd.DataFrame:

    photons_levels = [0.5, 1, 2, 5, 10, 20, 50]
    seeds = [3, 11, 29, 47, 71]
    rows = []
    datadir = ROOT / "data" / "fmd_minipairs"
    for pair_id, cfg in PAIRS.items():
        ref = read_image(datadir / pair_id / "reference.png")
        ref = normalize(ref, p_low=0.5, p_high=99.5)


        if ref.shape[0] < 256 or ref.shape[1] < 256:
            raise ValueError(f"{pair_id} reference smaller than 256x256")
        y0, x0 = (ref.shape[0] - 256) // 2, (ref.shape[1] - 256) // 2
        ref = ref[y0:y0 + 256, x0:x0 + 256]


        foundation_clean = unsharp_mask(ref, radius=0.8, amount=0.35, preserve_range=True).astype(np.float32)
        foundation_hall, halluc_mask, inj = inject_plausible_duplicate(foundation_clean, patch=28, strength=0.95)

        psf_cfg = MicroscopyPSFConfig(
            wavelength_nm=cfg["wavelength_nm"],
            na=cfg["na"],
            pixel_size_nm=cfg["pixel_size_nm"],
            kernel_size=41,
            aberrations_waves={"defocus": 0.12},
        )
        psf = make_psf(psf_cfg)
        ident = convolution_identifiability(psf, ref.shape, gamma=0.05).to_dict()
        psf_np = psf.detach().cpu().numpy()

        for photons in photons_levels:
            for seed in seeds:
                torch.manual_seed(seed)
                bg = max(0.2, 0.15 * photons)
                mic = PhotonMicroscope(
                    psf=psf, signal_photons=float(photons),
                    background_photons=bg, read_noise_std=0.15,
                )
                obs = mic.sample(torch.tensor(ref)[None, None])
                counts = obs["counts"]
                obs_img = obs["normalized"][0, 0].detach().numpy().astype(np.float32)


                rl = richardson_lucy(np.clip(obs_img, 0, 1), psf_np, num_iter=12, clip=False)
                rl = normalize(np.clip(rl, 0, None), p_low=0.1, p_high=99.7)
                physics_candidate = denoise_tv_chambolle(rl, weight=0.08, channel_axis=None).astype(np.float32)


                prompt = make_prompt(cfg, photons, bg)
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
                gate_scale = 0.02
                tau = float(np.quantile(support, tail_quantile))
                keep_prior = 1.0 / (1.0 + np.exp(-(support - tau) / gate_scale))
                fused = np.clip(keep_prior * foundation_hall + (1.0 - keep_prior) * physics_candidate, 0, 1)

                rows.append({
                    "pair": pair_id,
                    "modality": cfg["modality"],
                    "photons_per_pixel": photons,
                    "seed": seed,
                    "na": cfg["na"],
                    "wavelength_nm": cfg["wavelength_nm"],
                    "c_gamma": ident["c_gamma"],
                    "effective_rank_fraction": ident["effective_rank_fraction"],
                    "obs_psnr": peak_signal_noise_ratio(ref, obs_img, data_range=1.0),
                    "fluoresfm_psnr": peak_signal_noise_ratio(ref, foundation, data_range=1.0),
                    "physics_psnr": peak_signal_noise_ratio(ref, physics_candidate, data_range=1.0),
                    "fused_psnr": peak_signal_noise_ratio(ref, fused, data_range=1.0),
                    "hallucination_auc": auc_rank(halluc_mask, 1.0 - support),
                    "support_drop": float(support[~halluc_mask].mean() - support[halluc_mask].mean()),
                    "foundation_hallucination_mse": masked_mse(ref, foundation_hall, halluc_mask),
                    "fused_hallucination_mse": masked_mse(ref, fused, halluc_mask),
                    "physics_hallucination_mse": masked_mse(ref, physics_candidate, halluc_mask),
                    "mse_reduction_factor": float(masked_mse(ref, foundation_hall, halluc_mask) / max(masked_mse(ref, fused, halluc_mask), 1e-12)),
                })
                print(f"[track2] {pair_id} photons={photons} seed={seed} done", flush=True)
                if photons == 5 and seed == 11:
                    np.savez(outdir / f"track2_{pair_id}_representative.npz",
                             ref=ref, obs=obs_img, foundation=foundation,
                             support=support, fused=fused, halluc_mask=halluc_mask)
    return pd.DataFrame(rows)


def main():
    outdir = ROOT / "results" / "fmd_real_benchmark"
    outdir.mkdir(parents=True, exist_ok=True)
    gd = ROOT / "third_party" / "weights" / "fluoresfm_gdrive"
    inference = FluoResFMInference(
        repo=ROOT / "third_party" / "fluoresfm",
        checkpoint=gd / "checkpoints" / "fluoresfm" / "epoch_0_iter_700000.pt",
        biomedclip_json=gd / "checkpoints" / "biomedclip" / "open_clip_config.json",
        biomedclip_bin=gd / "checkpoints" / "biomedclip" / "open_clip_pytorch_model.bin",
        device="cuda:0",
    )

    t1 = track1(inference, outdir)
    t1.to_csv(outdir / "track1_real_noise.csv", index=False)
    print(t1.to_string(index=False))

    t2 = track2(inference, outdir)
    t2.to_csv(outdir / "track2_controlled_physics.csv", index=False)
    summary = t2.groupby(["pair", "photons_per_pixel"], as_index=False).agg(
        hallucination_auc_mean=("hallucination_auc", "mean"),
        hallucination_auc_std=("hallucination_auc", "std"),
        support_drop_mean=("support_drop", "mean"),
        mse_reduction_factor_mean=("mse_reduction_factor", "mean"),
        fluoresfm_psnr_mean=("fluoresfm_psnr", "mean"),
        fused_psnr_mean=("fused_psnr", "mean"),
    )
    summary.to_csv(outdir / "track2_summary_by_photons.csv", index=False)
    print(summary.to_string(index=False))


    corr_rows = []
    for pair_id, g in t2.groupby("pair"):

        rho, pval = spearmanr(g["photons_per_pixel"], g["hallucination_auc"])
        corr_rows.append({"pair": pair_id, "spearman_photons_vs_auc": float(rho), "p": float(pval)})
        fig, ax = plt.subplots(figsize=(6, 4.5))
        gx = g.groupby("photons_per_pixel")["hallucination_auc"]
        ax.errorbar(gx.mean().index, gx.mean().values, yerr=gx.std().values, marker="o", capsize=3)
        ax.set_xscale("log")
        ax.set_xlabel("Signal photons per pixel")
        ax.set_ylabel("Unsupported-structure detection AUC")
        ax.set_ylim(0.4, 1.05)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(outdir / f"track2_{pair_id}_auc_vs_photons.png", dpi=180)
        plt.close(fig)
    pd.DataFrame(corr_rows).to_csv(outdir / "track2_correlations.csv", index=False)

    report = {


        "pairs": {k: v for k, v in PAIRS.items()},
        "track1_metrics": json.loads(t1.to_json(orient="records")),
        "track2_correlations": corr_rows,
        "track1_measured_detector_noise": True,
        "track2_controlled_degradation": True,
        "weights_frozen": True,

    }
    with open(outdir / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else (int(o) if isinstance(o, np.integer) else str(o)))
    print("saved:", outdir)


if __name__ == "__main__":
    main()
