from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import uniform_filter
from skimage import data
from skimage.filters import unsharp_mask
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import richardson_lucy, denoise_tv_chambolle
from skimage.transform import resize

from phyguard.identifiability import convolution_identifiability
from phyguard.optics.forward import PhotonMicroscope
from phyguard.optics.psf import MicroscopyPSFConfig, make_psf
from phyguard.physics_stats import estimate_blind_stats
from phyguard.verification import verify_and_fuse, measurement_support_map


def normalize(x: np.ndarray, p_low=0.5, p_high=99.5) -> np.ndarray:
    x = np.asarray(x, np.float32)
    lo, hi = np.percentile(x, [p_low, p_high])
    return np.clip((x - lo) / max(hi - lo, 1e-8), 0, 1).astype(np.float32)


def get_real_microscopy() -> tuple[np.ndarray, dict]:


    x = normalize(data.cell())

    y0, x0, h, w = 170, 120, 384, 384
    crop = x[y0:y0+h, x0:x0+w]
    crop = resize(crop, (256, 256), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
    meta = {
        "name": "skimage.data.cell",
        "modality": "quantitative phase microscopy (real measurement)",

        "native_pixel_spacing_um": 0.107,
        "license": "CC0 / public domain",
        "reference": "Müller et al., Optics Express 26(8):10729-10743 (2018), DOI 10.1364/OE.26.010729",

    }
    return crop, meta


def best_patch_locations(x: np.ndarray, patch=28, margin=12) -> tuple[tuple[int, int], tuple[int, int]]:

    mean = uniform_filter(x.astype(np.float32), size=patch, mode="reflect")
    r = patch // 2
    valid = np.ones_like(x, bool)
    valid[:r+margin] = False
    valid[-(r+margin):] = False
    valid[:, :r+margin] = False
    valid[:, -(r+margin):] = False
    bright = mean.copy(); bright[~valid] = -np.inf
    sy, sx = np.unravel_index(np.argmax(bright), bright.shape)


    yy, xx = np.indices(x.shape)
    far = (yy - sy) ** 2 + (xx - sx) ** 2 > (patch * 3.0) ** 2
    dark_valid = valid & far
    dark = mean.copy(); dark[~dark_valid] = np.inf
    dy, dx = np.unravel_index(np.argmin(dark), dark.shape)
    return (sy - r, sx - r), (dy - r, dx - r)


def inject_plausible_duplicate(x: np.ndarray, patch=28, strength=0.9):
    src, dst = best_patch_locations(x, patch=patch)
    sy, sx = src; dy, dx = dst
    source = x[sy:sy+patch, sx:sx+patch].copy()

    t = np.hanning(patch).astype(np.float32)
    win = np.outer(t, t)
    candidate = x.copy()
    before = candidate[dy:dy+patch, dx:dx+patch].copy()
    proposal = np.maximum(before, strength * source * win)
    candidate[dy:dy+patch, dx:dx+patch] = proposal
    mask = np.zeros_like(x, bool)
    delta = proposal - before
    mask[dy:dy+patch, dx:dx+patch] = delta > 0.06
    return np.clip(candidate, 0, 1).astype(np.float32), mask, {"source_yx": src, "dest_yx": dst, "patch": patch}


def auc_rank(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool).ravel()
    scores = scores.astype(float).ravel()
    pos = np.flatnonzero(labels)
    neg = np.flatnonzero(~labels)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty_like(scores, dtype=float)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg_rank
        i = j
    sum_pos = ranks[pos].sum()
    u = sum_pos - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def masked_mse(gt, est, mask):
    if not np.any(mask):
        return 0.0
    return float(np.mean((est[mask] - gt[mask]) ** 2))


def bootstrap_ci(values, seed=2026, n_boot=5000, alpha=0.05):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def main():
    outdir = ROOT / "results" / "real_optical_pilot"
    outdir.mkdir(parents=True, exist_ok=True)
    datadir = ROOT / "data" / "real_optical_pilot"
    datadir.mkdir(parents=True, exist_ok=True)

    gt, meta = get_real_microscopy()
    np.save(datadir / "cell_qpi_real.npy", gt)
    with open(datadir / "source_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


    effective_pixel_nm = meta["native_pixel_spacing_um"] * 1000.0 * (384 / 256)
    psf_cfg = MicroscopyPSFConfig(
        wavelength_nm=520.0,
        na=1.20,
        pixel_size_nm=effective_pixel_nm,
        kernel_size=41,
        aberrations_waves={"defocus": 0.12},
    )
    psf = make_psf(psf_cfg)
    ident = convolution_identifiability(psf, gt.shape, gamma=0.05).to_dict()

    foundation_clean = unsharp_mask(gt, radius=0.8, amount=0.35, preserve_range=True).astype(np.float32)
    foundation_hall, halluc_mask, inj = inject_plausible_duplicate(foundation_clean, patch=28, strength=0.95)
    halluc_area = float(halluc_mask.mean())

    photon_levels = [0.5, 1, 2, 5, 10, 20, 50]
    seeds = [3, 11, 29, 47, 71, 97, 131, 173, 223, 281]
    rows = []
    representative = None

    for photons in photon_levels:
        for seed in seeds:
            torch.manual_seed(seed)
            mic = PhotonMicroscope(
                psf=psf,
                signal_photons=float(photons),
                background_photons=max(0.2, 0.15 * photons),
                read_noise_std=0.15,
            )
            obs = mic.sample(torch.tensor(gt)[None, None])
            counts = obs["counts"]
            obs_img = obs["normalized"][0, 0].detach().numpy().astype(np.float32)


            rl = richardson_lucy(np.clip(obs_img, 0, 1), psf.detach().cpu().numpy(), num_iter=12, clip=False)
            rl = normalize(np.clip(rl, 0, None), p_low=0.1, p_high=99.7)
            physics_candidate = denoise_tv_chambolle(rl, weight=0.08, channel_axis=None).astype(np.float32)

            tf = torch.tensor(foundation_hall)[None, None]
            tp = torch.tensor(physics_candidate)[None, None]
            ver = verify_and_fuse(
                counts, tf, tp, mic,
                local_window=9,
                temperature=0.70,
                conservative_bias=0.05,
            )
            support = np.clip(ver.support_foundation[0, 0].detach().numpy(), 0, 1)


            tail_quantile = 0.005
            gate_scale = 0.02
            tau = float(np.quantile(support, tail_quantile))
            keep_prior = 1.0 / (1.0 + np.exp(-(support - tau) / gate_scale))
            fused = np.clip(keep_prior * foundation_hall + (1.0 - keep_prior) * physics_candidate, 0, 1)


            detection_auc = auc_rank(halluc_mask, 1.0 - support)
            support_in = float(support[halluc_mask].mean())
            support_out = float(support[~halluc_mask].mean())

            stats = estimate_blind_stats(obs["normalized"][0, 0]).to_dict()
            row = {
                "photons_per_pixel": photons,
                "seed": seed,
                "obs_psnr": peak_signal_noise_ratio(gt, obs_img, data_range=1.0),
                "obs_ssim": structural_similarity(gt, obs_img, data_range=1.0),
                "rl_psnr": peak_signal_noise_ratio(gt, rl, data_range=1.0),
                "rl_ssim": structural_similarity(gt, rl, data_range=1.0),
                "physics_psnr": peak_signal_noise_ratio(gt, physics_candidate, data_range=1.0),
                "physics_ssim": structural_similarity(gt, physics_candidate, data_range=1.0),
                "foundation_psnr": peak_signal_noise_ratio(gt, foundation_hall, data_range=1.0),
                "foundation_ssim": structural_similarity(gt, foundation_hall, data_range=1.0),
                "fused_psnr": peak_signal_noise_ratio(gt, fused, data_range=1.0),
                "fused_ssim": structural_similarity(gt, fused, data_range=1.0),
                "hallucination_auc": detection_auc,
                "support_inside_hallucination": support_in,
                "support_outside_hallucination": support_out,
                "support_drop": support_out - support_in,
                "tail_gate_tau": tau,
                "keep_prior_inside_hallucination": float(keep_prior[halluc_mask].mean()),
                "keep_prior_outside_hallucination": float(keep_prior[~halluc_mask].mean()),
                "foundation_hallucination_mse": masked_mse(gt, foundation_hall, halluc_mask),
                "fused_hallucination_mse": masked_mse(gt, fused, halluc_mask),
                "rl_hallucination_mse": masked_mse(gt, rl, halluc_mask),
                "physics_hallucination_mse": masked_mse(gt, physics_candidate, halluc_mask),
                "blind_snr_proxy_db": stats["snr_proxy_db"],
                "blind_hf_noise_std": stats["high_frequency_noise_std"],
                "background_photons": mic.background_photons,
            }
            rows.append(row)

            if photons == 5 and seed == 11:
                representative = {
                    "obs": obs_img,
                    "rl": physics_candidate,
                    "foundation": foundation_hall,
                    "support": support,
                    "fused": fused,
                }

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "per_run_metrics.csv", index=False)
    summary = df.groupby("photons_per_pixel").agg(["mean", "std"])
    summary.to_csv(outdir / "summary_by_photons.csv")


    compact = []
    for photons, g in df.groupby("photons_per_pixel"):
        reduction = g.foundation_hallucination_mse.to_numpy() / np.maximum(g.fused_hallucination_mse.to_numpy(), 1e-12)
        auc_ci = bootstrap_ci(g.hallucination_auc.to_numpy(), seed=2026 + int(10 * photons))
        reduction_ci = bootstrap_ci(reduction, seed=3026 + int(10 * photons))
        compact.append({
            "photons_per_pixel": float(photons),
            "n_seeds": int(len(g)),
            "hallucination_auc_mean": float(g.hallucination_auc.mean()),
            "hallucination_auc_std": float(g.hallucination_auc.std()),
            "hallucination_auc_ci95": list(auc_ci),
            "support_drop_mean": float(g.support_drop.mean()),
            "mse_reduction_factor_mean": float(reduction.mean()),
            "mse_reduction_factor_ci95": list(reduction_ci),
            "fused_psnr_mean": float(g.fused_psnr.mean()),
            "fused_psnr_std": float(g.fused_psnr.std()),
            "rl_psnr_mean": float(g.rl_psnr.mean()),
        })
    report = {
        "source": meta,
        "effective_pixel_nm_after_crop_resize": effective_pixel_nm,
        "optics": {
            "wavelength_nm": psf_cfg.wavelength_nm,
            "na": psf_cfg.na,
            "defocus_waves": psf_cfg.aberrations_waves["defocus"],
            "gamma": 0.05,
            "identifiability": ident,
        },
        "controlled_hallucination": {**inj, "area_fraction": halluc_area},
        "verification_gate": {"tail_quantile": 0.005, "sigmoid_scale": 0.02, "uses_hallucination_mask": False},
        "summary": compact,
        "controlled_degradation": True,
        "measured_aberration_ground_truth": False,

    }
    with open(outdir / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=lambda o: int(o) if isinstance(o, np.integer) else (float(o) if isinstance(o, np.floating) else str(o)))


    if representative is None:
        raise RuntimeError("representative run missing")
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    panels = [
        (gt, "Real microscopy image\n(controlled GT)"),
        (representative["obs"], "5 photons/pixel\nmeasurement"),
        (representative["rl"], "RL + TV\n(training-free physics)"),
        (representative["foundation"], "Prior proposal\n+ duplicated real structure"),
        (representative["support"], "Measurement support\nfor prior proposal"),
        (representative["fused"], "PhyGuard\nmeasurement-grounded fusion"),
    ]
    for ax, (im, title) in zip(axes.ravel(), panels):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(outdir / "real_microscopy_representative.png", dpi=180)
    plt.close(fig)


    g = df.groupby("photons_per_pixel")
    x = np.array(photon_levels, dtype=float)
    auc_m = g.hallucination_auc.mean().reindex(photon_levels).values
    auc_s = g.hallucination_auc.std().reindex(photon_levels).values
    drop_m = g.support_drop.mean().reindex(photon_levels).values
    factor_m = (g.apply(lambda z: np.mean(z.foundation_hallucination_mse / np.maximum(z.fused_hallucination_mse, 1e-12)), include_groups=False)
                  .reindex(photon_levels).values)

    fig = plt.figure(figsize=(7.2, 4.8))
    ax = fig.add_subplot(111)
    ax.errorbar(x, auc_m, yerr=auc_s, marker="o", capsize=3)
    ax.set_xscale("log")
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("Signal photons per pixel")
    ax.set_ylabel("Unsupported-structure detection AUC")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "hallucination_auc_vs_photons.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(7.2, 4.8))
    ax = fig.add_subplot(111)
    ax.plot(x, factor_m, marker="o")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Signal photons per pixel")
    ax.set_ylabel("Hallucination MSE reduction factor")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "hallucination_reduction_vs_photons.png", dpi=180)
    plt.close(fig)

    print(json.dumps(report, indent=2, default=lambda o: int(o) if isinstance(o, np.integer) else (float(o) if isinstance(o, np.floating) else str(o))))
    print("saved:", outdir)


if __name__ == "__main__":
    main()
