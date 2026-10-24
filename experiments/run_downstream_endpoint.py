from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.feature import peak_local_max
from skimage.restoration import richardson_lucy

from phyguard.optics.forward import PhotonMicroscope
from phyguard.optics.psf import MicroscopyPSFConfig, make_psf

from experiments.run_blind_calibration import blind_estimate, MODES
from experiments.run_biosr_identifiability_sweep import load_gt
from experiments.run_biotisr_external_test import load_gt_frame
from experiments.run_real_optical_pilot import normalize

TRUE_ABERRATION = {"defocus": 0.05, "astig_x": 0.08, "coma_x": 0.08, "spherical": 0.03}
STRONG_ABERRATION = {"defocus": 0.10, "astig_x": 0.12, "coma_x": 0.12, "spherical": 0.06}
OFFSETS = [-1.0, -0.5, 0.0, 0.5, 1.0]
PHOTONS = 30.0


def simulate_sequence(gt: torch.Tensor, photons: float, seed: int, device,
                      aberration: dict | None = None) -> tuple[torch.Tensor, float]:
    bg = max(0.2, 0.15 * photons)
    ab = aberration or TRUE_ABERRATION
    counts = []
    for off in OFFSETS:
        th = dict(ab); th["defocus"] = th["defocus"] + off
        psf = make_psf(MicroscopyPSFConfig(
            wavelength_nm=520.0, na=1.2, pixel_size_nm=62.6,
            kernel_size=41, aberrations_waves=th,
        )).to(device)
        mic = PhotonMicroscope(psf=psf, signal_photons=photons,
                               background_photons=bg, read_noise_std=0.15)
        torch.manual_seed(300 + seed)
        counts.append(mic.sample(gt)["counts"])
    return torch.cat(counts, dim=0), bg


def psf_from(coeffs: dict | None) -> np.ndarray:
    cfg = MicroscopyPSFConfig(
        wavelength_nm=520.0, na=1.2, pixel_size_nm=62.6, kernel_size=41,
        aberrations_waves=coeffs or {},
    )
    return make_psf(cfg).detach().cpu().numpy()


def deconv(obs_counts: torch.Tensor, psf_np: np.ndarray, photons: float, bg: float) -> np.ndarray:
    img = ((obs_counts[0, 0] - bg) / photons).clamp(0, 1).cpu().numpy().astype(np.float32)
    rl = richardson_lucy(np.clip(img, 0, None), psf_np, num_iter=20, clip=False)
    return normalize(np.clip(rl, 0, None), p_low=0.1, p_high=99.7)


def segment_dice(gt_img: np.ndarray, test_img: np.ndarray) -> float:
    g = gaussian_filter(gt_img, 1.0) > threshold_otsu(gaussian_filter(gt_img, 1.0))
    t = gaussian_filter(test_img, 1.0) > threshold_otsu(gaussian_filter(test_img, 1.0))
    inter = float(np.logical_and(g, t).sum())
    return 2.0 * inter / max(float(g.sum() + t.sum()), 1.0)


def detect_spots(img: np.ndarray, min_dist: int = 3) -> np.ndarray:
    smooth = gaussian_filter(img, 0.8)
    thr = threshold_otsu(smooth)
    return peak_local_max(smooth, min_distance=min_dist, threshold_abs=thr)


def spot_metrics(gt_img: np.ndarray, test_img: np.ndarray) -> dict:
    from scipy.spatial import cKDTree
    sp_gt, sp_te = detect_spots(gt_img), detect_spots(test_img)
    if len(sp_gt) == 0:
        return {"n_gt": 0, "n_test": len(sp_te), "count_err": len(sp_te), "f1": np.nan, "loc_err": np.nan}
    if len(sp_te) == 0:
        return {"n_gt": len(sp_gt), "n_test": 0, "count_err": len(sp_gt), "f1": 0.0, "loc_err": np.nan}
    d, idx = cKDTree(sp_te).query(sp_gt, distance_upper_bound=3.0)
    matched = d < 3.0
    tp = int(matched.sum())
    prec = tp / len(sp_te)
    rec = tp / len(sp_gt)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return {"n_gt": len(sp_gt), "n_test": len(sp_te),
            "count_err": abs(len(sp_te) - len(sp_gt)), "f1": float(f1),
            "loc_err": float(np.mean(d[matched])) if tp else np.nan}


def run_case(gt_np: np.ndarray, task: str, seed: int, device,
             aberration: dict | None = None) -> dict:
    ab = aberration or TRUE_ABERRATION
    gt = torch.tensor(gt_np)[None, None].to(device)
    counts, bg = simulate_sequence(gt, PHOTONS, seed, device, ab)

    mid = counts[len(OFFSETS) // 2: len(OFFSETS) // 2 + 1]
    obs_img = ((mid[0, 0] - bg) / PHOTONS).clamp(0, 1).cpu().numpy().astype(np.float32)


    rl0 = richardson_lucy(np.clip(obs_img, 0, None), psf_from(None), num_iter=15, clip=False)
    x_init = torch.tensor(normalize(np.clip(rl0, 0, None), p_low=0.1, p_high=99.7))[None, None].to(device)

    res = blind_estimate(
        counts, x_init, modes=MODES, wavelength_nm=520.0, na=1.2,
        pixel_size_nm=62.6, signal_photons=PHOTONS, background_photons=bg,
        read_noise_std=0.15, tv_weight=0.10, n_outer=15, n_inner=6,
        lr_x=0.01, lr_theta=0.2, seed=seed, defocus_offsets=OFFSETS,
    )
    est = res["theta"]

    x_uncal = deconv(mid, psf_from(None), PHOTONS, bg)
    x_cal = deconv(mid, psf_from(est), PHOTONS, bg)
    obs_norm = normalize(np.clip(obs_img, 0, None), p_low=0.1, p_high=99.7)

    row = {"task": task, "seed": seed, "aberration": str(ab)}
    for m in MODES:
        row[f"err_{m}"] = min(abs(est[m] - ab[m]), abs(est[m] + ab[m]))
    if task == "segmentation":
        row["metric_obs"] = segment_dice(gt_np, obs_norm)
        row["metric_uncal"] = segment_dice(gt_np, x_uncal)
        row["metric_cal"] = segment_dice(gt_np, x_cal)
        row["metric_name"] = "dice"
    else:
        for tag, img in [("obs", obs_norm), ("uncal", x_uncal), ("cal", x_cal)]:
            sm = spot_metrics(gt_np, img)
            row[f"f1_{tag}"] = sm["f1"]
            row[f"counterr_{tag}"] = sm["count_err"]
            row[f"locerr_{tag}"] = sm["loc_err"]
        row["metric_name"] = "spot_f1"
    return row


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ccp-cell", type=int, default=2)
    ap.add_argument("--ccp-frame", type=int, default=10)
    ap.add_argument("--strong", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    outdir = ROOT / "results" / "downstream_endpoint"
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rows = []

    ab = STRONG_ABERRATION if args.strong else TRUE_ABERRATION
    gt_mt = load_gt("MT", 1, 1)
    for seed in args.seeds:
        r = run_case(gt_mt, "segmentation", seed, device, ab)
        rows.append(r)
        print(r, flush=True)

    gt_ccp = load_gt_frame(args.ccp_cell, args.ccp_frame)
    for seed in args.seeds:
        r = run_case(gt_ccp, "spots", seed, device, ab)
        rows.append(r)
        print(r, flush=True)

    df = pd.DataFrame(rows)
    name = "downstream_endpoint_strong.csv" if args.strong else "downstream_endpoint.csv"
    if args.tag:
        name = f"downstream_endpoint_{args.tag}.csv"
    df.to_csv(outdir / name, index=False)
    for task, g in df.groupby("task"):
        cols = [c for c in g.columns if c.startswith(("f1_", "counterr_", "locerr_", "err_"))
                or (c.startswith("metric_") and c != "metric_name")]
        print(f"\n[{task}] mean +- sd over {len(g)} seeds")
        print(g[cols].agg(["mean", "std"]).T.round(4).to_string())
    print("saved:", outdir)


if __name__ == "__main__":
    main()
