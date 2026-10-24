from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import scipy.io as sio
import tifffile
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import richardson_lucy, denoise_tv_chambolle

from phyguard.optics.psf import MicroscopyPSFConfig, make_psf
from phyguard.optics.forward import PhotonMicroscope
from experiments.run_blind_calibration import blind_estimate, MODES
from experiments.run_real_optical_pilot import normalize


NA = 1.1
WAVELENGTH_NM = 515.0
PIXEL_NM = 86.0


GT_MODE_INDEX = {"astig_oblique": 0, "defocus": 1, "astig_vertical": 2,
                 "coma_y": 3, "coma_x": 4, "trefoil_y": 5}


def load_pair(crop: int = 256):

    base = ROOT / "data" / "cocoa_beads"
    ref = tifffile.imread(base / "ref_no_aberration.tif").astype(np.float32)
    aber = tifffile.imread(base / "aberrated_75nm.tif").astype(np.float32)
    gt = sio.loadmat(base / "gt_aberration.mat")["Modesset"].astype(np.float32)
    r = ref[len(ref) // 2]
    a = aber[len(aber) // 2]
    h, w = r.shape
    y0, x0 = (h - crop) // 2, (w - crop) // 2
    r = r[y0:y0 + crop, x0:x0 + crop]
    a = a[y0:y0 + crop, x0:x0 + crop]
    return r, a, gt


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--n-outer", type=int, default=20)
    ap.add_argument("--n-inner", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    outdir = ROOT / "results" / "real_aberration_validation"
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    ref_img, aber_img, gt = load_pair(args.crop)
    ref_n = normalize(ref_img, p_low=0.5, p_high=99.5)
    aber_n = normalize(aber_img, p_low=0.5, p_high=99.5)


    base = ROOT / "data" / "cocoa_beads"
    aber_full = tifffile.imread(base / "aberrated_75nm.tif").astype(np.float32)
    zi = aber_full.shape[0] // 2
    z_offsets = [-4, 0, 4]
    h, w = aber_full.shape[-2:]
    y0, x0 = (h - args.crop) // 2, (w - args.crop) // 2
    frames = [normalize(aber_full[zi + dz][y0:y0+args.crop, x0:x0+args.crop], p_low=0.5, p_high=99.5) for dz in z_offsets]

    photons = 30.0
    bg = max(0.2, 0.15 * photons)
    counts3 = torch.stack([torch.tensor(f * photons)[None, None] for f in frames], dim=0).to(device)

    nom_psf = make_psf(MicroscopyPSFConfig(
        wavelength_nm=WAVELENGTH_NM, na=NA, pixel_size_nm=PIXEL_NM, kernel_size=41,
    )).detach().cpu().numpy()
    rl0 = richardson_lucy(np.clip(frames[1], 0, 1), nom_psf, num_iter=15, clip=False)
    x_init = torch.tensor(normalize(np.clip(rl0, 0, None), p_low=0.1, p_high=99.7))[None, None].to(device)

    res = blind_estimate(
        counts3, x_init, modes=MODES, wavelength_nm=WAVELENGTH_NM, na=NA,
        pixel_size_nm=PIXEL_NM, signal_photons=photons, background_photons=bg,
        read_noise_std=0.15, tv_weight=0.08, n_outer=args.n_outer, n_inner=args.n_inner,
        lr_x=0.01, lr_theta=0.15, seed=0, defocus_offsets=[-0.15, 0.0, 0.15],
    )
    est = res["theta"]


    gt_flat = gt.flatten()
    gt_map = {name: float(gt_flat[i]) for name, i in GT_MODE_INDEX.items()}
    rows = []
    our_to_gt = {
        "defocus": gt_map["defocus"],
        "astig_x": gt_map["astig_vertical"],
        "coma_x": gt_map["coma_x"],
        "spherical": float("nan"),
    }
    print("\n=== Estimated vs GT (waves) ===")
    for m in MODES:
        g = our_to_gt[m]
        e = est[m]
        err = min(abs(e - g), abs(e + g)) if np.isfinite(g) else float("nan")
        print(f"  {m:10s} GT={g:+.4f}  est={e:+.4f}  sign-aware err={err:.4f}")
        rows.append({"mode": m, "gt": g, "estimated": e, "sign_aware_error": err})


    est_psf = make_psf(MicroscopyPSFConfig(
        wavelength_nm=WAVELENGTH_NM, na=NA, pixel_size_nm=PIXEL_NM, kernel_size=41,
        aberrations_waves=est,
    )).detach().cpu().numpy()

    def recon(psf_np, img):
        r = richardson_lucy(np.clip(img, 0, 1), psf_np, num_iter=15, clip=False)
        return normalize(np.clip(r, 0, None), p_low=0.1, p_high=99.7)

    r_nom = recon(nom_psf, aber_n)
    r_cal = recon(est_psf, aber_n)
    print("\n=== Reconstruction vs no-aberration reference ===")
    print(f"  aberrated(raw)  PSNR={peak_signal_noise_ratio(ref_n, aber_n, data_range=1.0):.2f} SSIM={structural_similarity(ref_n, aber_n, data_range=1.0):.3f}")
    print(f"  RL nominal      PSNR={peak_signal_noise_ratio(ref_n, r_nom, data_range=1.0):.2f} SSIM={structural_similarity(ref_n, r_nom, data_range=1.0):.3f}")
    print(f"  RL calibrated   PSNR={peak_signal_noise_ratio(ref_n, r_cal, data_range=1.0):.2f} SSIM={structural_similarity(ref_n, r_cal, data_range=1.0):.3f}")

    pd.DataFrame(rows).to_csv(outdir / "real_gt_comparison.csv", index=False)
    summary = {
        "optical": {"NA": NA, "wavelength_nm": WAVELENGTH_NM, "pixel_nm": PIXEL_NM},
        "gt_modesset_waves": gt.tolist(),
        "estimated_waves": est,
        "reconstruction": {
            "aberrated_psnr": float(peak_signal_noise_ratio(ref_n, aber_n, data_range=1.0)),
            "rl_nominal_psnr": float(peak_signal_noise_ratio(ref_n, r_nom, data_range=1.0)),
            "rl_calibrated_psnr": float(peak_signal_noise_ratio(ref_n, r_cal, data_range=1.0)),
        },

    }
    import json
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("saved:", outdir)


if __name__ == "__main__":
    main()
