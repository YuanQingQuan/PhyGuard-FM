from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

import numpy as np
import pandas as pd
import torch

from phyguard.optics.psf import MicroscopyPSFConfig, make_psf
from experiments.run_blind_calibration import blind_estimate
from experiments.run_real_optical_pilot import normalize
from experiments.run_cocoa_validation_v2 import (
    NA, WAVEL_NM, PIXEL_NM, N_OBJ, DZ_UM, Z_FRAMES, MODES7,
    MODESET_UM, ROW_USED, GT, load_frames,
)

AXIAL_OFFSETS_UM = [dz * DZ_UM for dz in Z_FRAMES]


def estimate(frames, device, seed=0):
    from skimage.restoration import richardson_lucy
    import tifffile  # noqa
    photons = 30.0
    bg = max(0.2, 0.15 * photons)
    counts3 = torch.stack([torch.tensor(f * photons)[None, None] for f in frames], dim=0).to(device)
    nom = make_psf(MicroscopyPSFConfig(wavelength_nm=WAVEL_NM, na=NA,
                                       pixel_size_nm=PIXEL_NM, kernel_size=41,
                                       n_obj=N_OBJ)).cpu().numpy()
    rl0 = richardson_lucy(np.clip(frames[1], 0, 1), nom, num_iter=15, clip=False)
    x_init = torch.tensor(normalize(np.clip(rl0, 0, None), p_low=0.1, p_high=99.7))[None, None].to(device)
    res = blind_estimate(
        counts3, x_init, modes=MODES7, wavelength_nm=WAVEL_NM, na=NA,
        pixel_size_nm=PIXEL_NM, signal_photons=photons, background_photons=bg,
        read_noise_std=0.15, tv_weight=0.08, n_outer=20, n_inner=8,
        lr_x=0.01, lr_theta=0.15, seed=seed,
        axial_offsets_um=AXIAL_OFFSETS_UM, n_obj=N_OBJ,
    )
    return res["theta"]


def main():
    import tifffile  # noqa: F401  (used by load_frames)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    outdir = ROOT / "results" / "real_aberration_validation_v3"
    outdir.mkdir(parents=True, exist_ok=True)

    est = {}
    for tag, fname in [("aberrated", "aberrated_75nm.tif"), ("reference", "ref_no_aberration.tif")]:
        frames = load_frames(fname)
        theta = estimate(frames, device)
        est[tag] = {m: float(theta[m]) for m in MODES7}
        print(f"[{tag}] estimate: {est[tag]}", flush=True)

    diff = {m: est["aberrated"][m] - est["reference"][m] for m in MODES7}
    rows = []
    for m in MODES7:
        gt = GT.get(m, 0.0)
        rows.append({
            "mode": m, "gt_waves": gt,
            "est_aberrated": est["aberrated"][m],
            "est_reference": est["reference"][m],
            "est_differential": diff[m],
            "signed_err_raw": abs(est["aberrated"][m] - gt),
            "signed_err_differential": abs(diff[m] - gt),
        })
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "cocoa_v3_comparison.csv", index=False)
    with open(outdir / "summary.json", "w") as f:
        json.dump({"gt_waves_our_basis": GT,
                   "axial_offsets_um": AXIAL_OFFSETS_UM,
                   "axial_model": "exact (z/lambda)(sqrt(n^2-(NA rho)^2)-n)",
                   "estimates": est, "differential": diff}, f, indent=2)
    print(df.to_string(index=False))
    print("saved:", outdir)


if __name__ == "__main__":
    main()
