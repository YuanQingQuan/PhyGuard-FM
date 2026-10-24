from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

import numpy as np
import pandas as pd
import scipy.io as sio
import tifffile
import torch

from phyguard.optics.psf import MicroscopyPSFConfig, make_psf
from experiments.run_blind_calibration import blind_estimate
from experiments.run_real_optical_pilot import normalize

NA, WAVEL_NM, PIXEL_NM = 1.1, 515.0, 86.0
N_OBJ = 1.333
DZ_UM = 0.2
Z_FRAMES = [-4, 0, 4]
MODES7 = ["defocus", "astig_x", "astig_y", "coma_x", "coma_y", "trefoil_y", "spherical"]

MODESET_UM = np.array([[-0.00726215, -0.03136094, -0.01923546],
                       [-0.02539367, -0.02130912, -0.01753091],
                       [-0.00508785, -0.02769500, 0.02476592]])
ROW_USED = 1


conv = 2.0 / (WAVEL_NM / 1000.0)
row = MODESET_UM[ROW_USED]
GT = {"astig_x": float(row[0] * conv), "trefoil_y": float(row[1] * conv),
      "coma_y": float(row[2] * conv)}

defocus_per_frame = (abs(Z_FRAMES[0]) * DZ_UM) * NA**2 / (4 * N_OBJ * (WAVEL_NM / 1000.0))
OFFSETS = [-defocus_per_frame, 0.0, defocus_per_frame]

print(f"GT (our units, waves): {GT}")
print(f"closure: 2*|row1| = {2 * np.linalg.norm(row) * 1000:.1f} nm (dataset label: 75 nm)")
print(f"z offsets {Z_FRAMES} -> defocus offsets {OFFSETS}")


def load_frames(name: str, crop: int = 256):
    st = tifffile.imread(ROOT / "data" / "cocoa_beads" / name).astype(np.float32)
    zi = st.shape[0] // 2
    h, w = st.shape[-2:]
    y0, x0 = (h - crop) // 2, (w - crop) // 2
    return [normalize(st[zi + dz][y0:y0 + crop, x0:x0 + crop], p_low=0.5, p_high=99.5)
            for dz in Z_FRAMES]


def estimate(frames, device, seed=0):
    from skimage.restoration import richardson_lucy
    photons = 30.0
    bg = max(0.2, 0.15 * photons)
    counts3 = torch.stack([torch.tensor(f * photons)[None, None] for f in frames], dim=0).to(device)
    nom = make_psf(MicroscopyPSFConfig(wavelength_nm=WAVEL_NM, na=NA,
                                       pixel_size_nm=PIXEL_NM, kernel_size=41)).cpu().numpy()
    rl0 = richardson_lucy(np.clip(frames[1], 0, 1), nom, num_iter=15, clip=False)
    x_init = torch.tensor(normalize(np.clip(rl0, 0, None), p_low=0.1, p_high=99.7))[None, None].to(device)
    res = blind_estimate(
        counts3, x_init, modes=MODES7, wavelength_nm=WAVEL_NM, na=NA,
        pixel_size_nm=PIXEL_NM, signal_photons=photons, background_photons=bg,
        read_noise_std=0.15, tv_weight=0.08, n_outer=20, n_inner=8,
        lr_x=0.01, lr_theta=0.15, seed=seed, defocus_offsets=OFFSETS,
    )
    return res["theta"]


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    outdir = ROOT / "results" / "real_aberration_validation_v2"
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
    df.to_csv(outdir / "cocoa_v2_comparison.csv", index=False)

    repeatability_um = MODESET_UM.std(axis=0)
    summary = {
        "gt_waves_our_basis": GT,
        "modeset_rows_um": MODESET_UM.tolist(),
        "dm_row_spread_um_1std": repeatability_um.tolist(),
        "dm_row_spread_waves_1std": (repeatability_um * conv).tolist(),
        "closure_75nm": float(2 * np.linalg.norm(row) * 1000),
        "defocus_offsets_waves": OFFSETS,
        "estimates": est,
        "differential": diff,

    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(df.to_string(index=False))
    print("saved:", outdir)


if __name__ == "__main__":
    main()
