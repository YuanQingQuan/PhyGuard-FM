from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import tifffile
import torch

from phyguard.optics.psf import MicroscopyPSFConfig
from phyguard.optics.psf3d import make_psf_3d, forward_3d_stack
from phyguard.likelihood import poisson_gaussian_nll
from experiments.run_real_optical_pilot import normalize

NA, WAVELENGTH_NM, PIXEL_NM = 1.1, 515.0, 86.0
MODES = ["defocus", "astig_x", "coma_x", "spherical"]


def estimate_3d(counts, structure, photons, bg, n_steps=400, lr=0.08):
    est = {m: torch.tensor(0.0, requires_grad=True, device=counts.device) for m in MODES}
    opt = torch.optim.Adam(list(est.values()), lr=lr)
    for _ in range(n_steps):
        opt.zero_grad()
        psf = make_psf_3d(MicroscopyPSFConfig(
            wavelength_nm=WAVELENGTH_NM, na=NA, pixel_size_nm=PIXEL_NM, kernel_size=41,
            aberrations_waves=est,
        ), nz=counts.shape[0], dz_nm=200.0, device=counts.device)
        mu = forward_3d_stack(structure, psf) * photons + bg
        loss = poisson_gaussian_nll(counts, mu, 0.15)
        loss.backward()
        opt.step()
    return {m: float(est[m].detach()) for m in MODES}


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--magnitudes", type=float, nargs="+", default=[0.05, 0.15, 0.30])
    ap.add_argument("--photons", type=float, default=30.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    outdir = ROOT / "results" / "calibration_3d"
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    vol = tifffile.imread(ROOT / "data" / "cocoa_beads" / "ref_no_aberration.tif").astype(np.float32)
    zi = vol.shape[0] // 2
    sub = vol[zi - 4:zi + 5, 160:288, 160:288]
    structure = torch.tensor(np.stack([normalize(sub[i], p_low=0.5, p_high=99.5) for i in range(9)])).to(device)
    photons = args.photons
    bg = max(0.2, 0.15 * photons)

    rows = []
    for mag in args.magnitudes:
        for seed in args.seeds:
            rng = np.random.default_rng(70 + seed)
            true = {m: float(rng.uniform(-mag, mag)) for m in MODES}
            psf_true = make_psf_3d(MicroscopyPSFConfig(
                wavelength_nm=WAVELENGTH_NM, na=NA, pixel_size_nm=PIXEL_NM, kernel_size=41,
                aberrations_waves=true,
            ), nz=9, dz_nm=200.0, device=device)
            mu = forward_3d_stack(structure, psf_true) * photons + bg
            torch.manual_seed(80 + seed)
            counts = torch.poisson(mu)
            est = estimate_3d(counts, structure, photons, bg)
            errs = {m: min(abs(est[m] - true[m]), abs(est[m] + true[m])) for m in MODES}
            row = {"magnitude": mag, "seed": seed,
                   "mean_err": float(np.mean(list(errs.values()))),
                   "max_err": float(np.max(list(errs.values())))}
            for m in MODES:
                row[f"err_{m}"] = errs[m]
            rows.append(row)
            print(row, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "calibration_3d.csv", index=False)
    g = df.groupby("magnitude").agg(
        mean_err=("mean_err", "mean"), max_err=("max_err", "mean"),
        err_defocus=("err_defocus", "mean"), err_astig=("err_astig_x", "mean"),
        err_coma=("err_coma_x", "mean"), err_sph=("err_spherical", "mean"),
    ).round(4)
    print(g.to_string())
    print("saved:", outdir)


if __name__ == "__main__":
    main()
