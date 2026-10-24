from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from phyguard.optics.psf import (
    MicroscopyPSFConfig,
    make_psf,
    psf_aberration_derivatives,
    psf_to_otf,
)
from phyguard.optics.zernike_ansi import ANSI_ESTIMATED, ANSI_NAME, LAMBDA_UM

try:
    from experiments.run_phasediversity_external import DATASETS, load_iteration, _find_data_root
except ImportError:
    from run_phasediversity_external import DATASETS, load_iteration, _find_data_root  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
MODES = tuple(ANSI_NAME[j] for j in ANSI_ESTIMATED)


def frame_config(theta: dict, offset: dict, n: int, pixel_size_nm: float, na: float):
    combined = dict(theta)
    for k, v in offset.items():
        combined[k] = combined.get(k, 0.0) + v
    return MicroscopyPSFConfig(
        wavelength_nm=LAMBDA_UM * 1000.0, na=na, pixel_size_nm=pixel_size_nm,
        kernel_size=n, pupil_grid="exact", aberrations_waves=combined,
    )


def conv_circular(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:

    otf = psf_to_otf(h, x.shape[-2:])
    return torch.fft.ifft2(torch.fft.fft2(x) * otf).real


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="beads")
    ap.add_argument("--iteration", type=int, default=1)
    ap.add_argument("--pixel-size-nm", type=float, default=104.0)
    ap.add_argument("--na", type=float, default=1.2)
    ap.add_argument("--results", default=str(ROOT / "results" / "phasediversity_external"))
    ap.add_argument("--outdir", default=str(ROOT / "results" / "phasediversity_external"))
    args = ap.parse_args()

    tag = f"{args.dataset}_it{args.iteration}"
    res = Path(args.results)
    theta_est_vec = np.load(res / f"{tag}_theta_est.npy")
    x_est = np.load(res / f"{tag}_x_est.npy")
    meta = json.loads((res / f"{tag}.json").read_text())
    offsets = meta["diversity_offsets"]
    theta_est = dict(zip(meta["modes"], theta_est_vec.tolist()))

    counts, coeffs, _ = load_iteration(args.dataset, args.iteration)
    from phyguard.optics.zernike_ansi import ansi_coeffs_to_waves
    truth = ansi_coeffs_to_waves(
        {k: float(v) for k, v in enumerate(coeffs[0], start=1) if v != 0})
    theta_cmd = {m: truth.get(m, 0.0) for m in MODES}

    n = counts.shape[-1]
    x = torch.as_tensor(x_est, dtype=torch.float64)[None, None]
    raw = torch.as_tensor(counts[0], dtype=torch.float64)
    s = float(meta["signal_photons"])
    b = float(meta["background_photons"])

    F = torch.zeros((len(MODES), len(MODES)), dtype=torch.float64)
    per_frame_lam = []
    for off in offsets:
        cfg = frame_config(theta_est, off, n, args.pixel_size_nm, args.na)
        h = make_psf(cfg, dtype=torch.float64)
        lam = (s * conv_circular(x, h) + b).clamp_min(1e-6)
        derivs = psf_aberration_derivatives(cfg, MODES, dtype=torch.float64)
        U = torch.stack([
            s * conv_circular(x, derivs[m]).reshape(-1) for m in MODES
        ])
        w = (1.0 / lam).reshape(-1)
        F += (U * w) @ U.T
        per_frame_lam.append(float(lam.mean()))

    F = 0.5 * (F + F.T)
    evals, evecs = torch.linalg.eigh(F)
    order = torch.argsort(evals, descending=True)
    evals = evals[order]
    evecs = evecs[:, order]
    lam1 = float(evals[0])

    dtheta = torch.tensor([theta_est[m] - theta_cmd[m] for m in MODES],
                          dtype=torch.float64)
    proj = evecs.T @ dtheta

    print(f"{tag}: per-frame mean lambda {[round(v,1) for v in per_frame_lam]}")
    print(f"signal_photons={s:.1f} background_photons={b:.1f}")
    print(f"||dtheta|| = {float(dtheta.norm()):.4f} waves "
          f"({float(dtheta.norm())*LAMBDA_UM*1000:.1f} nm total)")
    print(f"\ncondition number kappa(F) [unregularised] = {lam1/float(evals[-1]):.3e}")
    for thr in (1e-2, 1e-3, 1e-4, 1e-6):
        k = int((evals / lam1 < thr).sum())
        print(f"  directions with lambda_k/lambda_1 < {thr:g}: {k}/18")

    print(f"\n{'k':>3s} {'lambda_k/lambda_1':>18s} {'|a_k| (nm)':>11s} "
          f"{'share of ||dtheta||^2':>22s}  dominant mode of v_k")
    tot = float((proj ** 2).sum())
    for k in range(len(MODES)):
        share = float(proj[k] ** 2) / max(tot, 1e-30)
        v = evecs[:, k]
        dom = MODES[int(torch.argmax(v.abs()))]
        print(f"{k+1:3d} {float(evals[k])/lam1:18.3e} "
              f"{abs(float(proj[k]))*LAMBDA_UM*1000:11.1f} {share:21.3f}   {dom}")


    import math
    a = (proj.abs() * LAMBDA_UM * 1000).numpy()
    pred = (1.0 / torch.sqrt(evals.clamp_min(1e-300))).numpy()
    def spearman(u, v):
        ru = np.argsort(np.argsort(u)).astype(float)
        rv = np.argsort(np.argsort(v)).astype(float)
        ru -= ru.mean(); rv -= rv.mean()
        return float((ru @ rv) / math.sqrt((ru @ ru) * (rv @ rv)))
    print(f"\nSpearman(|a_k|, 1/sqrt(lambda_k)) = {spearman(a, pred):+.3f}")
    print(f"Spearman(|a_k|, lambda_k)         = {spearman(a, evals.numpy()):+.3f}")

    weak = (evals / lam1 < 1e-3)
    print(f"\nerror energy in lambda_k/lambda_1 < 1e-3 : "
          f"{float((proj[weak]**2).sum())/max(tot,1e-30):.3f}")
    print(f"error energy in lambda_k/lambda_1 < 1e-2 : "
          f"{float((proj[evals/lam1 < 1e-2]**2).sum())/max(tot,1e-30):.3f}")

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{tag}_fisher.npz", F=F.numpy(), evals=evals.numpy(),
             evecs=evecs.numpy(), modes=np.array(MODES), proj=proj.numpy(),
             dtheta=dtheta.numpy())
    print(f"\nwrote {out / (tag + '_fisher.npz')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
