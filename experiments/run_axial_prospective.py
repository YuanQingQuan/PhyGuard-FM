from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import tifffile

from phyguard.optics.psf import MicroscopyPSFConfig, _zernike_phase, _zernike_basis, _pupil_coords
from phyguard.optics.psf3d import make_psf_3d, forward_3d_stack
from phyguard.likelihood import poisson_gaussian_nll
from phyguard.estimation.info import crb_identifiable
from experiments.run_real_optical_pilot import normalize

NA, WAVELENGTH_NM, PIXEL_NM, NZ, DZ_NM, KERNEL = 1.1, 515.0, 86.0, 9, 200.0, 41
MODES = ["defocus", "astig_x", "coma_x", "spherical"]

SIGMA = np.array([1 / np.sqrt(3), 1 / np.sqrt(6), 1 / np.sqrt(8), 1 / np.sqrt(5)])
THRESHOLD = 0.05
PHOTONS, BG, READ = 30.0, 4.5, 0.15
MAGS = [0.05, 0.15, 0.30, 0.50, 0.75, 1.00]
SEEDS = [0, 1, 2, 3, 4]
OUTDIR = ROOT / "results" / "axial_prospective"


def pupil_stack(coeffs_waves, *, n=KERNEL, device="cpu", dtype=torch.float64):

    fy = torch.fft.fftfreq(n, d=PIXEL_NM, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(fy, fy, indexing="ij")
    rr = torch.sqrt(xx**2 + yy**2)
    cutoff = NA / WAVELENGTH_NM
    rho = rr / cutoff
    theta = torch.atan2(yy, xx)
    aperture = (rho <= 1.0).to(dtype)
    phase = _zernike_phase(rho, theta, coeffs_waves)

    inside = 1.518 ** 2 - (WAVELENGTH_NM * rr) ** 2
    kz = (2.0 * torch.pi / WAVELENGTH_NM) * torch.sqrt(inside.clamp_min(0.0))
    zs = (torch.arange(NZ, device=device, dtype=dtype) - NZ // 2) * DZ_NM
    prop = torch.exp(1j * (kz[None] * zs[:, None, None]).to(
        torch.complex128 if dtype == torch.float64 else torch.complex64))
    pupil = (aperture * torch.exp(1j * phase)).to(prop.dtype)[None] * prop
    return pupil, _zernike_basis(rho, theta), aperture


def psf_and_derivatives(coeffs_waves, *, n=KERNEL):

    pupil, basis, _ = pupil_stack(coeffs_waves, n=n)
    cplx = pupil.dtype
    amp = torch.fft.ifft2(pupil)
    raw = torch.abs(amp) ** 2
    s = raw.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-300)
    derivs = {}
    for m in MODES:
        mod = (1j * 2.0 * torch.pi * basis[m]).to(cplx)[None] * pupil
        damp = torch.fft.ifft2(mod)
        draw = 2.0 * torch.real(torch.conj(amp) * damp)
        dh = (draw * s - raw * draw.sum(dim=(-2, -1), keepdim=True)) / s**2
        derivs[m] = torch.fft.fftshift(dh, dim=(-2, -1))
    psf = torch.fft.fftshift(raw / s, dim=(-2, -1))
    return psf, derivs


def axial_fisher(structure_3d, coeffs_waves, *, photons=PHOTONS, bg=BG):

    psf, derivs = psf_and_derivatives({m: float(coeffs_waves[m]) for m in MODES})
    x = structure_3d.to(torch.float64)
    kh = psf.shape[-1]
    pad = kh // 2
    import torch.nn.functional as tF
    mu0 = forward_3d_stack(x, psf.to(x.dtype))
    lam = (mu0.clamp_min(0.0) * photons + bg).clamp_min(1e-6)
    mask = (mu0 > 0).to(x.dtype)
    dmu = []
    for m in MODES:
        d = forward_3d_stack(x, derivs[m].to(x.dtype)) * mask * photons
        dmu.append(d)
    lam_np = lam.cpu().numpy()
    dmu_np = [d.cpu().numpy() for d in dmu]
    k = len(MODES)
    F = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            F[i, j] = F[j, i] = float(np.sum(dmu_np[i] * dmu_np[j] / lam_np))
    return F


def load_volume():
    vol = tifffile.imread(ROOT / "data" / "cocoa_beads" / "ref_no_aberration.tif").astype(np.float32)
    zi = vol.shape[0] // 2
    sub = vol[zi - 4:zi + 5, 160:288, 160:288]
    return torch.tensor(np.stack([normalize(sub[i], p_low=0.5, p_high=99.5) for i in range(NZ)]))


def draw_true(mag, seed):
    rng = np.random.default_rng(70 + seed)
    return {m: float(rng.uniform(-mag, mag)) for m in MODES}


def rms_of(coeffs):
    return float(np.sqrt(sum(coeffs[m] ** 2 * SIGMA[i] ** 2 for i, m in enumerate(MODES))))


def crossing(x, y, thresh):

    order = np.argsort(x)
    lx, ly = np.log10(np.asarray(x)[order]), np.log10(np.maximum(np.asarray(y)[order], 1e-9))
    lt = np.log10(thresh)
    for i in range(len(lx) - 1):
        if (ly[i] - lt) * (ly[i + 1] - lt) <= 0 and ly[i] != ly[i + 1]:
            return float(10 ** (lx[i] + (lt - ly[i]) * (lx[i + 1] - lx[i]) / (ly[i + 1] - ly[i])))
    return None


def validate_derivative(tol=2e-6):

    base = {"defocus": -0.07, "astig_x": 0.11, "coma_x": 0.05, "spherical": -0.09}
    _, derivs = psf_and_derivatives(base)
    eps = 1e-4
    worst = 0.0
    for m in MODES:
        up = dict(base); up[m] += eps
        dn = dict(base); dn[m] -= eps
        fd = (psf_and_derivatives(up)[0] - psf_and_derivatives(dn)[0]) / (2 * eps)
        err = float((fd - derivs[m]).abs().max() / max(float(derivs[m].abs().max()), 1e-30))
        worst = max(worst, err)
    if worst > tol:
        raise SystemExit(f"derivative validation FAILED: worst relative error {worst:.3e}")
    return worst


def stage_predict(structure):
    t0 = time.time()
    rows, curves = [], []
    for mag in MAGS:
        for seed in SEEDS:
            true = draw_true(mag, seed)
            F = axial_fisher(structure, true)
            crb, ndeg = crb_identifiable(F, MODES, crb_max=THRESHOLD ** 2)
            w = rms_of(true)
            row = {"magnitude": mag, "seed": seed, "rms": w, "n_degenerate": ndeg,
                   "fisher_min_eig": float(np.linalg.eigvalsh(F)[0]),
                   "fisher_cond": float(np.linalg.cond(F))}
            for i, m in enumerate(MODES):
                row[f"true_{m}"] = true[m]
                row[f"sigma_{m}"] = float(np.sqrt(crb[m]))
            row["sigma_max"] = max(row[f"sigma_{m}"] for m in MODES)
            rows.append(row)
        curves.append(np.mean([r["sigma_max"] for r in rows if r["magnitude"] == mag]))
    df = pd.DataFrame(rows)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTDIR / "prediction_grid.csv", index=False)
    mags = np.array(MAGS)
    w_rms = np.array([rms_of(draw_true(m, SEEDS[0])) for m in MAGS])
    w_rms = np.array([df[df.magnitude == m]["rms"].mean() for m in MAGS])
    sig = np.array(curves)
    cx = crossing(w_rms, sig, THRESHOLD)
    payload = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                     text=True, cwd=ROOT).stdout.strip() or None,
        "model": {"na": NA, "wavelength_nm": WAVELENGTH_NM, "pixel_nm": PIXEL_NM,
                  "nz": NZ, "dz_nm": DZ_NM, "kernel": KERNEL,
                  "photons": PHOTONS, "background": BG, "read_noise": READ,
                  "modes": MODES, "threshold_waves": THRESHOLD,
                  "rms_factors": SIGMA.tolist(),
                  "structure": "CoCoA ref_no_aberration.tif, planes zi-4..zi+4, "
                               "crop 160:288 (128x128), per-plane 0.5/99.5 percentile"},
        "swept_variable": "per-mode uniform draw U(-mag, mag), 5 seeds; RMS by Eq. (9)",
        "predicted_crossing_rms_waves": cx,
        "curve_rms": w_rms.tolist(),
        "curve_sigma_max": sig.tolist(),
        "derivative_check_rel_err": VALIDATED,
        "wall_s": round(time.time() - t0, 1),
    }
    (OUTDIR / "predicted_crossing.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print("\nPREDICTION FROZEN. Do not modify before running stage=measure.")


def estimate_axial(counts, structure, n_steps, lr):
    est = {m: torch.tensor(0.0, requires_grad=True, device=counts.device) for m in MODES}
    opt = torch.optim.Adam(list(est.values()), lr=lr)
    for _ in range(n_steps):
        opt.zero_grad()
        psf = make_psf_3d(MicroscopyPSFConfig(
            wavelength_nm=WAVELENGTH_NM, na=NA, pixel_size_nm=PIXEL_NM, kernel_size=KERNEL,
            aberrations_waves=est), nz=NZ, dz_nm=DZ_NM, device=counts.device)
        mu = forward_3d_stack(structure, psf) * PHOTONS + BG
        loss = poisson_gaussian_nll(counts, mu, READ)
        loss.backward()
        opt.step()
    return {m: float(est[m].detach()) for m in MODES}, float(loss.detach())


def stage_measure(structure, n_steps=400, lr=0.08, device="cuda:0"):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    structure = structure.to(dev)
    rows = []
    for mag in MAGS:
        for seed in SEEDS:
            true = draw_true(mag, seed)
            psf_true = make_psf_3d(MicroscopyPSFConfig(
                wavelength_nm=WAVELENGTH_NM, na=NA, pixel_size_nm=PIXEL_NM, kernel_size=KERNEL,
                aberrations_waves=true), nz=NZ, dz_nm=DZ_NM, device=dev)
            mu = forward_3d_stack(structure, psf_true) * PHOTONS + BG
            torch.manual_seed(80 + seed)
            counts = torch.poisson(mu)
            est, nll = estimate_axial(counts, structure, n_steps, lr)
            row = {"magnitude": mag, "seed": seed, "rms": rms_of(true), "terminal_nll": nll}
            for m in MODES:
                e = est[m] - true[m]
                row[f"true_{m}"] = true[m]
                row[f"est_{m}"] = est[m]
                row[f"err_{m}"] = e
                row[f"signed_sq_{m}"] = e ** 2
                row[f"signaware_abs_{m}"] = min(abs(e), abs(est[m] + true[m]))
            rows.append(row)
            print({k: round(v, 4) for k, v in row.items() if k in ("magnitude", "seed", "rms")}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "empirical_sweep.csv", index=False)
    print("saved", OUTDIR / "empirical_sweep.csv", df.shape)


def stage_compare():
    pred_p = OUTDIR / "predicted_crossing.json"
    emp_p = OUTDIR / "empirical_sweep.csv"
    if not pred_p.exists() or not emp_p.exists():
        raise SystemExit("need both prediction and measurement")
    if pred_p.stat().st_mtime > emp_p.stat().st_mtime:
        raise SystemExit("REFUSING: prediction file is newer than the measurement file "
                         "(the prediction was not frozen before the run)")
    pred = json.loads(pred_p.read_text(encoding="utf-8"))
    emp = pd.read_csv(emp_p)
    g = emp.groupby("magnitude").agg(rms=("rms", "mean")).reset_index()
    w_pred = pred["predicted_crossing_rms_waves"]
    out = {"predicted_crossing_rms_waves": w_pred,
           "frozen_utc": pred["frozen_utc"], "per_mode": {}}
    for m in MODES:
        rmse = np.sqrt(emp.groupby("magnitude")[f"signed_sq_{m}"].mean().values)
        sw = np.sqrt(emp.groupby("magnitude")[f"signaware_abs_{m}"].apply(
            lambda s: np.mean(np.asarray(s) ** 2)).values)
        cx_plain = crossing(g["rms"].values, rmse, THRESHOLD)
        cx_sw = crossing(g["rms"].values, sw, THRESHOLD)
        out["per_mode"][m] = {
            "empirical_crossing_plain_rms": cx_plain,
            "empirical_crossing_signaware_rms": cx_sw,
            "rmse_by_level": dict(zip(g["magnitude"], np.round(rmse, 5))),
            "signaware_rmse_by_level": dict(zip(g["magnitude"], np.round(sw, 5))),
            "ratio_plain": (None if (not cx_plain or not w_pred) else cx_plain / w_pred),
            "ratio_signaware": (None if (not cx_sw or not w_pred) else cx_sw / w_pred),
        }
    out["verdict"] = ("consistent_with_prediction" if w_pred is None and
                      all(v["empirical_crossing_plain_rms"] is None for v in out["per_mode"].values())
                      else "FALSIFIED_or_partial")
    (OUTDIR / "comparison.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


VALIDATED = None

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["validate", "predict", "measure", "compare"])
    ap.add_argument("--n-steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    if a.stage == "validate":
        print("derivative worst relative error vs central differences:", validate_derivative())
        raise SystemExit(0)

    VALIDATED = validate_derivative()
    print(f"[check] analytic dh/dc vs central differences: worst rel err {VALIDATED:.2e}")
    structure = load_volume()
    if a.stage == "predict":
        stage_predict(structure)
    elif a.stage == "measure":
        stage_measure(structure, n_steps=a.n_steps, lr=a.lr, device=a.device)
    else:
        stage_compare()
