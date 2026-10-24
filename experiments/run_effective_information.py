from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as tF

from phyguard.optics.psf import (
    MicroscopyPSFConfig, make_psf, psf_aberration_derivatives,
)
from experiments.run_biosr_identifiability_sweep import load_gt

OUT = ROOT / "results" / "effective_information"
NA, WL, PX, KS = 1.2, 520.0, 62.6, 41
MODES = ["defocus", "astig_x", "coma_x", "spherical"]
THETA = {m: v for m, v in zip(MODES, (-0.11, 0.06, 0.17, 0.19))}
EVEN = {"defocus", "astig_x", "spherical"}


def load_structure(n=256):
    x = load_gt("ER", 1, 1)
    x = torch.tensor(np.asarray(x, dtype=np.float64))
    c = (x.shape[0] - n) // 2
    return x[c:c + n, c:c + n].clamp(0.0, 1.0)


def moments(x, theta, s, kernel=KS):

    n = x.shape[0]
    cfg = MicroscopyPSFConfig(wavelength_nm=WL, na=NA, pixel_size_nm=PX,
                              kernel_size=kernel, aberrations_waves=theta)
    psf = make_psf(cfg, dtype=torch.float64)
    derivs = psf_aberration_derivatives(cfg, tuple(MODES), dtype=torch.float64)
    xx = x[None, None]
    hx = tF.conv2d(xx, psf[None, None], padding=kernel // 2)[0, 0]
    mask = (hx > 0).to(torch.float64)
    b = max(0.2, 0.15 * s)
    lam = (hx.clamp_min(0.0) * s + b).numpy()
    dmu = np.stack([(tF.conv2d(xx, derivs[m][None, None], padding=kernel // 2)[0, 0]
                     * mask * s).numpy() for m in MODES])
    F = np.zeros((len(MODES), len(MODES)))
    for i in range(len(MODES)):
        for j in range(i, len(MODES)):
            F[i, j] = F[j, i] = float(np.sum(dmu[i] * dmu[j] / lam))
    h_full = np.zeros((n, n))
    k = kernel // 2
    idx = np.arange(-k, k + 1)
    h_full[np.ix_(idx % n, idx % n)] = psf.numpy()
    return F, dmu, lam, h_full


def real_basis(n, radius):

    fy = np.fft.fftfreq(n, d=1.0) * n
    ky, kx = np.meshgrid(fy, fy, indexing="ij")
    rr = np.sqrt(ky ** 2 + kx ** 2)
    half = (ky > 0) | ((ky == 0) & (kx > 0))
    sel = half & (rr <= radius)
    kys, kxs = np.nonzero(sel)
    r = np.arange(n)
    rows = [np.ones((n, n))]
    freqs = [(0, 0, "dc")]
    for a, b in zip(kys, kxs):
        ang = 2 * np.pi * (a * r[:, None] + b * r[None, :]) / n
        rows.append(np.cos(ang))
        rows.append(np.sin(ang))
        freqs.append((int(a), int(b), "cos"))
        freqs.append((int(a), int(b), "sin"))
    return np.stack(rows), freqs


def schur_real_basis(F_tt, dmu, lam, h_full, s, radius, dtype=np.float32,
                     chunk=512, memmap_dir=None):

    n = lam.shape[0]
    B, freqs = real_basis(n, radius)
    Hhat = np.fft.fft2(h_full)
    K = B.shape[0]
    w = (1.0 / lam).astype(dtype)
    wflat = w.ravel()
    bytes_need = K * n * n * np.dtype(dtype).itemsize
    cleanup = None
    if memmap_dir is not None and bytes_need > 1.0e9:
        path = Path(memmap_dir) / "_dB_memmap.dat"
        dB = np.memmap(path, dtype=dtype, mode="w+", shape=(K, n * n))
        cleanup = path
    else:
        dB = np.empty((K, n * n), dtype=dtype)
    rr = np.arange(n)
    dmu_w = (dmu * w[None, :, :]).reshape(4, -1)
    for t in range(K):
        a, b, kind = freqs[t]
        if kind == "dc":
            dB[t] = s * Hhat[0, 0].real
            continue
        ang = 2 * np.pi * (a * rr[:, None] + b * rr[None, :]) / n
        phase = np.exp(1j * ang) * Hhat[a % n, b % n]
        dB[t] = (s * phase.real if kind == "cos" else s * phase.imag).ravel()
    dB = np.asarray(dB)
    A = np.zeros((K, K), dtype=np.float64)
    G = np.zeros((4, K))
    for start in range(0, K, chunk):
        stop = min(start + chunk, K)
        A[start:stop, :] = (dB[start:stop] * wflat[None, :]) @ dB.T
        G[:, start:stop] = dmu_w @ dB[start:stop].T
    A = 0.5 * (A + A.T)
    reg = 1e-30 * np.trace(A) / max(K, 1)
    X = np.linalg.solve(A + reg * np.eye(K), G.T)
    F_eff = F_tt - G @ X
    if cleanup is not None:
        del dB
        cleanup.unlink(missing_ok=True)
    return 0.5 * (F_eff + F_eff.T), K


def schur_bruteforce_real(F_tt, dmu, lam, h_full, s, radius):

    n = lam.shape[0]
    B, _ = real_basis(n, radius)
    K = B.shape[0]
    Hhat = np.fft.fft2(h_full)
    dB = np.stack([np.fft.ifft2(Hhat * np.fft.fft2(b)).real * s for b in B])
    w = 1.0 / lam
    Bf = dB.reshape(K, -1)
    A = (Bf * w.ravel()[None, :]) @ Bf.T
    G = (dmu * w[None, :, :]).reshape(4, -1) @ Bf.T
    A = 0.5 * (A + A.T)
    F_eff = F_tt - G @ np.linalg.pinv(A) @ G.T
    return 0.5 * (F_eff + F_eff.T), K


def crb(F, crb_max=0.0025):
    w, V = np.linalg.eigh(F)
    keep = w > 1.0 / crb_max
    inv = (V[:, keep] * (1.0 / np.maximum(w[keep], 1e-300))) @ V[:, keep].T
    return np.sqrt(np.maximum(np.diag(inv), 0.0)), int((~keep).sum()), w


def stage_validation():
    n, ks, s = 64, 11, 20.0
    x = load_structure(n)
    F_tt, dmu, lam, h_full = moments(x, THETA, s, kernel=ks)
    out = {"n": n, "kernel": ks, "s": s}
    for radius in (4, 8):
        F_a, Ka = schur_real_basis(F_tt, dmu, lam, h_full, s, radius,
                                   dtype=np.float64)
        F_b, Kb = schur_bruteforce_real(F_tt, dmu, lam, h_full, s, radius)
        out[f"radius_{radius}"] = {
            "K": int(Ka), "K_bruteforce": int(Kb),
            "max_rel_deviation": float(np.max(np.abs(F_a - F_b))
                                       / np.max(np.abs(F_b)))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "validation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return out


def stage_schur(radii=(6, 10, 16, 22, 26, 32), s_list=(50.0,), invariance_s=5.0):

    x = load_structure(256)
    rows = []
    for s in s_list:
        F_tt, dmu, lam, h_full = moments(x, THETA, s)
        for radius in radii:
            F_eff, K = schur_real_basis(F_tt, dmu, lam, h_full, s, radius,
                                        memmap_dir=OUT)
            w_tt = np.linalg.eigvalsh(F_tt)
            w_eff = np.linalg.eigvalsh(F_eff)
            c_tt_p = np.sqrt(np.diag(np.linalg.pinv(F_tt)))
            c_eff_p = np.sqrt(np.diag(np.linalg.pinv(F_eff)))
            c_tt_t, nd_tt, _ = crb(F_tt)
            c_eff_t, nd_eff, _ = crb(F_eff)
            th = np.array([THETA[m] for m in MODES])
            rows.append({
                "s": s, "radius": radius, "K": K,
                "lambda_min_tt": float(w_tt[0]),
                "lambda_min_eff": float(w_eff[0]),
                "lambda_min_ratio": float(w_eff[0] / w_tt[0]),
                "lambda_max_ratio": float(w_eff[-1] / w_tt[-1]),
                "crb_norm_inflation_pinv": float(
                    np.linalg.norm(c_eff_p) / np.linalg.norm(c_tt_p)),
                **{f"crb_inflation_pinv_{m}": float(c_eff_p[i] / c_tt_p[i])
                   for i, m in enumerate(MODES)},
                "crb_norm_inflation_thresholded": float(
                    np.linalg.norm(c_eff_t) / np.linalg.norm(c_tt_t)),
                "n_degenerate_tt": nd_tt, "n_degenerate_eff": nd_eff,
                "crb_eff_max_mode_waves": float(np.max(c_eff_t)),
            })
            print(f"s={s} R={radius} K={K}: lam_min ratio "
                  f"{w_eff[0]/w_tt[0]:.3f}, CRB(pinv) inflation "
                  f"{np.linalg.norm(c_eff_p)/np.linalg.norm(c_tt_p):.3f} "
                  f"norm / {np.max(c_eff_p/c_tt_p):.2f} worst mode; "
                  f"thresholded {np.linalg.norm(c_eff_t)/np.linalg.norm(c_tt_t):.3f} "
                  f"(degenerate {nd_eff})", flush=True)
    if invariance_s is not None:
        s = invariance_s
        F_tt, dmu, lam, h_full = moments(x, THETA, s)
        F_eff, K = schur_real_basis(F_tt, dmu, lam, h_full, s, 16,
                                    memmap_dir=OUT)
        w_tt = np.linalg.eigvalsh(F_tt)
        w_eff = np.linalg.eigvalsh(F_eff)
        c_tt_p = np.sqrt(np.diag(np.linalg.pinv(F_tt)))
        c_eff_p = np.sqrt(np.diag(np.linalg.pinv(F_eff)))
        rows.append({"s": s, "radius": 16, "K": K,
                     "lambda_min_tt": float(w_tt[0]),
                     "lambda_min_eff": float(w_eff[0]),
                     "lambda_min_ratio": float(w_eff[0] / w_tt[0]),
                     "lambda_max_ratio": float(w_eff[-1] / w_tt[-1]),
                     "crb_norm_inflation_pinv": float(
                         np.linalg.norm(c_eff_p) / np.linalg.norm(c_tt_p)),
                     **{f"crb_inflation_pinv_{m}": float(c_eff_p[i] / c_tt_p[i])
                        for i, m in enumerate(MODES)},
                     "crb_norm_inflation_thresholded": float("nan"),
                     "n_degenerate_tt": None, "n_degenerate_eff": None,
                     "crb_eff_max_mode_waves": None})
        print(f"scale-invariance check s={s} R=16: lam_min ratio "
              f"{w_eff[0]/w_tt[0]:.3f}, CRB(pinv) inflation "
              f"{np.linalg.norm(c_eff_p)/np.linalg.norm(c_tt_p):.3f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "schur.csv", index=False)


def stage_parity(radius=16, s=50.0):

    x = load_structure(256)
    rows = []
    for a in np.logspace(-3, np.log10(0.3), 8):
        theta = {"defocus": 0.0, "astig_x": float(a),
                 "coma_x": 0.17, "spherical": 0.0}
        F_tt, dmu, lam, h_full = moments(x, theta, s)
        F_eff, K = schur_real_basis(F_tt, dmu, lam, h_full, s, radius)
        w_eff = np.linalg.eigvalsh(F_eff)
        rows.append({"alpha": float(a), "F_tt_xx": float(F_tt[1, 1]),
                     "F_eff_xx": float(F_eff[1, 1]),
                     "lambda_min_eff": float(w_eff[0]),
                     "lambda_max_eff": float(w_eff[-1]), "K": K})
        print(f"alpha={a:.4g}: F_tt={F_tt[1,1]:.4g} F_eff={F_eff[1,1]:.4g} "
              f"lam_min={w_eff[0]:.4g}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "parity.csv", index=False)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    OUT.mkdir(parents=True, exist_ok=True)
    if stage in ("validation", "all"):
        stage_validation()
    if stage in ("schur", "all"):
        stage_schur()
    if stage in ("parity", "all"):
        stage_parity()
