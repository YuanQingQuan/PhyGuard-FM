from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as tF

from phyguard.optics.psf import (
    MicroscopyPSFConfig, make_psf, psf_aberration_derivatives, _zernike_phase,
)
from experiments.run_biosr_identifiability_sweep import load_gt  # noqa: F401

OUT = ROOT / "results" / "vectorial_minimal"
WL, PX, KS = 520.0, 62.6, 41
N_MEDIUM = 1.518
PHOTONS, BG = 50.0, 7.5
EIGHT = ["defocus", "astig_x", "astig_y", "coma_x", "coma_y",
         "trefoil_x", "trefoil_y", "spherical"]
EVEN = {"defocus", "astig_x", "astig_y", "spherical"}
FOUR = ["defocus", "astig_x", "coma_x", "spherical"]
FOUR_DRAW = {"defocus": 0.10, "astig_x": 0.12, "coma_x": 0.12, "spherical": 0.06}


def _pupil_grid(na, n=KS, dtype=torch.float64):
    fy = torch.fft.fftfreq(n, d=PX, dtype=dtype)
    yy, xx = torch.meshgrid(fy, fy, indexing="ij")
    rr = torch.sqrt(xx**2 + yy**2)
    rho = rr / (na / WL)
    theta = torch.atan2(yy, xx)
    aperture = (rho <= 1.0).to(dtype)
    return rho, theta, aperture


def _rw_columns(sin_t, cos_t, theta):

    ct, st = torch.cos(theta), torch.sin(theta)
    col_x = torch.stack([cos_t * ct**2 + st**2,
                         ct * st * (cos_t - 1.0),
                         sin_t * ct])
    col_y = torch.stack([ct * st * (cos_t - 1.0),
                         cos_t * st**2 + ct**2,
                         sin_t * st])
    col_z = torch.stack([-cos_t * sin_t * ct,
                         -cos_t * sin_t * st,
                         sin_t**2])
    return col_x, col_y, col_z


def vector_psf(coeffs, na, *, n=KS, n_medium=N_MEDIUM, emitter="iso",
               dtype=torch.float64):

    rho, theta, aperture = _pupil_grid(na, n=n, dtype=dtype)
    phase = _zernike_phase(rho, theta, coeffs).to(dtype)
    sin_t = ((na / n_medium) * rho).clamp(max=1.0)
    cos_t = torch.sqrt((1.0 - sin_t**2).clamp_min(0.0))
    apod = torch.sqrt(cos_t)
    common = (aperture * apod * torch.exp(1j * phase)).to(torch.complex128)
    cols = _rw_columns(sin_t, cos_t, theta)
    if emitter == "x":
        cols = (cols[0],)
    inten = None
    for col in cols:
        for j in range(3):
            c = torch.abs(torch.fft.ifft2(common * col[j])) ** 2
            inten = c if inten is None else inten + c
    if emitter == "iso":
        inten = inten / 3.0
    inten = torch.fft.fftshift(inten)
    return (inten / inten.sum().clamp_min(1e-300)).to(dtype)


def scalar_psf(coeffs, na, *, n=KS, dtype=torch.float64):
    cfg = MicroscopyPSFConfig(wavelength_nm=WL, na=na, pixel_size_nm=PX,
                              kernel_size=n, aberrations_waves=coeffs)
    return make_psf(cfg, dtype=dtype)


def _structure():
    x = torch.tensor(load_gt("ER", 1, 1), dtype=torch.float64).clamp(0.0, 1.0)
    return x[None, None]


def fisher_from_psf(psf_fn, coeffs, na, modes, *, eps=1e-4):

    x = _structure()
    kh = KS
    hx = tF.conv2d(x, psf_fn(coeffs, na)[None, None], padding=(kh // 2, kh // 2))
    mask = (hx > 0).to(torch.float64)
    lam = (hx.clamp_min(0.0) * PHOTONS + BG).clamp_min(1e-6)
    dmu = {}
    for m in modes:
        up = dict(coeffs); dn = dict(coeffs)
        up[m] = up.get(m, 0.0) + eps
        dn[m] = dn.get(m, 0.0) - eps
        dh = (psf_fn(up, na)[None, None] - psf_fn(dn, na)[None, None]) / (2 * eps)
        dmu[m] = tF.conv2d(x, dh, padding=(kh // 2, kh // 2)) * mask * PHOTONS
    F = np.zeros((len(modes), len(modes)))
    lam_np = lam[0, 0].numpy()
    dm = {m: dmu[m][0, 0].numpy() for m in modes}
    for i, mi in enumerate(modes):
        for j, mj in enumerate(modes):
            F[i, j] = float(np.sum(dm[mi] * dm[mj] / lam_np))
    return F


def fisher_analytic_scalar(coeffs, na, modes):
    cfg = MicroscopyPSFConfig(wavelength_nm=WL, na=na, pixel_size_nm=PX,
                              kernel_size=KS, aberrations_waves=coeffs)
    derivs = psf_aberration_derivatives(cfg, tuple(modes), dtype=torch.float64)
    x = _structure()
    hx = tF.conv2d(x, make_psf(cfg, dtype=torch.float64)[None, None],
                   padding=(KS // 2, KS // 2))
    mask = (hx > 0).to(torch.float64)
    lam = (hx.clamp_min(0.0) * PHOTONS + BG).clamp_min(1e-6)
    dm = {m: (tF.conv2d(x, derivs[m][None, None], padding=(KS // 2, KS // 2))
              * mask * PHOTONS)[0, 0].numpy() for m in modes}
    lam_np = lam[0, 0].numpy()
    F = np.zeros((len(modes), len(modes)))
    for i, mi in enumerate(modes):
        for j, mj in enumerate(modes):
            F[i, j] = float(np.sum(dm[mi] * dm[mj] / lam_np))
    return F


def crb_waves(F):
    w, V = np.linalg.eigh(F)
    keep = w > 1.0 / (0.05 ** 2)
    inv = (V[:, keep] * (1.0 / np.maximum(w[keep], 1e-300))) @ V[:, keep].T
    return np.sqrt(np.maximum(np.diag(inv), 0.0)), int((~keep).sum())


def stage_checks():
    out = {}

    base = {m: 0.05 * (1 if i % 2 == 0 else -1) for i, m in enumerate(FOUR)}
    F_an = fisher_analytic_scalar(base, 1.2, FOUR)
    F_fd = fisher_from_psf(scalar_psf, base, 1.2, FOUR)
    rel = np.abs(F_fd - F_an) / np.maximum(np.abs(F_an), 1e-30)
    out["fd_vs_analytic_scalar_max_rel_all_entries"] = float(np.max(rel))
    out["fd_vs_analytic_scalar_max_rel_diagonal"] = float(
        np.max(np.diag(rel)))


    out["fd_eps"] = 1e-4


    draw = {"defocus": 0.09, "astig_x": 0.07, "coma_x": -0.05, "spherical": 0.06}
    flip_even = dict(draw)
    for m in [k for k in draw if k in EVEN]:
        flip_even[m] = -flip_even[m]
    for name, fn in (("scalar", scalar_psf),
                     ("vec_x", lambda c, na: vector_psf(c, na, emitter="x")),
                     ("vec_iso", lambda c, na: vector_psf(c, na, emitter="iso"))):
        h1, h2 = fn(draw, 1.2), fn(flip_even, 1.2)
        out[f"parity_residual_{name}"] = float((h1 - h2).abs().max() / h1.max())
        h3 = fn({m: -v for m, v in draw.items()}, 1.2)
        out[f"fullflip_l1_{name}"] = float((h1 - h3).abs().sum() / h1.sum())


    for em in ("x", "iso"):
        for na in (0.8, 1.2):
            hs, hv = scalar_psf(draw, na), vector_psf(draw, na, emitter=em)
            out[f"kernel_l1_scalar_vs_vec_{em}_NA{na}"] = float(
                (hs - hv).abs().sum() / hs.sum())
            out[f"strehl_vec_{em}_NA{na}"] = float(hv.max() / hv.sum())
    (OUT).mkdir(parents=True, exist_ok=True)
    (OUT / "checks.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return out


def stage_slopes(na=1.2, emitter="iso"):
    alphas = np.logspace(-3, np.log10(0.3), 12)
    rows = []
    for m in EIGHT:
        for a in alphas:
            coeffs = {k: (float(a) if k == m else 0.0) for k in EIGHT}
            F = fisher_from_psf(
                lambda c, na_: vector_psf(c, na_, emitter=emitter),
                coeffs, na, [m])
            rows.append({"mode": m, "parity": "even" if m in EVEN else "odd",
                         "alpha": float(a), "F_ii": float(F[0, 0])})
        print(f"vectorial({emitter}) slope scan:", m, flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"slopes_{emitter}.csv", index=False)
    summary = []
    for m in EIGHT:
        q = df[df["mode"] == m]
        la, lf = np.log10(q["alpha"]), np.log10(np.maximum(q["F_ii"], 1e-30))
        slope, intercept = np.polyfit(la, lf, 1)
        pred = slope * la + intercept
        r2 = 1 - ((lf - pred) ** 2).sum() / ((lf - lf.mean()) ** 2).sum()
        summary.append({"mode": m, "parity": "even" if m in EVEN else "odd",
                        "slope": round(float(slope), 3), "R2": round(float(r2), 4)})
    pd.DataFrame(summary).to_csv(OUT / f"slopes_summary_{emitter}.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))


def stage_crb_ratio(emitter="iso"):
    rows = []
    for na in (0.8, 1.2):
        Fs = fisher_analytic_scalar(FOUR_DRAW, na, FOUR)
        Fv = fisher_from_psf(
            lambda c, na_: vector_psf(c, na_, emitter=emitter), FOUR_DRAW, na, FOUR)
        cs, nds = crb_waves(Fs)
        cv, ndv = crb_waves(Fv)
        for i, m in enumerate(FOUR):
            rows.append({"na": na, "mode": m,
                         "crb_scalar_waves": round(float(cs[i]), 4),
                         "crb_vectorial_waves": round(float(cv[i]), 4),
                         "ratio_vectorial_over_scalar": round(float(cv[i] / cs[i]), 3)})
        rows.append({"na": na, "mode": "norm(4 modes)",
                     "crb_scalar_waves": round(float(np.linalg.norm(cs)), 4),
                     "crb_vectorial_waves": round(float(np.linalg.norm(cv)), 4),
                     "ratio_vectorial_over_scalar": round(
                         float(np.linalg.norm(cv) / np.linalg.norm(cs)), 3)})
        rows.append({"na": na, "mode": "n_degenerate(scalar/vectorial)",
                     "crb_scalar_waves": nds, "crb_vectorial_waves": ndv,
                     "ratio_vectorial_over_scalar": ""})
        print(f"NA={na} ({emitter}): scalar CRB {np.round(cs,4)}  "
              f"vectorial CRB {np.round(cv,4)}", flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"crb_ratio_{emitter}.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    emitter = sys.argv[2] if len(sys.argv) > 2 else "iso"
    if stage in ("checks", "all"):
        stage_checks()
    if stage in ("slopes", "all"):
        stage_slopes(emitter=emitter)
    if stage in ("crb", "all"):
        stage_crb_ratio(emitter=emitter)
