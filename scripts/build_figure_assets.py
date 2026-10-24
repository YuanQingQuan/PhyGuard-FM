from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from skimage.restoration import richardson_lucy

from phyguard.optics.psf import MicroscopyPSFConfig, make_psf, _zernike_phase
from phyguard.optics.forward import PhotonMicroscope
from phyguard.likelihood import poisson_gaussian_nll
from phyguard.estimation.calibration import estimate_aberrations_multiframe
from experiments.run_blind_calibration import blind_estimate, MODES
from experiments.run_biosr_identifiability_sweep import load_gt
from experiments.run_real_optical_pilot import normalize

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

WL, NA, PX, KS = 520.0, 1.2, 62.6, 41
PHOTONS = 30.0
AB_MILD = {"defocus": 0.05, "astig_x": 0.08, "coma_x": 0.08, "spherical": 0.03}
AB_STRONG = {"defocus": 0.10, "astig_x": 0.12, "coma_x": 0.12, "spherical": 0.06}
OFFSETS5 = [-1.0, -0.5, 0.0, 0.5, 1.0]


def pupil_phase(coeffs: dict, n: int = 128) -> np.ndarray:
    fy = torch.fft.fftfreq(n, d=PX)
    fx = torch.fft.fftfreq(n, d=PX)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    rho = torch.sqrt(xx**2 + yy**2) / (NA / WL)
    theta = torch.atan2(yy, xx)
    ph = _zernike_phase(rho, theta, coeffs) / (2 * torch.pi)
    ph[rho > 1] = 0
    return ph.numpy()


def simulate(gt: torch.Tensor, ab: dict, offsets, photons, seed=0):
    bg = max(0.2, 0.15 * photons)
    frames = []

    torch.manual_seed(100 + seed)
    for off in offsets:
        th = dict(ab); th["defocus"] += off
        psf = make_psf(MicroscopyPSFConfig(wavelength_nm=WL, na=NA, pixel_size_nm=PX,
                                           kernel_size=KS, aberrations_waves=th)).to(DEV)
        mic = PhotonMicroscope(psf=psf, signal_photons=photons, background_photons=bg,
                               read_noise_std=0.15)
        frames.append(mic.sample(gt)["counts"])
    return torch.cat(frames, 0), bg


def rl_image(counts_frame, psf_np, photons, bg, iters=20):
    img = ((counts_frame[0, 0] - bg) / photons).clamp(0, 1).cpu().numpy().astype(np.float32)
    rl = richardson_lucy(np.clip(img, 0, None), psf_np, num_iter=iters, clip=False)
    return normalize(np.clip(rl, 0, None), p_low=0.1, p_high=99.7), normalize(np.clip(img, 0, None), p_low=0.1, p_high=99.7)


def psf_np(coeffs=None):
    return make_psf(MicroscopyPSFConfig(wavelength_nm=WL, na=NA, pixel_size_nm=PX,
                                        kernel_size=KS, aberrations_waves=coeffs or {})).detach().cpu().numpy()


def build_assets() -> dict:

    out = {}
    for sname, ab, tag in [("ER", AB_STRONG, "ER"), ("MT", AB_MILD, "MT")]:
        gt = torch.tensor(load_gt(sname, 1, 1))[None, None].to(DEV)
        counts, bg = simulate(gt, ab, OFFSETS5, PHOTONS, seed=7)
        mid = counts[2:3]

        res_nb = estimate_aberrations_multiframe(
            counts, gt, modes=MODES, defocus_offsets=OFFSETS5, wavelength_nm=WL, na=NA,
            pixel_size_nm=PX, signal_photons=PHOTONS, background_photons=bg,
            read_noise_std=0.15, n_steps=500, n_restarts=4, seed=0)
        est_nb = res_nb.estimated_coeffs

        obs_img = ((mid[0, 0] - bg) / PHOTONS).clamp(0, 1).cpu().numpy().astype(np.float32)
        x0 = richardson_lucy(np.clip(obs_img, 0, None), psf_np(), num_iter=15, clip=False)
        x_init = torch.tensor(normalize(np.clip(x0, 0, None), p_low=0.1, p_high=99.7))[None, None].to(DEV)
        res_bl = blind_estimate(counts, x_init, modes=MODES, wavelength_nm=WL, na=NA,
                                pixel_size_nm=PX, signal_photons=PHOTONS, background_photons=bg,
                                read_noise_std=0.15, tv_weight=0.10, n_outer=15, n_inner=6,
                                lr_x=0.01, lr_theta=0.2, seed=0, defocus_offsets=OFFSETS5)
        est_bl = res_bl["theta"]

        _, obs_norm = rl_image(mid, psf_np(), PHOTONS, bg)
        rl_nom, _ = rl_image(mid, psf_np(), PHOTONS, bg)
        rl_cal, _ = rl_image(mid, psf_np(est_nb), PHOTONS, bg)
        th_true_psf = dict(ab)
        rl_true, _ = rl_image(mid, psf_np(th_true_psf), PHOTONS, bg)
        blind_x = res_bl["x"][0, 0].cpu().numpy()

        out[f"{tag}_gt"] = gt[0, 0].cpu().numpy()
        out[f"{tag}_obs"] = obs_norm
        out[f"{tag}_rl_nom"] = rl_nom
        out[f"{tag}_rl_cal"] = rl_cal
        out[f"{tag}_rl_true"] = rl_true
        out[f"{tag}_blind_x"] = normalize(np.clip(blind_x, 0, None), p_low=0.1, p_high=99.7)
        out[f"{tag}_wf_true"] = pupil_phase(ab)
        out[f"{tag}_wf_est_nb"] = pupil_phase(est_nb)
        out[f"{tag}_wf_est_bl"] = pupil_phase(est_bl)
        out[f"{tag}_hist_bl"] = np.array(res_bl["loss_history"])
        for m in MODES:
            out[f"{tag}_true_{m}"] = ab[m]
            out[f"{tag}_estnb_{m}"] = est_nb[m]
            out[f"{tag}_estbl_{m}"] = est_bl[m]
        print(tag, "est_nb", {m: round(est_nb[m], 3) for m in MODES}, flush=True)
        print(tag, "est_bl", {m: round(est_bl[m], 3) for m in MODES}, flush=True)


    gt128 = torch.tensor(load_gt("ER", 1, 1)[64:192, 64:192].copy())[None, None].to(DEV)
    ph128, bg128 = 30.0, 4.5
    truth = {"astig_x": 0.15, "astig_y": -0.10}
    offs_map = {"1f": [0.0], "3f": [-0.5, 0.0, 0.5]}

    def counts128(ab, offsets, seed):
        frames = []

        torch.manual_seed(300 + seed)
        for off in offsets:
            th = dict(ab); th["defocus"] = th.get("defocus", 0.0) + off
            psf = make_psf(MicroscopyPSFConfig(wavelength_nm=WL, na=NA, pixel_size_nm=PX,
                                               kernel_size=KS, aberrations_waves=th)).to(DEV)
            mic = PhotonMicroscope(psf=psf, signal_photons=ph128, background_photons=bg128,
                                   read_noise_std=0.15)
            frames.append(mic.sample(gt128)["counts"])
        return frames

    def nll_at(a3, a5, obs_frames, offsets):
        tot = 0.0
        for off, ob in zip(offsets, obs_frames):
            psf = make_psf(MicroscopyPSFConfig(wavelength_nm=WL, na=NA, pixel_size_nm=PX,
                                               kernel_size=KS,
                                               aberrations_waves={"astig_x": float(a3), "astig_y": float(a5),
                                                                  "defocus": float(off)})).to(DEV)
            mic = PhotonMicroscope(psf=psf, signal_photons=ph128, background_photons=bg128,
                                   read_noise_std=0.15)
            tot += float(poisson_gaussian_nll(ob, mic.expected_counts(gt128), 0.15))
        return tot

    grid = np.linspace(-0.3, 0.3, 17)
    for tag, offs in offs_map.items():
        obs_frames = counts128(truth, offs, seed=11)
        surf = np.zeros((17, 17))
        for i, a5 in enumerate(grid):
            for j, a3 in enumerate(grid):
                surf[i, j] = nll_at(a3, a5, obs_frames, offs)
        err = np.zeros((17, 17))
        for i, a5 in enumerate(grid):
            for j, a3 in enumerate(grid):
                err[i, j] = abs(a3 - truth["astig_x"]) + abs(a5 - truth["astig_y"])
        out[f"nllsurf_{tag}"] = surf
        out[f"errmap_{tag}"] = err

        cg = np.linspace(-0.3, 0.3, 13)
        preds = []
        trues = []
        for k, ang in enumerate(np.linspace(0, 2 * np.pi, 8, endpoint=False)):
            t = {"astig_x": 0.15 * np.cos(ang), "astig_y": 0.15 * np.sin(ang)}
            frames = counts128(t, offs, seed=50 + k)
            best, bi = None, None
            for i, a5 in enumerate(cg):
                for j, a3 in enumerate(cg):
                    v = nll_at(a3, a5, frames, offs)
                    if best is None or v < best:
                        best, bi = v, (a3, a5)
            trues.append((t["astig_x"], t["astig_y"])); preds.append(bi)
        out[f"circle_true_{tag}"] = np.array(trues)
        out[f"circle_pred_{tag}"] = np.array(preds)
        print("circle", tag, "done", flush=True)

    return out


def main() -> None:
    default = ROOT / "results" / "paper_figures" / "v2" / "assets.npz"
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(default))
    args = ap.parse_args()

    if not (ROOT / "data" / "BioSR").exists():
        raise SystemExit(
            "Required directory: data/BioSR.")

    print("device:", DEV)
    assets = build_assets()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **assets)
    print(f"assets saved -> {out} ({out.stat().st_size/1e6:.1f} MB, {len(assets)} arrays)")


if __name__ == "__main__":
    main()
