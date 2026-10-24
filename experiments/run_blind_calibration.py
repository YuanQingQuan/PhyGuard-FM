from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import richardson_lucy, denoise_tv_chambolle

from phyguard.calibration import parameterized_psf
from phyguard.optics.forward import PhotonMicroscope
from phyguard.optics.psf import MicroscopyPSFConfig, make_psf
from phyguard.likelihood import poisson_nll, poisson_gaussian_nll

from experiments.run_real_optical_pilot import normalize
from experiments.run_biosr_identifiability_sweep import load_gt
from phyguard.io import read_image

MODES = ["defocus", "astig_x", "coma_x", "spherical"]


def tv_prior(x: torch.Tensor) -> torch.Tensor:

    dy = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    dx = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    return dy + dx


def blind_estimate(
    counts: torch.Tensor,
    x_init: torch.Tensor,
    *,
    modes: list[str],
    wavelength_nm: float,
    na: float,
    pixel_size_nm: float,
    signal_photons: float,
    background_photons: float,
    read_noise_std: float,
    tv_weight: float = 0.05,
    n_outer: int = 25,
    n_inner: int = 15,
    lr_x: float = 0.02,
    lr_theta: float = 0.10,
    seed: int = 0,
    defocus_offsets: list[float] | None = None,
    axial_offsets_um: list[float] | None = None,
    n_obj: float = 1.0,
) -> dict:

    rng = np.random.default_rng(seed)
    if defocus_offsets is not None:
        offsets = defocus_offsets
    elif axial_offsets_um is not None:
        offsets = axial_offsets_um
    else:
        offsets = [0.0]
    multi = len(offsets) > 1
    if multi and counts.ndim == 4 and counts.shape[0] != len(offsets):
        raise ValueError("counts frame count must match defocus_offsets length")
    best = None
    for r in range(5):
        x = x_init.clone().requires_grad_(True)
        theta = {m: torch.tensor(float(rng.uniform(-0.2, 0.2)) if r > 0 else 0.0, requires_grad=True) for m in modes}
        history = []

        def _nll(off_list, x_in, theta_in):
            tot = 0.0
            for i, off in enumerate(off_list):
                th = dict(theta_in)
                ax_um = 0.0
                if axial_offsets_um is not None:
                    ax_um = float(axial_offsets_um[i])
                else:
                    th["defocus"] = th["defocus"] + off
                psf = parameterized_psf(th, wavelength_nm=wavelength_nm, na=na,
                                        pixel_size_nm=pixel_size_nm, kernel_size=41,
                                        n_obj=n_obj, axial_offset_um=ax_um)
                mic = PhotonMicroscope(psf=psf, signal_photons=signal_photons,
                                       background_photons=background_photons,
                                       read_noise_std=read_noise_std)
                c = counts[i] if multi else counts
                tot = tot + poisson_gaussian_nll(c, mic.expected_counts(x_in), read_noise_std)
            return tot / len(off_list)

        for _ in range(n_outer):

            opt_x = torch.optim.Adam([x], lr=lr_x)
            for _ in range(n_inner):
                opt_x.zero_grad()
                xc = x.clamp(0, 1)
                loss = _nll(offsets, xc, theta) + tv_weight * tv_prior(xc)
                loss.backward()
                opt_x.step()

            opt_t = torch.optim.Adam(list(theta.values()), lr=lr_theta)
            for _ in range(n_inner):
                opt_t.zero_grad()
                xc = x.detach().clamp(0, 1)
                loss = _nll(offsets, xc, theta)
                loss.backward()
                opt_t.step()

            with torch.no_grad():
                xc = x.clamp(0, 1)
                val = float(_nll(offsets, xc, theta).detach())
            history.append(val)
        final = history[-1]
        est = {m: float(theta[m].detach()) for m in modes}
        x_est = x.detach().clamp(0, 1)
        if best is None or final < best[0]:
            best = (final, est, x_est)
    return {"theta": best[1], "x": best[2], "loss": best[0], "loss_history": history}


def stage_a(device: torch.device, outdir: Path, photons: float = 20.0, seed: int = 0, structure: str = "ER", multi: bool = False):

    rng = np.random.default_rng(seed)
    true = {m: float(rng.uniform(-0.25, 0.25)) for m in MODES}
    gt = torch.tensor(load_gt(structure, 1, 1))[None, None].to(device)
    bg = max(0.2, 0.15 * photons)
    offsets = [-0.3, 0.0, 0.3] if multi else [0.0]

    def make_counts(off):
        th = dict(true); th["defocus"] = th["defocus"] + off
        psf = make_psf(MicroscopyPSFConfig(
            wavelength_nm=520.0, na=1.2, pixel_size_nm=62.6,
            kernel_size=41, aberrations_waves=th,
        )).to(device)
        mic = PhotonMicroscope(psf=psf, signal_photons=photons,
                               background_photons=bg, read_noise_std=0.15)
        torch.manual_seed(100 + seed)
        return mic.sample(gt)["counts"]

    counts_all = torch.stack([make_counts(off) for off in offsets], dim=0) if multi else make_counts(0.0)


    mid = 1 if multi else 0
    c = counts_all[mid] if multi else counts_all
    obs_img = torch.clamp((c - bg) / photons, 0, 1)[0, 0].cpu().numpy().astype(np.float32)
    nominal = make_psf(MicroscopyPSFConfig(
        wavelength_nm=520.0, na=1.2, pixel_size_nm=62.6, kernel_size=41,
    )).detach().cpu().numpy()
    rl = richardson_lucy(np.clip(obs_img, 0, 1), nominal, num_iter=15, clip=False)
    x_init = torch.tensor(normalize(np.clip(rl, 0, None), p_low=0.1, p_high=99.7))[None, None].to(device)

    res = blind_estimate(counts_all, x_init, modes=MODES, wavelength_nm=520.0, na=1.2,
                         pixel_size_nm=62.6, signal_photons=photons, background_photons=bg,
                         read_noise_std=0.15, tv_weight=0.10, n_outer=20, n_inner=8,
                         lr_x=0.01, lr_theta=0.2, seed=seed, defocus_offsets=offsets)
    est = res["theta"]
    x_est = res["x"][0, 0].detach().cpu().numpy()
    gt_np = gt[0, 0].detach().cpu().numpy()

    row = {"structure": structure, "multi": multi, "photons": photons, "seed": seed,
           "recon_psnr": float(peak_signal_noise_ratio(gt_np, x_est, data_range=1.0)),
           "obs_psnr": float(peak_signal_noise_ratio(gt_np, obs_img, data_range=1.0))}
    for m in MODES:
        row[f"true_{m}"] = true[m]
        row[f"est_{m}"] = est[m]
        row[f"err_{m}"] = min(abs(est[m] - true[m]), abs(est[m] + true[m]))
    print(f"[stage A] {structure} multi={multi} photons={photons} seed={seed}", flush=True)
    for m in MODES:
        print(f"  {m:10s} true={true[m]:+.3f} est={est[m]:+.3f} err_sa={row[f'err_{m}']:.4f}", flush=True)
    print(f"  recon PSNR {row['recon_psnr']:.2f} (obs {row['obs_psnr']:.2f})", flush=True)
    return row


def stage_b(device: torch.device, outdir: Path, pair: str = "Confocal_FISH_3", seed: int = 0, inject: bool = True):

    datadir = ROOT / "data" / "fmd_minipairs" / pair
    raw = read_image(datadir / "raw.png")
    ref = read_image(datadir / "reference.png")
    h = min(raw.shape[0], ref.shape[0]); w = min(raw.shape[1], ref.shape[1])
    raw, ref = raw[:h, :w], ref[:h, :w]
    y0, x0 = (h - 256) // 2, (w - 256) // 2
    raw_c, ref_c = raw[y0:y0+256, x0:x0+256], ref[y0:y0+256, x0:x0+256]

    rng = np.random.default_rng(seed)
    true = {m: float(rng.uniform(-0.25, 0.25)) for m in MODES} if inject else {m: 0.0 for m in MODES}


    ref_n = normalize(ref_c, p_low=0.5, p_high=99.5)
    raw_n = normalize(raw_c, p_low=0.5, p_high=99.5)
    photons = max(float(np.percentile(ref_c, 90) - np.percentile(ref_c, 10)), 2.0)
    bg = max(0.2, 0.15 * photons)

    gt_t = torch.tensor(ref_n)[None, None].to(device)
    if inject:

        psf_true = make_psf(MicroscopyPSFConfig(
            wavelength_nm=520.0, na=1.15, pixel_size_nm=300.0,
            kernel_size=41, aberrations_waves=true,
        )).to(device)
        mic = PhotonMicroscope(psf=psf_true, signal_photons=photons,
                               background_photons=bg, read_noise_std=0.15)
        torch.manual_seed(100 + seed)
        counts = mic.sample(gt_t)["counts"]
        obs_img = torch.clamp((counts - bg) / photons, 0, 1)[0, 0].cpu().numpy().astype(np.float32)
    else:

        counts = torch.tensor(raw_n * photons)[None, None].to(device)
        obs_img = raw_n

    x_init = torch.tensor(obs_img)[None, None].to(device)
    res = blind_estimate(counts, x_init, modes=MODES, wavelength_nm=520.0, na=1.15,
                         pixel_size_nm=300.0, signal_photons=photons, background_photons=bg,
                         read_noise_std=0.15, tv_weight=0.08, n_outer=20, n_inner=8,
                         lr_x=0.01, lr_theta=0.2, seed=seed)
    est = res["theta"]
    x_est = res["x"][0, 0].detach().cpu().numpy()
    gt_np = gt_t[0, 0].detach().cpu().numpy()

    row = {"pair": pair, "inject": inject, "photons": photons, "seed": seed,
           "recon_psnr": float(peak_signal_noise_ratio(gt_np, x_est, data_range=1.0)),
           "obs_psnr": float(peak_signal_noise_ratio(gt_np, obs_img, data_range=1.0))}
    for m in MODES:
        row[f"true_{m}"] = true[m]
        row[f"est_{m}"] = est[m]
        row[f"err_{m}"] = min(abs(est[m] - true[m]), abs(est[m] + true[m])) if inject else float("nan")
    print(f"[stage B] {pair} inject={inject} seed={seed} photons={photons:.1f}", flush=True)
    for m in MODES:
        print(f"  {m:10s} true={true[m]:+.3f} est={est[m]:+.3f} err_sa={row[f'err_{m}']:.4f}", flush=True)
    print(f"  recon PSNR {row['recon_psnr']:.2f} (obs {row['obs_psnr']:.2f})", flush=True)
    return row


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--stage", default="A", choices=["A", "B"])
    ap.add_argument("--pair", default="Confocal_FISH_3")
    ap.add_argument("--photons", type=float, default=20.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--structures", nargs="+", default=["ER", "MT"])
    ap.add_argument("--multi", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    outdir = ROOT / "results" / "blind_calibration"
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    results = []
    if args.stage == "A":
        for sname in args.structures:
            for seed in args.seeds:
                results.append(stage_a(device, outdir, photons=args.photons, seed=seed,
                                       structure=sname, multi=args.multi))
        import pandas as pd
        pd.DataFrame(results).to_csv(outdir / f"stageA_photons{args.photons:g}_multi{int(args.multi)}.csv", index=False)
    else:
        for seed in args.seeds:
            results.append(stage_b(device, outdir, pair=args.pair, seed=seed, inject=True))
        import pandas as pd
        pd.DataFrame(results).to_csv(outdir / f"stageB_{args.pair}_inject.csv", index=False)
        results2 = []
        for seed in args.seeds:
            results2.append(stage_b(device, outdir, pair=args.pair, seed=seed, inject=False))
        pd.DataFrame(results2).to_csv(outdir / f"stageB_{args.pair}_realraw.csv", index=False)
    print("saved:", outdir)


if __name__ == "__main__":
    main()
