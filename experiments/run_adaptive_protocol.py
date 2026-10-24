from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from skimage.restoration import richardson_lucy

from phyguard.optics.psf import MicroscopyPSFConfig, make_psf
from phyguard.optics.forward import PhotonMicroscope
from phyguard.estimation.protocol import select_next_offset, fixed_offsets, candidate_offsets
from phyguard.estimation.info import parameter_fisher, crb_identifiable
from experiments.run_blind_calibration import blind_estimate, MODES
from experiments.run_real_optical_pilot import normalize
from experiments.run_biosr_identifiability_sweep import load_gt


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--structure", default="ER")
    ap.add_argument("--photons", type=float, default=20.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--n-frames", type=int, default=4)
    ap.add_argument("--true-max", type=float, default=0.25)
    ap.add_argument("--fixed-step", type=float, default=0.3)
    ap.add_argument("--criterion", default="aopt",
                    choices=["aopt", "eopt", "trace"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    outdir = ROOT / "results" / "adaptive_protocol"
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    gt = torch.tensor(load_gt(args.structure, 1, 1))[None, None].to(device)
    photons = args.photons
    bg = max(0.2, 0.15 * photons)
    import pandas as pd
    rows = []
    for seed in args.seeds:
        rng = np.random.default_rng(100 + seed)
        true = {m: float(rng.uniform(-args.true_max, args.true_max)) for m in MODES}

        def acquire(off):
            th = dict(true); th["defocus"] = th["defocus"] + off
            psf = make_psf(MicroscopyPSFConfig(
                wavelength_nm=520.0, na=1.2, pixel_size_nm=62.6,
                kernel_size=41, aberrations_waves=th,
            )).to(device)
            mic = PhotonMicroscope(psf=psf, signal_photons=photons,
                                   background_photons=bg, read_noise_std=0.15)
            torch.manual_seed(200 + seed)
            return mic.sample(gt)["counts"]

        def init_x(counts):
            if counts.ndim == 4:
                img = counts[0, 0]
            elif counts.ndim == 3:
                img = counts[0]
            else:
                img = counts
            obs_img = torch.clamp((img - bg) / photons, 0, 1).cpu().numpy().astype(np.float32)
            nom = make_psf(MicroscopyPSFConfig(
                wavelength_nm=520.0, na=1.2, pixel_size_nm=62.6, kernel_size=41,
            )).detach().cpu().numpy()
            rl = richardson_lucy(np.clip(obs_img, 0, 1), nom, num_iter=15, clip=False)
            return torch.tensor(normalize(np.clip(rl, 0, None), p_low=0.1, p_high=99.7))[None, None].to(device)

        def estimate(counts_stack, offsets, est_seed):
            res = blind_estimate(
                counts_stack, init_x(counts_stack[0] if len(offsets) > 1 else counts_stack),
                modes=MODES, wavelength_nm=520.0, na=1.2, pixel_size_nm=62.6,
                signal_photons=photons, background_photons=bg, read_noise_std=0.15,
                tv_weight=0.10, n_outer=10, n_inner=5, lr_x=0.01, lr_theta=0.2,
                seed=est_seed, defocus_offsets=offsets if len(offsets) > 1 else None,
            )
            return res


        offsets_ad = [0.0]
        frames = [acquire(0.0)]
        est = None
        for f in range(1, args.n_frames):
            counts_stack = torch.cat(frames, dim=0)
            est = estimate(counts_stack, offsets_ad, 1000 + seed + f)
            theta_t = {m: torch.tensor(est["theta"][m], requires_grad=True, device=device) for m in MODES}

            F_acc = np.zeros((len(MODES), len(MODES)))
            for off in offsets_ad:
                th_o = dict(theta_t)
                th_o["defocus"] = th_o["defocus"] + float(off)
                Fi = parameter_fisher(th_o, gt, wavelength_nm=520.0, na=1.2,
                                      pixel_size_nm=62.6, signal_photons=photons,
                                      background_photons=bg, read_noise_std=0.15)
                if Fi is not None:
                    F_acc += Fi
            nxt, gains = select_next_offset(
                theta_t, gt, candidate_offsets(step=0.25),
                wavelength_nm=520.0, na=1.2, pixel_size_nm=62.6,
                signal_photons=photons, background_photons=bg, read_noise_std=0.15,
                criterion=args.criterion, accumulated=F_acc,
            )
            offsets_ad.append(nxt)
            frames.append(acquire(nxt))
        counts_stack = torch.cat(frames, dim=0)
        est_final = estimate(counts_stack, offsets_ad, 1000 + seed + args.n_frames)

        EVEN = {"defocus", "astig_x", "spherical"}
        err_ad = {m: (min(abs(est_final["theta"][m] - true[m]), abs(est_final["theta"][m] + true[m]))
                      if m in EVEN else abs(est_final["theta"][m] - true[m])) for m in MODES}


        off_fx = fixed_offsets(args.n_frames, args.fixed_step)
        frames_fx = [acquire(o) for o in off_fx]
        est_fx = estimate(torch.cat(frames_fx, dim=0), off_fx, 2000 + seed)
        err_fx = {m: (min(abs(est_fx["theta"][m] - true[m]), abs(est_fx["theta"][m] + true[m]))
                      if m in EVEN else abs(est_fx["theta"][m] - true[m])) for m in MODES}

        row = {"seed": seed, "n_frames": args.n_frames, "criterion": args.criterion,
               "true_max": args.true_max, "fixed_step": args.fixed_step,
               "adaptive_offsets": str([round(o, 2) for o in offsets_ad]),
               "fixed_offsets": str(off_fx)}
        for m in MODES:
            row[f"ad_err_{m}"] = err_ad[m]
            row[f"fx_err_{m}"] = err_fx[m]
            row[f"gain_{m}"] = err_fx[m] - err_ad[m]
        rows.append(row)
        print(row, flush=True)

    df = pd.DataFrame(rows)
    tag = args.tag or f"{args.structure}"
    df.to_csv(outdir / f"adaptive_vs_fixed_{tag}.csv", index=False)
    print(df.to_string(index=False))
    print("saved:", outdir)


if __name__ == "__main__":
    main()
