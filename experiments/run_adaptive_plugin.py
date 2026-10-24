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
from phyguard.estimation.info import parameter_fisher
from experiments.run_blind_calibration import blind_estimate, MODES
from experiments.run_real_optical_pilot import normalize
from experiments.run_biosr_identifiability_sweep import load_gt

EVEN = ["defocus", "astig_x", "spherical"]


def block_errors(est, true):

    s = 1.0 if sum(est[m] * true[m] for m in EVEN) >= 0 else -1.0
    return {m: (abs(est[m] - s * true[m]) if m in EVEN else abs(est[m] - true[m]))
            for m in MODES}, s


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--structure", default="ER")
    ap.add_argument("--photons", type=float, default=20.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-frames", type=int, default=4)
    ap.add_argument("--true-max", type=float, nargs="+", default=[0.25, 0.05])
    ap.add_argument("--fixed-step", type=float, default=0.3)
    ap.add_argument("--criterion", default="aopt")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    outdir = ROOT / "results" / "adaptive_protocol_plugin"
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    gt = torch.tensor(load_gt(args.structure, 1, 1))[None, None].to(device)
    photons = args.photons
    bg = max(0.2, 0.15 * photons)
    import pandas as pd
    import time

    def acquire(true, off, seed):
        th = dict(true); th["defocus"] = th["defocus"] + float(off)
        psf = make_psf(MicroscopyPSFConfig(
            wavelength_nm=520.0, na=1.2, pixel_size_nm=62.6,
            kernel_size=41, aberrations_waves=th)).to(device)
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
        return blind_estimate(
            counts_stack, init_x(counts_stack[0] if len(offsets) > 1 else counts_stack),
            modes=MODES, wavelength_nm=520.0, na=1.2, pixel_size_nm=62.6,
            signal_photons=photons, background_photons=bg, read_noise_std=0.15,
            tv_weight=0.10, n_outer=10, n_inner=5, lr_x=0.01, lr_theta=0.2,
            seed=est_seed, defocus_offsets=offsets if len(offsets) > 1 else None,
        )

    def fisher_design(theta_t, structure_for_fisher, offsets_so_far):
        F_acc = np.zeros((len(MODES), len(MODES)))
        for off in offsets_so_far:
            th_o = dict(theta_t)
            th_o["defocus"] = th_o["defocus"] + float(off)
            Fi = parameter_fisher(th_o, structure_for_fisher, wavelength_nm=520.0, na=1.2,
                                  pixel_size_nm=62.6, signal_photons=photons,
                                  background_photons=bg, read_noise_std=0.15)
            if Fi is not None:
                F_acc += Fi
        nxt, gains = select_next_offset(
            theta_t, structure_for_fisher, candidate_offsets(step=0.25),
            wavelength_nm=520.0, na=1.2, pixel_size_nm=62.6,
            signal_photons=photons, background_photons=bg, read_noise_std=0.15,
            criterion=args.criterion, accumulated=F_acc,
        )
        return nxt

    rows = []
    for tm in args.true_max:
        for seed in args.seeds:
            rng = np.random.default_rng(100 + seed)
            true = {m: float(rng.uniform(-tm, tm)) for m in MODES}
            t0 = time.time()

            first = acquire(true, 0.0, seed)
            finals = {}
            for arm in ["oracle", "plugin"]:
                offsets = [0.0]
                frames = [first]
                est = None
                for f in range(1, args.n_frames):
                    counts_stack = torch.cat(frames, dim=0)
                    est = estimate(counts_stack, offsets, 1000 + seed + f)
                    theta_t = {m: torch.tensor(est["theta"][m], requires_grad=True,
                                               device=device) for m in MODES}
                    design_structure = gt if arm == "oracle" else est["x"].detach()
                    nxt = fisher_design(theta_t, design_structure, offsets)
                    offsets.append(nxt)
                    frames.append(acquire(true, nxt, seed))
                est_final = estimate(torch.cat(frames, dim=0), offsets,
                                     1000 + seed + args.n_frames)
                finals[arm] = (offsets, est_final)
            off_fx = fixed_offsets(args.n_frames, args.fixed_step)
            frames_fx = [acquire(true, o, seed) for o in off_fx]
            est_fx = estimate(torch.cat(frames_fx, dim=0), off_fx, 2000 + seed)
            finals["fixed"] = (off_fx, est_fx)

            row = {"structure": args.structure, "true_max": tm, "seed": seed,
                   "photons": photons,
                   "oracle_offsets": str([round(o, 2) for o in finals["oracle"][0]]),
                   "plugin_offsets": str([round(o, 2) for o in finals["plugin"][0]]),
                   "fixed_offsets": str(off_fx),
                   "elapsed_s": round(time.time() - t0, 1)}
            for arm in ["oracle", "plugin", "fixed"]:
                errs, s = block_errors(finals[arm][1]["theta"], true)
                for m in MODES:
                    row[f"{arm}_err_{m}"] = errs[m]
                    row[f"{arm}_est_{m}"] = finals[arm][1]["theta"][m]
                row[f"{arm}_sign"] = s
            for m in MODES:
                row[f"true_{m}"] = true[m]
            rows.append(row)
            print({k: v for k, v in row.items() if "est_" not in k}, flush=True)
            pd.DataFrame(rows).to_csv(
                outdir / f"adaptive_plugin_{args.structure}.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / f"adaptive_plugin_{args.structure}.csv", index=False)
    for tm in args.true_max:
        sub = df[df.true_max == tm]
        print(f"\n=== true_max={tm} (mean |err| over seeds, block scoring) ===")
        for arm in ["oracle", "plugin", "fixed"]:
            e = {m: float(sub[f"{arm}_err_{m}"].mean()) for m in MODES}
            print(f"  {arm:7s}: " + "  ".join(f"{m} {v:.4f}" for m, v in e.items()))
    print("saved:", outdir / f"adaptive_plugin_{args.structure}.csv")


if __name__ == "__main__":
    main()
