from __future__ import annotations

from pathlib import Path
import sys
import json
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from phyguard.optics.psf import MicroscopyPSFConfig, make_psf
from phyguard.optics.forward import PhotonMicroscope
from phyguard.estimation.info import parameter_fisher, parameter_fisher_multiframe
from phyguard.estimation.calibration import estimate_aberrations_multiframe
from experiments.run_biosr_identifiability_sweep import load_gt

MODES = ["defocus", "astig_x", "coma_x", "spherical"]
WL, NA, PX, KS = 520.0, 1.2, 62.6, 41

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def timed(fn, n_repeat=1, warmup=True):

    if warmup:
        fn()
    ts = []
    for _ in range(n_repeat):
        if DEV.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if DEV.type == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.min(ts)), float(np.mean(ts))


def main():
    outdir = ROOT / "results" / "runtime_benchmark"
    outdir.mkdir(parents=True, exist_ok=True)
    gt = torch.tensor(load_gt("ER", 1, 1))[None, None].to(DEV)
    true = {"defocus": -0.11, "astig_x": 0.06, "coma_x": 0.17, "spherical": 0.19}
    coeffs = {m: torch.tensor(true[m]) for m in MODES}
    rows = []

    def rec(step, t_min, t_mean):
        rows.append(dict(step=step, seconds_min=round(t_min, 2),
                         seconds_mean=round(t_mean, 2)))
        print(f"{step:44s} {t_min:8.2f} s", flush=True)


    t = timed(lambda: parameter_fisher(coeffs, gt, wavelength_nm=WL, na=NA, pixel_size_nm=PX,
                                       signal_photons=50.0, background_photons=7.5,
                                       read_noise_std=0.15), n_repeat=3)
    rec("Fisher matrix, single frame (4 modes)", *t)
    t = timed(lambda: parameter_fisher_multiframe(coeffs, gt, defocus_offsets=[-1, -.5, 0, .5, 1],
                                                  wavelength_nm=WL, na=NA, pixel_size_nm=PX,
                                                  signal_photons=10.0, background_photons=1.5,
                                                  read_noise_std=0.15), n_repeat=3)
    rec("Fisher matrix, 5-frame diversity", *t)


    def counts_for(offsets, s):
        bg = max(0.2, 0.15 * s)
        cs = []

        torch.manual_seed(7)
        for off in offsets:
            th = dict(true); th["defocus"] = th["defocus"] + off
            psf = make_psf(MicroscopyPSFConfig(wavelength_nm=WL, na=NA, pixel_size_nm=PX,
                                               kernel_size=KS, aberrations_waves=th)).to(DEV)
            mic = PhotonMicroscope(psf=psf, signal_photons=s, background_photons=bg,
                                   read_noise_std=0.15)
            cs.append(mic.sample(gt)["counts"])
        return torch.cat(cs, dim=0)


    c1 = counts_for([0.0], 20.0)
    t = timed(lambda: estimate_aberrations_multiframe(
        c1, gt, modes=MODES, defocus_offsets=[0.0], wavelength_nm=WL, na=NA, pixel_size_nm=PX,
        signal_photons=20.0, background_photons=3.0, read_noise_std=0.15,
        n_steps=500, n_restarts=4, seed=0), n_repeat=1)
    rec("Non-blind estimate, 1 frame", *t)

    c5 = counts_for([-1, -.5, 0, .5, 1], 4.0)
    t = timed(lambda: estimate_aberrations_multiframe(
        c5, gt, modes=MODES, defocus_offsets=[-1, -.5, 0, .5, 1], wavelength_nm=WL, na=NA,
        pixel_size_nm=PX, signal_photons=4.0, background_photons=0.6, read_noise_std=0.15,
        n_steps=500, n_restarts=4, seed=0), n_repeat=1)
    rec("Non-blind estimate, 5-frame diversity", *t)


    from experiments.run_blind_calibration import blind_estimate
    from skimage.restoration import richardson_lucy
    c3 = counts_for([-0.5, 0.0, 0.5], 20.0)
    nom = make_psf(MicroscopyPSFConfig(wavelength_nm=WL, na=NA, pixel_size_nm=PX,
                                       kernel_size=KS)).cpu().numpy()
    center = c3[1, 0].cpu().numpy() / 20.0
    rl0 = richardson_lucy(np.clip(center / max(center.max(), 1e-9), 0, 1), nom, num_iter=15, clip=False)
    x_init = torch.tensor(np.clip(rl0 / max(rl0.max(), 1e-9), 0, 1), dtype=torch.float32)[None, None].to(DEV)

    def run_blind():
        return blind_estimate(c3, x_init, modes=MODES, wavelength_nm=WL, na=NA, pixel_size_nm=PX,
                              signal_photons=20.0, background_photons=3.0, read_noise_std=0.15,
                              tv_weight=0.10, n_outer=15, n_inner=6, lr_x=0.01, lr_theta=0.15,
                              seed=0, defocus_offsets=[-0.5, 0.0, 0.5])
    t = timed(run_blind, n_repeat=1)
    rec("Blind estimate, 3-frame diversity (BioSR ER)", *t)


    from experiments.run_cocoa_validation_v3 import estimate as cocoa_estimate
    from experiments.run_cocoa_validation_v2 import load_frames

    def run_cocoa():
        th = {}
        for fname in ["aberrated_75nm.tif", "ref_no_aberration.tif"]:
            th[fname] = cocoa_estimate(load_frames(fname), DEV, seed=0)
        return th
    t = timed(run_cocoa, n_repeat=1)
    rec("CoCoA differential estimate (2 stacks, 7 modes)", *t)


    from phyguard.optics.psf3d import make_psf_3d, forward_3d_stack
    import tifffile
    from experiments.run_real_optical_pilot import normalize
    vol = tifffile.imread(ROOT / "data" / "cocoa_beads" / "ref_no_aberration.tif").astype(np.float32)
    zi = vol.shape[0] // 2
    sub = vol[zi - 4:zi + 5, 160:288, 160:288]
    structure3d = torch.tensor(np.stack([normalize(sub[i], p_low=0.5, p_high=99.5)
                                         for i in range(9)])).to(DEV)
    from experiments.run_3d_calibration import estimate_3d
    psf_t = make_psf_3d(MicroscopyPSFConfig(wavelength_nm=515.0, na=1.1, pixel_size_nm=86.0,
                                            kernel_size=41,
                                            aberrations_waves={"defocus": 0.1, "astig_x": 0.08}),
                        nz=9, dz_nm=200.0, device=DEV)
    mu3 = forward_3d_stack(structure3d, psf_t) * 30.0 + 4.5
    torch.manual_seed(0)
    counts3d = torch.poisson(mu3)
    t = timed(lambda: estimate_3d(counts3d, structure3d, 30.0, 4.5), n_repeat=1)
    rec("3D estimate, 9-plane stack 128x128", *t)


    from skimage.restoration import richardson_lucy as rl
    obs = c1[0, 0].cpu().numpy()
    obsn = obs / max(obs.max(), 1e-9)
    t = timed(lambda: rl(obsn, nom, num_iter=20, clip=False), n_repeat=3)
    rec("Richardson-Lucy, 20 iterations (CPU)", *t)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "runtime.csv", index=False)
    meta = dict(
        device=torch.cuda.get_device_name(0) if DEV.type == "cuda" else "CPU",
        torch=torch.__version__, python=sys.version.split()[0],
        image_size="256x256 (BioSR) / 128x128x9 (3D) / 256x256 crop (CoCoA)",
        date=time.strftime("%Y-%m-%d %H:%M"),
    )
    (outdir / "runtime.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    print("saved:", outdir)


if __name__ == "__main__":
    main()
