from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import tifffile

from phyguard.optics.psf import MicroscopyPSFConfig
from phyguard.optics.psf3d import make_psf_3d, forward_3d_stack
from phyguard.likelihood import poisson_gaussian_nll

from experiments.run_axial_prospective import (
    NA, WAVELENGTH_NM, PIXEL_NM, NZ, DZ_NM, KERNEL, MODES, SIGMA,
    THRESHOLD, PHOTONS, BG, READ, draw_true, rms_of, load_volume,
    estimate_axial, axial_fisher,
)

OUTDIR = ROOT / "results" / "axial_basin"
R_INIT = 32
MAG_SCAN = 0.50
MAG_CONTROL = 0.30
TRANSITION_MAGS = [0.36, 0.40, 0.44]
TRANSITION_SEEDS = list(range(10))
INIT_RANGE = 0.6


def make_counts(structure, true, seed, dev):
    psf_true = make_psf_3d(MicroscopyPSFConfig(
        wavelength_nm=WAVELENGTH_NM, na=NA, pixel_size_nm=PIXEL_NM, kernel_size=KERNEL,
        aberrations_waves=true), nz=NZ, dz_nm=DZ_NM, device=dev)
    mu = forward_3d_stack(structure, psf_true) * PHOTONS + BG
    torch.manual_seed(80 + seed)
    return torch.poisson(mu), mu


def estimate_from_init(counts, structure, init, n_steps=400, lr=0.08):
    est = {m: torch.tensor(float(init[m]), requires_grad=True, device=counts.device)
           for m in MODES}
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


def wilson(k, n, z=1.959964):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def append_csv(path, row):
    df = pd.DataFrame([row])
    header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=header)


def nll_at(counts, structure, coeffs, dev):
    psf = make_psf_3d(MicroscopyPSFConfig(
        wavelength_nm=WAVELENGTH_NM, na=NA, pixel_size_nm=PIXEL_NM, kernel_size=KERNEL,
        aberrations_waves=coeffs), nz=NZ, dz_nm=DZ_NM, device=dev)
    mu = forward_3d_stack(structure, psf) * PHOTONS + BG
    return float(poisson_gaussian_nll(counts, mu, READ))


def stage_initscan(device):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    structure = load_volume().to(dev)
    path = OUTDIR / "init_scan.csv"
    for mag, seeds in ((MAG_SCAN, [0, 1, 2, 3, 4]), (MAG_CONTROL, [1, 4])):
        for seed in seeds:
            true = draw_true(mag, seed)
            counts, _ = make_counts(structure, true, seed, dev)
            rng = np.random.default_rng(1000 + 100 * seed + int(mag * 100))
            for r in range(R_INIT):
                init = {m: float(rng.uniform(-INIT_RANGE, INIT_RANGE)) for m in MODES}
                est, nll = estimate_from_init(counts, structure, init)
                row = {"magnitude": mag, "seed": seed, "restart": r, "rms": rms_of(true),
                       "terminal_nll": nll}
                plain2, sw2 = 0.0, 0.0
                for m in MODES:
                    e = est[m] - true[m]
                    row[f"true_{m}"] = true[m]
                    row[f"init_{m}"] = init[m]
                    row[f"est_{m}"] = est[m]
                    row[f"err_{m}"] = e
                    plain2 += e ** 2 * SIGMA[MODES.index(m)] ** 2
                    sw2 += min(e ** 2, (est[m] + true[m]) ** 2) * SIGMA[MODES.index(m)] ** 2
                row["err_rms"] = float(np.sqrt(plain2))
                row["err_rms_signaware"] = float(np.sqrt(sw2))
                append_csv(path, row)
            print(f"initscan mag={mag} seed={seed} done ({R_INIT} restarts)", flush=True)
    print("saved", path)


def stage_transition(device, n_steps=400, lr=0.08):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    structure = load_volume().to(dev)
    path = OUTDIR / "transition_scan.csv"
    for mag in TRANSITION_MAGS:
        for seed in TRANSITION_SEEDS:
            true = draw_true(mag, seed)
            counts, _ = make_counts(structure, true, seed, dev)
            est, nll = estimate_axial(counts, structure, n_steps, lr)
            plain2, sw2 = 0.0, 0.0
            row = {"magnitude": mag, "seed": seed, "rms": rms_of(true), "terminal_nll": nll}
            for i, m in enumerate(MODES):
                e = est[m] - true[m]
                row[f"err_{m}"] = e
                plain2 += e ** 2 * SIGMA[i] ** 2
                sw2 += min(e ** 2, (est[m] + true[m]) ** 2) * SIGMA[i] ** 2
            row["err_rms"] = float(np.sqrt(plain2))
            row["err_rms_signaware"] = float(np.sqrt(sw2))
            append_csv(path, row)
        n = len(TRANSITION_SEEDS)
        k = int((pd.read_csv(path).query("magnitude == @mag")["err_rms"] < THRESHOLD).sum())
        lo, hi = wilson(k, n)
        print(f"transition mag={mag}: success {k}/{n} (Wilson 95% [{lo:.2f},{hi:.2f}])",
              flush=True)
    print("saved", path)


def stage_gatesweep(device, n_steps=400, lr=0.08):

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    structure = load_volume().to(dev)
    path = OUTDIR / "gatesweep.csv"
    for mag in TRANSITION_MAGS:
        for seed in TRANSITION_SEEDS:
            true = draw_true(mag, seed)
            counts, _ = make_counts(structure, true, seed, dev)
            rng = np.random.default_rng(1000 + 100 * seed + int(mag * 100))
            for r in range(R_INIT):
                init = {m: float(rng.uniform(-INIT_RANGE, INIT_RANGE)) for m in MODES}
                est, nll = estimate_from_init(counts, structure, init,
                                              n_steps, lr)
                row = {"magnitude": mag, "seed": seed, "restart": r,
                       "rms": rms_of(true), "terminal_nll": nll}
                plain2, sw2 = 0.0, 0.0
                for i, m in enumerate(MODES):
                    e = est[m] - true[m]
                    row[f"true_{m}"] = true[m]
                    row[f"init_{m}"] = init[m]
                    row[f"est_{m}"] = est[m]
                    row[f"err_{m}"] = e
                    plain2 += e ** 2 * SIGMA[i] ** 2
                    sw2 += min(e ** 2, (est[m] + true[m]) ** 2) * SIGMA[i] ** 2
                row["err_rms"] = float(np.sqrt(plain2))
                row["err_rms_signaware"] = float(np.sqrt(sw2))
                append_csv(path, row)
            print(f"gatesweep mag={mag} seed={seed} done ({R_INIT} restarts)",
                  flush=True)
    print("saved", path)


def stage_continuation(device, n_steps=400, lr=0.08):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    structure = load_volume().to(dev)
    path = OUTDIR / "continuation.csv"
    for seed in [0, 1, 2, 3, 4]:
        true_lo = draw_true(MAG_CONTROL, seed)
        counts_lo, _ = make_counts(structure, true_lo, seed, dev)
        warm, _ = estimate_axial(counts_lo, structure, n_steps, lr)
        true_hi = draw_true(MAG_SCAN, seed)
        counts_hi, _ = make_counts(structure, true_hi, seed, dev)
        for arm, init in (("cold_zero", {m: 0.0 for m in MODES}),
                          ("warm_from_0.30", warm)):
            est, nll = estimate_from_init(counts_hi, structure, init, n_steps, lr)
            plain2, sw2 = 0.0, 0.0
            row = {"seed": seed, "arm": arm, "rms": rms_of(true_hi), "terminal_nll": nll}
            for i, m in enumerate(MODES):
                e = est[m] - true_hi[m]
                row[f"err_{m}"] = e
                plain2 += e ** 2 * SIGMA[i] ** 2
                sw2 += min(e ** 2, (est[m] + true_hi[m]) ** 2) * SIGMA[i] ** 2
            row["err_rms"] = float(np.sqrt(plain2))
            row["err_rms_signaware"] = float(np.sqrt(sw2))
            append_csv(path, row)
        print(f"continuation seed={seed} done", flush=True)
    print("saved", path)


def load_volume_shift(z_shift=0):

    vol = tifffile.imread(ROOT / "data" / "cocoa_beads" / "ref_no_aberration.tif").astype(np.float32)
    zi = vol.shape[0] // 2 + z_shift
    sub = vol[zi - 4:zi + 5, 160:288, 160:288]
    from experiments.run_real_optical_pilot import normalize
    return torch.tensor(np.stack([normalize(sub[i], p_low=0.5, p_high=99.5) for i in range(NZ)]))


def stage_mechanism(device, n_steps=400, lr=0.08):

    import tifffile
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    path = OUTDIR / "mechanism.csv"
    for z_shift in (0, 3, -3):
        structure = load_volume_shift(z_shift)
        for seed in [0, 1, 2, 3, 4]:
            true = draw_true(MAG_SCAN, seed)
            F = axial_fisher(structure, true)
            th = np.array([true[m] for m in MODES])
            delta = float(2.0 * th @ F @ th)
            row = {"z_shift": z_shift, "seed": seed, "probe": "noiseless_gap",
                   "rms": rms_of(true), "well_depth_gap": delta}
            if z_shift == 0:

                ref = pd.read_csv(OUTDIR / "zero_init_repeat.csv")
                row["zero_init_success_frac"] = float(
                    (ref.query("seed == @seed").err_rms < THRESHOLD).mean())
            append_csv(path, row)
        print(f"mechanism noiseless z_shift={z_shift} done", flush=True)

    structure = load_volume_shift(3).to(dev)
    for seed in [0, 1, 2, 3, 4]:
        true = draw_true(MAG_SCAN, seed)
        counts, _ = make_counts(structure, true, seed, dev)
        est, nll = estimate_axial(counts, structure, n_steps, lr)
        plain2 = 0.0
        row = {"z_shift": 3, "seed": seed, "probe": "empirical_shifted",
               "rms": rms_of(true), "terminal_nll": nll}
        for i, m in enumerate(MODES):
            e = est[m] - true[m]
            row[f"err_{m}"] = e
            plain2 += e ** 2 * SIGMA[i] ** 2
        row["err_rms"] = float(np.sqrt(plain2))
        row["zero_init_success_frac"] = float(plain2 ** 0.5 < THRESHOLD)
        append_csv(path, row)
        print(f"mechanism empirical shifted seed={seed} err_rms={row['err_rms']:.4f}",
              flush=True)
    print("saved", path)


def stage_repeat(device, n_rep=8, n_steps=400, lr=0.08):

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    structure = load_volume().to(dev)
    path = OUTDIR / "zero_init_repeat.csv"
    for seed in [0, 1, 2, 3, 4]:
        true = draw_true(MAG_SCAN, seed)
        counts, _ = make_counts(structure, true, seed, dev)
        for rep in range(n_rep):
            est, nll = estimate_from_init(counts, structure, {m: 0.0 for m in MODES},
                                          n_steps, lr)
            plain2 = 0.0
            row = {"seed": seed, "rep": rep, "rms": rms_of(true), "terminal_nll": nll}
            for i, m in enumerate(MODES):
                e = est[m] - true[m]
                row[f"err_{m}"] = e
                plain2 += e ** 2 * SIGMA[i] ** 2
            row["err_rms"] = float(np.sqrt(plain2))
            append_csv(path, row)
        g = pd.read_csv(path).query("seed == @seed")
        k = int((g.err_rms < THRESHOLD).sum())
        print(f"repeat seed={seed}: success {k}/{n_rep}", flush=True)
    print("saved", path)


def stage_profile(device, seed=0, k_fresh=8, n_line=41):

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    structure = load_volume().to(dev)
    df = pd.read_csv(OUTDIR / "init_scan.csv")
    sub = df[(df.magnitude == MAG_SCAN) & (df.seed == seed)].copy()
    good = sub[sub.err_rms < THRESHOLD]
    bad = sub[sub.err_rms > 5 * THRESHOLD]
    if len(good) == 0 or len(bad) == 0:
        raise SystemExit(f"seed {seed}: need both clusters (good={len(good)}, bad={len(bad)})")

    s = good.loc[good.terminal_nll.idxmin()]
    f = bad.loc[bad.terminal_nll.idxmin()]
    ts = np.array([s[f"est_{m}"] for m in MODES])
    tf = np.array([f[f"est_{m}"] for m in MODES])

    true = draw_true(MAG_SCAN, seed)
    counts, mu_true = make_counts(structure, true, seed, dev)

    ts_line = np.linspace(0.0, 1.0, n_line)
    nll_fixed, nll_fresh = [], []
    for t in ts_line:
        th = {m: float((1 - t) * ts[i] + t * tf[i]) for i, m in enumerate(MODES)}
        nll_fixed.append(nll_at(counts, structure, th, dev))
        fresh = []
        g = torch.Generator(device=dev).manual_seed(9000 + int(t * 1000))
        for _ in range(k_fresh):
            c = torch.poisson(mu_true, generator=g)
            fresh.append(nll_at(c, structure, th, dev))
        nll_fresh.append(fresh)


    fim = {}
    for name, th in (("successful", ts), ("failed", tf)):
        F = axial_fisher(structure.cpu(), {m: float(th[i]) for i, m in enumerate(MODES)})
        eig = np.linalg.eigvalsh(F)
        fim[name] = {"lambda_min": float(eig[0]), "lambda_max": float(eig[-1]),
                     "cond": float(eig[-1] / max(eig[0], 1e-30)),
                     "crb_diag_max": float(np.sqrt(np.max(np.diag(np.linalg.pinv(F)))))}

    payload = {
        "seed": seed,
        "true": {m: float(true[m]) for m in MODES},
        "theta_successful": {m: float(ts[i]) for i, m in enumerate(MODES)},
        "theta_failed": {m: float(tf[i]) for i, m in enumerate(MODES)},
        "err_rms_successful": float(s["err_rms"]),
        "err_rms_failed": float(f["err_rms"]),
        "terminal_nll_successful": float(s["terminal_nll"]),
        "terminal_nll_failed": float(f["terminal_nll"]),
        "nll_on_fixed_counts": {"t": ts_line.tolist(), "nll": nll_fixed},
        "nll_on_fresh_realizations": {"t": ts_line.tolist(),
                                      "nll_mean": np.mean(nll_fresh, axis=1).tolist(),
                                      "nll_std": np.std(nll_fresh, axis=1).tolist()},
        "fim_at_endpoints": fim,
    }
    (OUTDIR / "profile.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items()
                      if k not in ("nll_on_fixed_counts", "nll_on_fresh_realizations")},
                     indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["initscan", "transition", "gatesweep", "continuation",
                             "repeat", "mechanism", "profile"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if a.stage == "profile":
        stage_profile(a.device, seed=a.seed)
    else:
        {"initscan": stage_initscan, "transition": stage_transition,
         "gatesweep": stage_gatesweep, "continuation": stage_continuation,
         "repeat": stage_repeat, "mechanism": stage_mechanism}[a.stage](a.device)
