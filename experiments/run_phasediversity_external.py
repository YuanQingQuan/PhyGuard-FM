from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import tifffile

from phyguard.optics.psf import MicroscopyPSFConfig, make_psf, psf_to_otf
from phyguard.optics.zernike_ansi import (
    ANSI_ESTIMATED,
    ANSI_INDEX,
    ANSI_NAME,
    LAMBDA_UM,
    NA,
    PIXEL_SIZE_UM,
    ansi_coeffs_to_waves,
)

ROOT = Path(__file__).resolve().parents[1]


def _find_data_root() -> Path:

    cands = []
    env = os.environ.get("PHASEDIVERSITY_DATA")
    if env:
        cands.append(Path(env))
    for up in (2, 1):
        base = ROOT.parents[up] if up < len(ROOT.parents) else ROOT
        cands.append(base / "_external" / "PhaseDiversity_data" / "extracted"
                     / "Datasets" / "AO")
    cands.append(ROOT / "data" / "PhaseDiversity" / "Datasets" / "AO")
    for c in cands:
        if (c / DATASETS["beads"]).is_dir():
            return c
    tried = "\n  ".join(str(c) for c in cands)
    raise FileNotFoundError(
        "PhaseDiversity AO data not found. Download Datasets.zip from "
        "https://figshare.com/s/774457a804ee0686930d, extract "
        "Datasets/AO, and set PHASEDIVERSITY_DATA to that directory. Tried:\n  "
        + tried
    )

DATASETS = {
    "beads": "230904 AO2649 Beads",
    "cell": "230921 AO0057 U2OS_Cell",
}
MODES = tuple(ANSI_NAME[j] for j in ANSI_ESTIMATED)
BACKGROUND_OFFSET = 400.0
CROP = 512


def load_iteration(dataset: str, iteration: int, crop: int = CROP):

    folder = _find_data_root() / DATASETS[dataset]
    stack = tifffile.imread(folder / f"Iteration {iteration}" / "Stack.tif")
    stack = np.asarray(stack, dtype=np.float32)
    if stack.shape[0] != 5:
        raise ValueError(f"expected 5 frames, got {stack.shape[0]}")
    s = stack.shape[-1]

    start = int(round((s - crop) / 2.0 + 1)) - 1
    counts = stack[:, start:start + crop, start:start + crop] - BACKGROUND_OFFSET
    counts = np.maximum(counts, 0.01)
    coeffs = np.loadtxt(folder / "CoeffsIn.txt", dtype=np.float64)
    gt = tifffile.imread(folder / "GT.tif").astype(np.float32)
    return counts, coeffs, gt


def diversity_offsets(coeffs: np.ndarray) -> list[dict[str, float]]:

    offsets = [{}]
    for row in coeffs[1:5]:
        per_index = {col: float(v) for col, v in enumerate(row, start=1) if v != 0.0}
        offsets.append(ansi_coeffs_to_waves(per_index))
    return offsets


def wavefront_rms(values_by_mode: dict[str, float]) -> float:

    by_name = {ANSI_NAME[j]: (n, m) for j, (n, m) in ANSI_INDEX.items() if j in ANSI_NAME}
    total = 0.0
    for name, c in values_by_mode.items():
        n, m = by_name[name]
        norm = np.sqrt(n + 1) if m == 0 else np.sqrt(2 * (n + 1))
        total += (c / norm) ** 2
    return float(np.sqrt(total) * LAMBDA_UM * 1000.0)


def build_otf(theta_waves: dict, extra: dict | None, *, n: int,
              wavelength_nm: float, pixel_size_nm: float, na: float) -> torch.Tensor:
    combined = dict(theta_waves)
    for k, v in (extra or {}).items():
        combined[k] = combined.get(k, 0.0) + v
    cfg = MicroscopyPSFConfig(
        wavelength_nm=wavelength_nm, na=na, pixel_size_nm=pixel_size_nm,
        kernel_size=n, pupil_grid="exact", aberrations_waves=combined,
    )
    psf = make_psf(cfg)
    return psf_to_otf(psf, (n, n))


def expected_counts(x: torch.Tensor, otf: torch.Tensor, signal: float, bg: float) -> torch.Tensor:

    xc = x.clamp(0.0, 1.0)
    hx = torch.fft.ifft2(torch.fft.fft2(xc) * otf).real.clamp_min(0.0)
    return hx * float(signal) + float(bg)


def fit(
    counts_t: torch.Tensor,
    *,
    x_init: torch.Tensor,
    offsets: list[dict],
    modes: tuple[str, ...],
    signal: float,
    bg: float,
    read_noise_std: float,
    n_outer: int,
    n_inner: int,
    restarts: int,
    lr_x: float,
    lr_theta: float,
    tv_weight: float,
    seed: int,
    n: int,
    wavelength_nm: float,
    pixel_size_nm: float,
    na: float,
    truth: dict[str, float] | None = None,
):

    rng = np.random.default_rng(seed)

    def nll(theta, x):
        tot = 0.0
        for i in range(len(offsets)):
            o = build_otf(theta, offsets[i], n=n, wavelength_nm=wavelength_nm,
                          pixel_size_nm=pixel_size_nm, na=na)
            lam = expected_counts(x, o, signal, bg)
            c = counts_t[i]
            r = torch.clamp(lam, min=1e-6)
            tot = tot + (r - c * torch.log(r)).mean()
        return tot / len(offsets)

    def tv(x):
        return ((x[..., 1:, :] - x[..., :-1, :]).abs().mean()
                + (x[..., :, 1:] - x[..., :, :-1]).abs().mean())

    def best_object_for(theta_fixed: dict) -> float:

        x = x_init.clone().requires_grad_(True)
        t = {m: torch.tensor(float(theta_fixed.get(m, 0.0))) for m in modes}
        opt = torch.optim.Adam([x], lr=lr_x)
        for _ in range(n_outer * n_inner):
            opt.zero_grad()
            xc = x.clamp(0, 1)
            loss = nll(t, xc) + tv_weight * tv(xc)
            loss.backward()
            opt.step()
        with torch.no_grad():
            return float(nll(t, x.clamp(0, 1)).detach())

    truth_score = None
    if truth is not None:
        truth_score = best_object_for(truth)

    best = None
    for r in range(restarts):
        x = x_init.clone().requires_grad_(True)
        theta = {
            m: torch.tensor(float(rng.uniform(-0.3, 0.3)) if r > 0 else 0.0,
                            requires_grad=True)
            for m in modes
        }
        hist = []
        for _ in range(n_outer):
            opt_x = torch.optim.Adam([x], lr=lr_x)
            for _ in range(n_inner):
                opt_x.zero_grad()
                xc = x.clamp(0, 1)
                (nll(theta, xc) + tv_weight * tv(xc)).backward()
                opt_x.step()
            opt_t = torch.optim.Adam(list(theta.values()), lr=lr_theta)
            for _ in range(n_inner):
                opt_t.zero_grad()
                xc = x.detach().clamp(0, 1)
                nll(theta, xc).backward()
                opt_t.step()
            with torch.no_grad():
                hist.append(float(nll(theta, x.clamp(0, 1)).detach()))
        est = {m: float(theta[m].detach()) for m in modes}
        final = hist[-1]
        if best is None or final < best["loss"]:
            best = {"loss": final, "theta": est, "x": x.detach().clamp(0, 1),
                    "history": hist}
    best["truth_score"] = truth_score
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="beads", choices=sorted(DATASETS))
    ap.add_argument("--iteration", type=int, default=1)
    ap.add_argument("--n-outer", type=int, default=30)
    ap.add_argument("--n-inner", type=int, default=12)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--lr-x", type=float, default=0.05)
    ap.add_argument("--lr-theta", type=float, default=0.05)
    ap.add_argument("--tv-weight", type=float, default=0.01)
    ap.add_argument("--modes", default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=str(ROOT / "results" / "phasediversity_external"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    counts, coeffs, _ = load_iteration(args.dataset, args.iteration)
    offsets = diversity_offsets(coeffs)
    test_um = {col: float(v) for col, v in enumerate(coeffs[0], start=1) if v != 0.0}
    truth = ansi_coeffs_to_waves(test_um)

    modes = MODES if args.modes == "all" else tuple(args.modes.split(","))
    print(f"dataset={args.dataset} iteration={args.iteration} crop={counts.shape[-1]}")
    print("diversity offsets (waves):", offsets[1:])
    print("test aberration (ANSI idx -> um):", test_um)
    print(f"modes estimated: {len(modes)}")
    print(f"truth wavefront RMS: {wavefront_rms(truth):.1f} nm")

    n = counts.shape[-1]
    counts_t = torch.as_tensor(counts[:, None], dtype=torch.float32)

    raw = torch.as_tensor(counts[0], dtype=torch.float32)
    lo = torch.quantile(raw.flatten(), 0.1)
    hi = torch.quantile(raw.flatten(), 0.997)
    x_init = ((raw - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)[None, None]
    bg = float(torch.median(raw))
    signal = float(raw.mean() - bg) / max(float(x_init.mean()), 1e-6)
    print(f"signal_photons={signal:.1f} background_photons={bg:.1f} "
          f"x_init mean={float(x_init.mean()):.3f}")

    res = fit(counts_t, x_init=x_init, offsets=offsets, modes=modes,
              signal=signal, bg=bg, read_noise_std=0.0,
              n_outer=args.n_outer, n_inner=args.n_inner, restarts=args.restarts,
              lr_x=args.lr_x, lr_theta=args.lr_theta, tv_weight=args.tv_weight,
              seed=args.seed, n=n, wavelength_nm=LAMBDA_UM * 1000,
              pixel_size_nm=PIXEL_SIZE_UM * 1000, na=NA,
              truth={m: truth.get(m, 0.0) for m in modes})

    print(f"\nloss at blind estimate : {res['loss']:.6f}")
    print(f"loss at TRUE coefficients: {res['truth_score']:.6f}")
    converged = res["loss"] < res["truth_score"]
    print(f"blind fit beats truth-scored object fit: {converged}")
    if not converged:
        print("  -> the optimiser did not converge; this run says NOTHING about "
              "model error and must not be reported as a validation result.")
    print(f"loss history tail: {[round(v, 4) for v in res['history'][-5:]]}")

    print(f"\n{'ANSI':>5s} {'mode':>10s} {'truth(nm)':>10s} {'est(nm)':>10s} {'err(nm)':>9s}")
    errs = []
    for m in modes:
        t_nm = truth.get(m, 0.0) * LAMBDA_UM * 1000
        e_nm = res["theta"][m] * LAMBDA_UM * 1000
        errs.append(e_nm - t_nm)
        flag = "  <<" if abs(t_nm) > 1e-9 else ""
        print(f"{'':>5s} {m:>10s} {t_nm:10.1f} {e_nm:10.1f} {e_nm - t_nm:9.1f}{flag}")
    rms_err = float(np.sqrt(np.mean(np.square(errs))))
    print(f"\nper-mode coefficient RMS error: {rms_err:.1f} nm")
    print(f"wavefront RMS of estimate: {wavefront_rms(res['theta']):.1f} nm "
          f"(truth {wavefront_rms(truth):.1f} nm)")
    print("reference: the authors report their own loop reaches a flat wavefront "
          "to within 15 nm RMS.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.dataset}_it{args.iteration}"
    np.save(outdir / f"{tag}_theta_est.npy", np.array([res["theta"][m] for m in modes]))
    np.save(outdir / f"{tag}_x_est.npy", res["x"][0, 0].detach().numpy())
    (outdir / f"{tag}.json").write_text(json.dumps({
        "dataset": args.dataset, "iteration": args.iteration, "modes": list(modes),
        "truth_um": {str(k): v for k, v in test_um.items()},
        "diversity_offsets": offsets,
        "theta_est_waves": res["theta"], "loss_est": res["loss"],
        "loss_at_truth": res["truth_score"], "converged": converged,
        "per_mode_error_nm": dict(zip(modes, errs)),
        "coeff_rms_error_nm": rms_err,
        "wavefront_rms_nm_est": wavefront_rms(res["theta"]),
        "wavefront_rms_nm_truth": wavefront_rms(truth),
        "signal_photons": signal, "background_photons": bg,
        "settings": vars(args),
    }, indent=2))
    print(f"\nwrote {outdir / (tag + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
