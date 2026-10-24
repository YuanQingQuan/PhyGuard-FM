from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from phyguard.estimation.calibration import estimate_aberrations_multiframe
from phyguard.estimation.info import parameter_fisher_multiframe, crb_identifiable
from phyguard.optics.forward import PhotonMicroscope
from phyguard.optics.psf import MicroscopyPSFConfig, make_psf

from experiments.run_biosr_identifiability_sweep import load_gt

MODES = ["defocus", "astig_x", "coma_x", "spherical"]
PROTOCOLS = {"1f": [0.0], "5f": [-1.0, -0.5, 0.0, 0.5, 1.0]}
THRESHOLD = 0.05


def piecewise_linear_knee(logx: np.ndarray, y: np.ndarray) -> float | None:

    order = np.argsort(logx)
    x, y = logx[order], y[order]
    best = None
    for i in range(2, len(x) - 2):
        x1, x2 = x[:i], x[i:]
        y1, y2 = y[:i], y[i:]
        a1 = np.polyfit(x1, y1, 1)
        a2 = np.polyfit(x2, y2, 1)
        sse = float(np.sum((y1 - np.polyval(a1, x1)) ** 2) + np.sum((y2 - np.polyval(a2, x2)) ** 2))
        if best is None or sse < best[0]:
            best = (sse, x[i])
    return None if best is None else float(best[1])


def crossing(logx: np.ndarray, y: np.ndarray, thresh: float) -> float | None:

    order = np.argsort(logx)
    x, y = logx[order], np.log10(np.maximum(y[order], 1e-9))
    lt = np.log10(thresh)
    for i in range(len(x) - 1):
        if (y[i] - lt) * (y[i + 1] - lt) <= 0 and y[i] != y[i + 1]:
            return float(x[i] + (lt - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    return None


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--structures", nargs="+", default=["ER", "MT"])
    ap.add_argument("--photons", type=float, nargs="+",
                    default=[3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 350, 600, 1000])
    ap.add_argument("--protocols", nargs="+", default=["1f", "5f"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--n-steps", type=int, default=500)
    ap.add_argument("--n-restarts", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    outdir = ROOT / "results" / "crb_cutoff"
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rows, fisher_rows = [], []

    for sname in args.structures:
        structure = torch.tensor(load_gt(sname, 1, 1))[None, None].to(device)
        rng = np.random.default_rng(7)
        true = {m: float(rng.uniform(-0.2, 0.2)) for m in MODES}
        print(f"[{sname}] true aberrations: {true}", flush=True)

        for pname in args.protocols:
            offsets = PROTOCOLS[pname]
            for photons in args.photons:
                bg = max(0.2, 0.15 * photons)

                coeffs = {m: torch.tensor(true[m], requires_grad=True) for m in MODES}
                F = parameter_fisher_multiframe(
                    coeffs, structure, defocus_offsets=offsets,
                    wavelength_nm=520.0, na=1.2, pixel_size_nm=62.6,
                    signal_photons=photons, background_photons=bg, read_noise_std=0.15,
                )
                crb, n_degen = crb_identifiable(F, MODES)
                for m in MODES:
                    fisher_rows.append({
                        "structure": sname, "protocol": pname, "photons": photons,
                        "mode": m, "sqrt_crb": float(np.sqrt(max(crb.get(m, np.nan), 0.0)))
                        if m in crb else float("nan"),
                        "identifiable": m in crb, "n_degenerate": n_degen,
                    })


                counts_all = []


                torch.manual_seed(1000 + int(photons * 10))
                for off in offsets:
                    th = dict(true); th["defocus"] = th["defocus"] + off
                    psf = make_psf(MicroscopyPSFConfig(
                        wavelength_nm=520.0, na=1.2, pixel_size_nm=62.6,
                        kernel_size=41, aberrations_waves=th,
                    )).to(device)
                    mic = PhotonMicroscope(psf=psf, signal_photons=photons,
                                           background_photons=bg, read_noise_std=0.15)
                    counts_all.append(mic.sample(structure)["counts"])
                counts = torch.cat(counts_all, dim=0)

                ests = {m: [] for m in MODES}
                for seed in args.seeds:
                    res = estimate_aberrations_multiframe(
                        counts, structure, modes=MODES, defocus_offsets=offsets,
                        wavelength_nm=520.0, na=1.2, pixel_size_nm=62.6,
                        signal_photons=photons, background_photons=bg,
                        read_noise_std=0.15, n_steps=args.n_steps,
                        n_restarts=args.n_restarts, seed=seed,
                    )
                    for m in MODES:
                        ests[m].append(res.estimated_coeffs[m])
                for m in MODES:
                    e = np.array(ests[m])
                    err = np.minimum((e - true[m]) ** 2, (e + true[m]) ** 2)
                    rows.append({
                        "structure": sname, "protocol": pname, "photons": photons,
                        "mode": m, "true": true[m],
                        "rmse": float(np.sqrt(err.mean())),
                        "rmse_sd": float(np.sqrt(err.std())),
                    })
                print(f"  {pname} photons={photons:g} done", flush=True)

    pd.DataFrame(rows).to_csv(outdir / "empirical.csv", index=False)
    pd.DataFrame(fisher_rows).to_csv(outdir / "fisher.csv", index=False)


    emp = pd.DataFrame(rows)
    fis = pd.DataFrame(fisher_rows)
    cut_rows = []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for sname in args.structures:
        fig, axes = plt.subplots(1, len(MODES), figsize=(4.2 * len(MODES), 3.6), sharex=True)
        for j, m in enumerate(MODES):
            ax = axes[j]
            for pname, color in [("1f", "tab:red"), ("5f", "tab:blue")]:
                e = emp.query("structure==@sname and protocol==@pname and mode==@m").sort_values("photons")
                f = fis.query("structure==@sname and protocol==@pname and mode==@m").sort_values("photons")
                lx = np.log10(e["photons"].values)
                ax.errorbar(e["photons"], e["rmse"], yerr=e["rmse_sd"], fmt="o-", ms=3,
                            lw=1, color=color, alpha=0.85, label=f"empirical {pname}")
                fc = f.dropna(subset=["sqrt_crb"])
                if len(fc):
                    ax.plot(fc["photons"], fc["sqrt_crb"], "--", color=color, lw=1.2,
                            alpha=0.7, label=f"sqrt(CRB) {pname}")
                knee = piecewise_linear_knee(lx, e["rmse"].values)
                cx_crb = crossing(np.log10(fc["photons"].values), fc["sqrt_crb"].values, THRESHOLD) if len(fc) else None
                cx_emp = crossing(lx, e["rmse"].values, THRESHOLD)
                if knee is not None:
                    ax.axvline(10 ** knee, color=color, ls=":", lw=1)
                cut_rows.append({
                    "structure": sname, "mode": m, "protocol": pname,
                    "empirical_knee_photons": None if knee is None else 10 ** knee,
                    "empirical_cross_0.05": None if cx_emp is None else 10 ** cx_emp,
                    "crb_cross_0.05": None if cx_crb is None else 10 ** cx_crb,
                })
            ax.axhline(THRESHOLD, color="k", ls="-", lw=0.6, alpha=0.5)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_title(m); ax.set_xlabel("signal photons")
            if j == 0:
                ax.set_ylabel("estimation RMSE (waves)")
                ax.legend(fontsize=6)
        fig.suptitle(f"{sname}: CRB-predicted vs empirical cutoff (threshold={THRESHOLD} waves)")
        fig.tight_layout()
        fig.savefig(outdir / f"cutoff_{sname}.png", dpi=160)
        plt.close(fig)

    cut = pd.DataFrame(cut_rows)
    cut.to_csv(outdir / "cutoffs.csv", index=False)
    print(cut.to_string(index=False))
    print("saved:", outdir)


if __name__ == "__main__":
    main()
