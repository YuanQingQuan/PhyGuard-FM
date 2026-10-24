from __future__ import annotations

from pathlib import Path
from typing import Dict
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
EVEN_MODES = {"defocus", "astig_x", "spherical"}
PROTOCOLS = {"1f": [0.0], "5f": [-1.0, -0.5, 0.0, 0.5, 1.0]}
THRESHOLD = 0.05
def crb_per_mode(F: np.ndarray) -> Dict[str, float]:

    inv = np.linalg.pinv(F, rcond=1e-12)
    return {m: float(np.diag(inv)[i]) for i, m in enumerate(MODES)}


def per_frame_budget(total: float, k: int) -> tuple[float, float]:

    s = total / k
    return s, max(0.2, 0.15 * s)


def mode_err(est: np.ndarray, true: float, mode: str) -> np.ndarray:
    if mode in EVEN_MODES:
        return np.minimum((est - true) ** 2, (est + true) ** 2)
    return (est - true) ** 2


def piecewise_linear_knee(logx: np.ndarray, y: np.ndarray) -> float | None:
    order = np.argsort(logx)
    x, y = logx[order], y[order]
    best = None
    for i in range(2, len(x) - 2):
        a1 = np.polyfit(x[:i], y[:i], 1)
        a2 = np.polyfit(x[i:], y[i:], 1)
        sse = float(np.sum((y[:i] - np.polyval(a1, x[:i])) ** 2) + np.sum((y[i:] - np.polyval(a2, x[i:])) ** 2))
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
    ap.add_argument("--total-photons", type=float, nargs="+",
                    default=[2, 4, 8, 15, 30, 60, 120, 250, 500, 1000])
    ap.add_argument("--theory-photons", type=float, nargs="+",
                    default=np.round(np.logspace(np.log10(0.5), np.log10(1500), 22), 3).tolist())
    ap.add_argument("--protocols", nargs="+", default=["1f", "5f"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--n-steps", type=int, default=500)
    ap.add_argument("--n-restarts", type=int, default=4)
    ap.add_argument("--theory-only", action="store_true")
    ap.add_argument("--emp-tag", default="")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    outdir = ROOT / "results" / "crb_cutoff_totalbudget"
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rows, fisher_rows, perseed_rows = [], [], []

    for sname in args.structures:
        structure = torch.tensor(load_gt(sname, 1, 1))[None, None].to(device)
        rng = np.random.default_rng(7)
        true = {m: float(rng.uniform(-0.2, 0.2)) for m in MODES}
        print(f"[{sname}] true aberrations: {true}", flush=True)

        for pname in args.protocols:
            offsets = PROTOCOLS[pname]
            k = len(offsets)

            for total in args.theory_photons:
                s_f, bg_f = per_frame_budget(total, k)
                coeffs = {m: torch.tensor(true[m]) for m in MODES}
                F = parameter_fisher_multiframe(
                    coeffs, structure, defocus_offsets=offsets,
                    wavelength_nm=520.0, na=1.2, pixel_size_nm=62.6,
                    signal_photons=s_f, background_photons=bg_f, read_noise_std=0.15,
                )
                crb = crb_per_mode(F)
                for m in MODES:
                    fisher_rows.append({
                        "structure": sname, "protocol": pname, "total_photons": total,
                        "photons_per_frame": s_f, "bg_per_frame": bg_f, "mode": m,
                        "sqrt_crb": float(np.sqrt(max(crb[m], 0.0))),
                        "identifiable": True, "n_degenerate": 0,
                    })
            if args.theory_only:
                continue

            for total in args.total_photons:
                s_f, bg_f = per_frame_budget(total, k)
                counts_all = []


                torch.manual_seed(1000 + int(total * 10))
                for off in offsets:
                    th = dict(true); th["defocus"] = th["defocus"] + off
                    psf = make_psf(MicroscopyPSFConfig(
                        wavelength_nm=520.0, na=1.2, pixel_size_nm=62.6,
                        kernel_size=41, aberrations_waves=th,
                    )).to(device)
                    mic = PhotonMicroscope(psf=psf, signal_photons=s_f,
                                           background_photons=bg_f, read_noise_std=0.15)
                    counts_all.append(mic.sample(structure)["counts"])
                counts = torch.cat(counts_all, dim=0)

                ests = {m: [] for m in MODES}
                for seed in args.seeds:
                    res = estimate_aberrations_multiframe(
                        counts, structure, modes=MODES, defocus_offsets=offsets,
                        wavelength_nm=520.0, na=1.2, pixel_size_nm=62.6,
                        signal_photons=s_f, background_photons=bg_f,
                        read_noise_std=0.15, n_steps=args.n_steps,
                        n_restarts=args.n_restarts, seed=seed,
                    )
                    for m in MODES:
                        ests[m].append(res.estimated_coeffs[m])
                        perseed_rows.append({
                            "structure": sname, "protocol": pname,
                            "total_photons": total, "photons_per_frame": s_f,
                            "mode": m, "seed": seed, "true": true[m],
                            "estimate": float(res.estimated_coeffs[m]),
                        })
                for m in MODES:
                    e = np.array(ests[m])
                    err = mode_err(e, true[m], m)
                    rows.append({
                        "structure": sname, "protocol": pname, "total_photons": total,
                        "photons_per_frame": s_f, "bg_per_frame": bg_f,
                        "mode": m, "true": true[m],
                        "rmse": float(np.sqrt(err.mean())),
                        "rmse_sd": float(np.sqrt(err.std())),
                        "n_seeds": len(args.seeds),
                    })
                print(f"  {pname} total={total:g} (s={s_f:.2f}/frame) done", flush=True)

    if args.theory_only:
        pd.DataFrame(fisher_rows).to_csv(outdir / "fisher_full.csv", index=False)
        print("saved:", outdir / "fisher_full.csv")
        return
    emp_name = f"empirical_{args.emp_tag}.csv" if args.emp_tag else "empirical.csv"
    pd.DataFrame(rows).to_csv(outdir / emp_name, index=False)
    if perseed_rows:
        ps_name = f"perseed_{args.emp_tag}.csv" if args.emp_tag else "perseed.csv"
        pd.DataFrame(perseed_rows).to_csv(outdir / ps_name, index=False)
    pd.DataFrame(fisher_rows).to_csv(outdir / "fisher.csv", index=False)
    if args.emp_tag:
        print("saved:", outdir / emp_name)
        return


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
                e = emp.query("structure==@sname and protocol==@pname and mode==@m").sort_values("total_photons")
                f = fis.query("structure==@sname and protocol==@pname and mode==@m").sort_values("total_photons")
                lx = np.log10(e["total_photons"].values)
                ax.errorbar(e["total_photons"], e["rmse"], yerr=e["rmse_sd"], fmt="o-", ms=3,
                            lw=1, color=color, alpha=0.85, label=f"empirical {pname}")
                fc = f.dropna(subset=["sqrt_crb"])
                if len(fc):
                    ax.plot(fc["total_photons"], fc["sqrt_crb"], "--", color=color, lw=1.2,
                            alpha=0.7, label=f"sqrt(CRB) {pname}")
                knee = piecewise_linear_knee(lx, e["rmse"].values)
                cx_crb = crossing(np.log10(fc["total_photons"].values), fc["sqrt_crb"].values, THRESHOLD) if len(fc) else None
                cx_emp = crossing(lx, e["rmse"].values, THRESHOLD)
                if knee is not None:
                    ax.axvline(10 ** knee, color=color, ls=":", lw=1)
                cut_rows.append({
                    "structure": sname, "mode": m, "protocol": pname,
                    "empirical_knee_total": None if knee is None else 10 ** knee,
                    "empirical_cross_0.05_total": None if cx_emp is None else 10 ** cx_emp,
                    "crb_cross_0.05_total": None if cx_crb is None else 10 ** cx_crb,
                })
            ax.axhline(THRESHOLD, color="k", ls="-", lw=0.6, alpha=0.5)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_title(m); ax.set_xlabel("total signal photons (all frames)")
            if j == 0:
                ax.set_ylabel("estimation RMSE (waves)")
                ax.legend(fontsize=6)
        fig.suptitle(f"{sname}: CRB vs empirical cutoff at FIXED TOTAL budget (threshold={THRESHOLD} waves)")
        fig.tight_layout()
        fig.savefig(outdir / f"cutoff_{sname}.png", dpi=160)
        plt.close(fig)

    cut = pd.DataFrame(cut_rows)
    cut.to_csv(outdir / "cutoffs.csv", index=False)
    print(cut.to_string(index=False))
    print("saved:", outdir)


if __name__ == "__main__":
    main()
