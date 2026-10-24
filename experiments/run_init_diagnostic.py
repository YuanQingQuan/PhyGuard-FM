from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from phyguard.estimation.calibration import parameterized_psf
from phyguard.optics.psf import MicroscopyPSFConfig, make_psf
from phyguard.optics.forward import PhotonMicroscope
from phyguard.likelihood import poisson_gaussian_nll
from experiments.run_biosr_identifiability_sweep import load_gt

MODES = ["defocus", "astig_x", "coma_x", "spherical"]
EVEN = {"defocus", "astig_x", "spherical"}
PROTOCOLS = {"1f": [0.0], "5f": [-1.0, -0.5, 0.0, 0.5, 1.0]}
WL, NA, PX, KS = 520.0, 1.2, 62.6, 41
N_STEPS = 500


def estimate(counts, structure, offsets, s, bg, *, init=None, n_restarts=4, seed=0):

    rng = np.random.default_rng(seed)
    dev = counts.device

    def _nll(theta):
        tot = 0.0
        for i, off in enumerate(offsets):
            th = dict(theta); th["defocus"] = th["defocus"] + float(off)
            psf = parameterized_psf(th, wavelength_nm=WL, na=NA, pixel_size_nm=PX, kernel_size=KS)
            mic = PhotonMicroscope(psf=psf, signal_photons=s, background_photons=bg,
                                   read_noise_std=0.15)
            tot = tot + poisson_gaussian_nll(counts[i], mic.expected_counts(structure), 0.15)
        return tot / len(offsets)

    best, finals = None, []
    for r in range(n_restarts):
        if init is not None and r == 0:
            start = {m: float(init[m]) for m in MODES}
        else:
            start = {m: float(rng.uniform(-0.25, 0.25)) for m in MODES}
        theta = {m: torch.tensor(start[m], requires_grad=True) for m in MODES}
        opt = torch.optim.Adam(list(theta.values()), lr=0.08)
        for _ in range(N_STEPS):
            opt.zero_grad()
            loss = _nll(theta)
            loss.backward()
            opt.step()
        finals.append(float(loss.detach()))
        if best is None or finals[-1] < best[0]:
            best = (finals[-1], {m: float(theta[m].detach()) for m in MODES})
    return best[1], finals


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--structure", default="MT")
    ap.add_argument("--totals", type=float, nargs="+", default=[60, 120, 250, 500, 1000])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--seeds-5f", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    outdir = ROOT / "results" / "init_diagnostic"
    outdir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    structure = torch.tensor(load_gt(args.structure, 1, 1))[None, None].to(dev)
    rng = np.random.default_rng(7)
    true = {m: float(rng.uniform(-0.2, 0.2)) for m in MODES}
    print("true:", true, flush=True)

    rows = []
    for pname, offsets in PROTOCOLS.items():
        k = len(offsets)
        seeds = args.seeds if pname == "1f" else args.seeds_5f
        for total in args.totals:
            s, bg = total / k, max(0.2, 0.15 * total / k)
            cs = []


            torch.manual_seed(1000 + int(total * 10))
            for off in offsets:
                th = dict(true); th["defocus"] = th["defocus"] + off
                psf = make_psf(MicroscopyPSFConfig(wavelength_nm=WL, na=NA, pixel_size_nm=PX,
                                                   kernel_size=KS, aberrations_waves=th)).to(dev)
                mic = PhotonMicroscope(psf=psf, signal_photons=s, background_photons=bg,
                                       read_noise_std=0.15)
                cs.append(mic.sample(structure)["counts"])
            counts = torch.cat(cs, dim=0)


            for seed in seeds:
                est, finals = estimate(counts, structure, offsets, s, bg, seed=seed)
                for m in MODES:
                    rows.append(dict(protocol=pname, total=total, init="random", seed=seed,
                                     mode=m, true=true[m], est=est[m],
                                     err_signed=(est[m] - true[m]) ** 2,
                                     err_signaware=(min((est[m] - true[m]) ** 2,
                                                        (est[m] + true[m]) ** 2) if m in EVEN
                                                    else (est[m] - true[m]) ** 2),
                                     nll_best=finals and min(finals), nll_spread=max(finals) - min(finals)))

            est, finals = estimate(counts, structure, offsets, s, bg, init=true, n_restarts=1)
            for m in MODES:
                rows.append(dict(protocol=pname, total=total, init="oracle", seed=0,
                                 mode=m, true=true[m], est=est[m],
                                 err_signed=(est[m] - true[m]) ** 2,
                                 err_signaware=(min((est[m] - true[m]) ** 2,
                                                    (est[m] + true[m]) ** 2) if m in EVEN
                                                else (est[m] - true[m]) ** 2),
                                 nll_best=finals[0], nll_spread=0.0))
            print(f"  {pname} total={total:g} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "per_run.csv", index=False)

    summ = (df.groupby(["protocol", "init", "mode", "total"])
              .agg(rmse_signed=("err_signed", lambda e: float(np.sqrt(np.mean(e)))),
                   rmse_signaware=("err_signaware", lambda e: float(np.sqrt(np.mean(e)))),
                   n=("seed", "count"))
              .reset_index())
    summ.to_csv(outdir / "summary.csv", index=False)
    print(summ.to_string(index=False))


    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.2))
    for j, m in enumerate(MODES):
        ax = axes[j]
        for pname, c in [("1f", "tab:red"), ("5f", "tab:blue")]:
            for init, mk, lb in [("random", "o-", "random restarts"), ("oracle", "s--", "oracle init")]:
                d = summ.query("protocol==@pname and init==@init and mode==@m").sort_values("total")
                ax.plot(d["total"], d["rmse_signed"], mk, ms=3.5, lw=1.0, color=c,
                        alpha=0.95 if init == "random" else 0.6,
                        label=f"{pname} {lb}")
        ax.axhline(0.05, color="k", lw=0.5, alpha=0.5)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("total signal photons")
        ax.set_title(m, fontsize=9)
        if j == 0:
            ax.set_ylabel("RMSE, signed (waves)")
            ax.legend(fontsize=5.5)
    fig.suptitle(f"RMSE fluctuation diagnostic ({args.structure}): random restarts vs oracle initialization",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "init_diagnostic.png", dpi=200)
    plt.close(fig)
    print("saved:", outdir)


if __name__ == "__main__":
    main()
