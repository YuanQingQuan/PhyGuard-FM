from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from phyguard.optics.psf import MicroscopyPSFConfig, make_psf, psf_to_otf
from phyguard.optics.psf3d import make_psf_3d
from phyguard.optics.forward import PhotonMicroscope
from phyguard.likelihood import poisson_gaussian_nll
from phyguard.estimation.info import parameter_fisher, parameter_fisher_multiframe, crb_identifiable
from phyguard.estimation.protocol import expected_info_gain, candidate_offsets
from experiments.run_biosr_identifiability_sweep import load_gt
from experiments.tighten_figure_rows import tighten as tighten_rows

OUT = ROOT / "results" / "paper_figures" / "v6"
OUT.mkdir(parents=True, exist_ok=True)
_ASSETS_PATH = ROOT / "results" / "paper_figures" / "v2" / "assets.npz"
if not _ASSETS_PATH.exists():
    raise SystemExit(
        f"missing figure-asset cache: {_ASSETS_PATH}\n"
        "Required cache: results/paper_figures/v2/assets.npz")
ASSETS = dict(np.load(_ASSETS_PATH))

plt.rcParams.update({

    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "stixsans",
    "font.size": 8.0, "axes.linewidth": 0.6, "figure.dpi": 120,
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "axes.grid": False,
    "axes.edgecolor": "0.15", "axes.labelcolor": "0.1",
    "xtick.color": "0.15", "ytick.color": "0.15",
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 2.4, "ytick.major.size": 2.4,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
    "axes.labelsize": 8.0, "axes.titlesize": 8.0,
    "legend.framealpha": 1.0, "legend.edgecolor": "none",
    "legend.borderpad": 0.35, "legend.labelspacing": 0.32,
    "legend.handlelength": 1.5, "legend.handletextpad": 0.5,
})
DPI = 400

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
WL, NA, PX, KS = 520.0, 1.2, 62.6, 41
MODES = ["defocus", "astig_x", "coma_x", "spherical"]

PAL = {"red": "#D55E00", "blue": "#0072B2", "green": "#009E73",
       "purple": "#CC79A7", "orange": "#E69F00", "sky": "#56B4E9",
       "gray": "#7F7F7F", "dark": "#4D4D4D"}
MCOL = dict(zip(MODES, [PAL["red"], PAL["blue"], PAL["green"], PAL["purple"]]))


def A(key):
    return ASSETS[key]


def cbar_axes(ax, size=0.04, pad=0.035, height=0.88):

    return ax.inset_axes([1.0 + pad, 0.0, size, height], transform=ax.transAxes)


def lab(ax, letter, text, sub=None):

    base = 2.5
    LINE = 9.6
    lines = text.split("\n") + (sub.split("\n") if sub else [])
    top = base + (len(lines) - 1) * LINE
    dx = 16.5 if letter else 0.0
    for i, s in enumerate(lines):
        dy = top - i * LINE
        if letter and i == 0:
            ax.text(0, 1.0, f"({letter})",
                    transform=matplotlib.transforms.offset_copy(
                        ax.transAxes, fig=ax.figure, x=0.0, y=dy, units="points"),
                    ha="left", va="bottom", fontsize=9.0, fontweight="bold",
                    color="0.1", clip_on=False)
        ax.text(0, 1.0, s,
                transform=matplotlib.transforms.offset_copy(
                    ax.transAxes, fig=ax.figure, x=dx, y=dy, units="points"),
                ha="left", va="bottom", fontsize=8.0, color="0.1", clip_on=False)


def show(ax, img, gamma=1.0, cmap="inferno", cbar=True, clabel="norm. intensity",
         px_nm=None, bar_nm=None, vmin=None, vmax=None, clabelpad=4.0):
    im = ax.imshow(np.clip(img, 0, None) ** gamma, cmap=cmap, interpolation="nearest",
                   vmin=vmin, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    if cbar:
        cax = cbar_axes(ax)
        cb = ax.figure.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=8.0, length=1.5)


        cb.set_label(clabel, fontsize=8.0, labelpad=clabelpad)
        cb.outline.set_linewidth(0.4)
    if px_nm and bar_nm:
        n = bar_nm / px_nm
        h, w = img.shape[-2:]


        y0, x1 = h * 0.94, w * 0.87
        ax.plot([x1 - n, x1], [y0, y0], color="w", lw=2.2, solid_capstyle="butt")
        ax.text(x1 - n / 2, y0 - h * 0.03, f"{bar_nm/1000:g} µm" if bar_nm >= 1000 else f"{bar_nm:g} nm",
                color="w", fontsize=8.0, ha="center", va="bottom")
    return im


def savefig(fig, name):
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
    plt.close(fig)


    tighten_rows(OUT / name, DPI)
    print(name, flush=True)


def nll_display(surf):
    span = surf.max() - surf.min()
    return np.log10(surf - surf.min() + 1e-3 * span)


def psf_img(coeffs):
    return make_psf(MicroscopyPSFConfig(wavelength_nm=WL, na=NA, pixel_size_nm=PX,
                                        kernel_size=KS, aberrations_waves=coeffs)).numpy()


def fig01():
    rec = pd.read_csv(ROOT / "results/aberration_calibration/reconstruction.csv")
    fig = plt.figure(figsize=(6.5, 3.25))


    gs = fig.add_gridspec(2, 5, hspace=0.7, wspace=0.55)
    row1 = [("ER_gt", "a", "GT (ER)"), ("ER_obs", "b", "aberrated obs"),
            ("ER_rl_nom", "c", "RL nominal"), ("ER_rl_cal", "d", "RL calibrated"),
            ("ER_rl_true", "e", "RL oracle")]
    for k, (key, lt, t) in enumerate(row1):
        ax = fig.add_subplot(gs[0, k])
        last = k == len(row1) - 1
        show(ax, A(key), gamma=0.7, px_nm=PX, bar_nm=2000, cbar=last,
             clabel="norm. intensity" if last else "")
        lab(ax, lt, t, sub="0.09 w RMS" if k == 1 else None)
    row2 = [("MT_gt", "f", "GT (MT)"), ("MT_obs", "g", "aberrated obs"),
            ("MT_rl_nom", "h", "RL nominal"), ("MT_rl_cal", "i", "RL calibrated")]
    for k, (key, lt, t) in enumerate(row2):
        ax = fig.add_subplot(gs[1, k])


        show(ax, A(key), gamma=0.7, px_nm=PX, bar_nm=2000, cbar=False)
        lab(ax, lt, t, sub="0.054 w RMS" if k == 1 else None)
    axj = fig.add_subplot(gs[1, 4])
    labels = ["obs", "nominal", "calibrated", "oracle"]
    colors = ["0.5", "#D55E00", "#0072B2", "#009E73"]
    for k, (_, r) in enumerate(rec.iterrows()):
        vals = [r["obs_psnr"], r["rl_nominal_psnr"], r["rl_calibrated_psnr"], r["rl_true_psnr"]]
        axj.bar(np.arange(4) + k * 5.0, vals, 1.0, color=colors)
    axj.set_xticks([1.5, 6.5]); axj.set_xticklabels(["5", "50"], fontsize=8.0)
    axj.set_xlabel("signal scale $s$")
    axj.set_ylim(0, max(rec["rl_true_psnr"].max() * 1.5, 13))
    axj.set_ylabel("PSNR (dB)", labelpad=0.5)
    handles = [Patch(facecolor=c, label=l) for l, c in zip(labels, colors)]


    axj.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.38),
               ncols=2, fontsize=8.0, columnspacing=0.8, handlelength=1.1)
    lab(axj, "j", "calibrated ≈ oracle", sub="BioSR ER, U(±0.3 w) draw")
    savefig(fig, "Fig01.png")


def fig02():
    gt = torch.tensor(A("ER_gt"))[None, None].to(DEV)
    fig = plt.figure(figsize=(6.5, 3.53))
    gs = fig.add_gridspec(2, 4, hspace=0.7, wspace=0.8)
    obs_by_d = {}
    for k, d in enumerate([-0.2, 0.0, 0.2]):
        psf = make_psf(MicroscopyPSFConfig(wavelength_nm=WL, na=NA, pixel_size_nm=PX, kernel_size=KS,
                                           aberrations_waves={"defocus": d})).to(DEV)
        mic = PhotonMicroscope(psf=psf, signal_photons=30.0, background_photons=4.5, read_noise_std=0.15)
        torch.manual_seed(9)
        ob = mic.sample(gt)["counts"][0, 0].cpu().numpy()
        obs_by_d[d] = ob
        ax = fig.add_subplot(gs[0, k])
        show(ax, ob, gamma=0.7, clabel="counts" if k == 0 else "", px_nm=PX, bar_nm=2000)
        lab(ax, chr(97 + k), f"obs, defocus {d:+.1f} w", sub="ER, 5.4 ph/px" if k == 0 else None)
    axg = fig.add_subplot(gs[0, 3])
    for d, c, ls in [(-0.2, "#D55E00", "-"), (0.0, "k", "--"), (0.2, "#0072B2", "-.")]:
        psf = psf_img({"defocus": d})
        axg.plot((np.arange(41) - 20) * PX / 1000, psf[20, :] / psf.max(), lw=1.0, color=c, ls=ls,
                 label=f"{d:+.1f} w")
    axg.set_xlabel("offset from center (µm)"); axg.set_ylabel("PSF (norm.)")


    axg.set_ylim(0, 2.4)


    axg.legend(fontsize=8.0, loc="upper left", ncol=2, title="defocus",
               handlelength=1.2, handletextpad=0.4, title_fontsize=8.0,
               borderpad=0.3, labelspacing=0.3)
    lab(axg, "d", "kernel line cuts")
    axe = fig.add_subplot(gs[1, 0:2])
    surf = A("nllsurf_1f")
    im = axe.imshow(nll_display(surf), cmap="viridis", extent=[-0.3, 0.3, -0.3, 0.3],
                    origin="lower", aspect="auto")
    axe.set_xlabel("astig_x (waves)"); axe.set_ylabel("astig_y (waves)")
    cax = cbar_axes(axe, size=0.03, pad=0.05)
    cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=8.0, length=1.5)
    cb.set_label("log$_{10}$ NLL (rel.)", fontsize=8.0)
    lab(axe, "e", "NLL landscape is multimodal (log display)",
        sub="single frame; true (a$_x$, a$_y$) = (0.15, −0.10) w; s=30")
    axf = fig.add_subplot(gs[1, 2])
    rng = np.random.default_rng(0)
    axf.imshow(nll_display(surf), cmap="viridis", extent=[-0.3, 0.3, -0.3, 0.3],
               origin="lower", aspect="auto")
    pts = rng.uniform(-0.25, 0.25, (6, 2))
    axf.plot(pts[:, 0], pts[:, 1], "wx", ms=5, mew=1.2)
    axf.plot(0.15, -0.10, "r+", ms=9, mew=1.6)
    axf.set_xticks([]); axf.set_yticks([])
    lab(axf, "f", "random restarts", sub="same surface as (e)")
    axh = fig.add_subplot(gs[1, 3])
    for d, c, ls in [(-0.2, "#D55E00", "-"), (0.0, "k", "--"), (0.2, "#0072B2", "-.")]:
        ob = obs_by_d[d]
        prof = ob[128, :] / ob.max()
        axh.plot((np.arange(256) - 128) * PX / 1000, prof, lw=0.9, color=c, ls=ls,
                 label=f"{d:+.1f} w")
    axh.set_xlabel("offset from field center (µm)"); axh.set_ylabel("counts (norm.)")
    axh.set_ylim(0, 1.45)

    axh.legend(fontsize=8.0, loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=2,
               title="defocus", handlelength=1.2, handletextpad=0.4, title_fontsize=8.0,
               frameon=False, borderpad=0.3, labelspacing=0.3)
    lab(axh, "g", "line cut through the observation")
    savefig(fig, "Fig02.png")


def fig03():
    gt = torch.tensor(load_gt("ER", 1, 1)[64:192, 64:192].copy())[None, None].to(DEV)
    photons, bg = 30.0, 4.5
    true_a = 0.20

    def counts_for(a, off, seed):
        psf = make_psf(MicroscopyPSFConfig(wavelength_nm=WL, na=NA, pixel_size_nm=PX, kernel_size=KS,
                                           aberrations_waves={"astig_x": a, "defocus": off})).to(DEV)
        mic = PhotonMicroscope(psf=psf, signal_photons=photons, background_photons=bg, read_noise_std=0.15)
        torch.manual_seed(seed)
        return mic.sample(gt)["counts"]

    obs1 = [counts_for(true_a, 0.0, 21)]
    obs3 = [obs1[0], counts_for(true_a, 0.5, 22), counts_for(true_a, -0.5, 23)]

    def nll_scan(obs_list, offs):
        vals = []
        for a in np.linspace(-0.35, 0.35, 41):
            tot = 0.0
            for off, ob in zip(offs, obs_list):
                psf = make_psf(MicroscopyPSFConfig(wavelength_nm=WL, na=NA, pixel_size_nm=PX, kernel_size=KS,
                                                   aberrations_waves={"astig_x": float(a), "defocus": off})).to(DEV)
                mic = PhotonMicroscope(psf=psf, signal_photons=photons, background_photons=bg, read_noise_std=0.15)
                tot += float(poisson_gaussian_nll(ob, mic.expected_counts(gt), 0.15))
            vals.append(tot)
        return np.array(vals)

    scan = np.linspace(-0.35, 0.35, 41)
    nll1, nll3 = nll_scan(obs1, [0.0]), nll_scan(obs3, [0.0, 0.5, -0.5])
    gt256 = torch.tensor(load_gt("ER", 1, 1))[None, None].to(DEV)
    base = {"defocus": -0.11, "astig_x": 0.06, "coma_x": 0.17, "spherical": 0.19}
    scales = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003]
    law1, law5, law1c = [], [], []
    icoma = MODES.index("coma_x")
    for s in scales:
        th = {m: base[m] * s for m in MODES}
        coeffs = {m: torch.tensor(th[m]) for m in MODES}
        F1 = parameter_fisher(coeffs, gt256, wavelength_nm=WL, na=NA, pixel_size_nm=PX,
                              signal_photons=50.0, background_photons=7.5, read_noise_std=0.15)
        F5 = parameter_fisher_multiframe(coeffs, gt256, defocus_offsets=[-1.0, -0.5, 0.0, 0.5, 1.0],
                                         wavelength_nm=WL, na=NA, pixel_size_nm=PX,
                                         signal_photons=50.0, background_photons=7.5, read_noise_std=0.15)
        law1.append(np.linalg.eigvalsh(F1)[0]); law5.append(np.linalg.eigvalsh(F5)[0])
        law1c.append(F1[icoma, icoma])

    fig = plt.figure(figsize=(6.5, 5.76))
    gs = fig.add_gridspec(3, 4, hspace=1.0, wspace=0.98)
    for k, sign in enumerate([+1, -1]):
        ax = fig.add_subplot(gs[0, k])
        show(ax, psf_img({"astig_x": 0.2 * sign}) * 100.0, px_nm=PX, bar_nm=500,
             clabel="norm. (1e−2)" if k == 0 else "")
        lab(ax, chr(97 + k), f"2D PSF {'+' if sign > 0 else '−'}0.2 w", sub="astig" if k == 0 else None)
    for k, sign in enumerate([+1, -1]):
        cfg = MicroscopyPSFConfig(wavelength_nm=WL, na=NA, pixel_size_nm=PX, kernel_size=KS,
                                  aberrations_waves={"astig_x": 0.2 * sign})
        xz = make_psf_3d(cfg, nz=9, dz_nm=250.0).numpy()[:, 20, :]
        ax = fig.add_subplot(gs[0, 2 + k])
        im = ax.imshow(np.clip(xz, 0, None) ** 0.4, cmap="inferno", interpolation="nearest",
                       extent=[(-20.5) * PX / 1000, 20.5 * PX / 1000, 4 * 0.25, -4 * 0.25], aspect="auto")
        ax.set_xlabel("x (µm)", fontsize=8.0)
        if k == 0:
            ax.set_ylabel("z (µm)", fontsize=8.0)
        ax.tick_params(labelsize=8.0)
        cax = cbar_axes(ax)
        cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=8.0, length=1.5)
        lab(ax, chr(99 + k), f"3D xz {'+' if sign > 0 else '−'}0.2 w", sub="astig" if k == 0 else None)
    p2dp, p2dm = psf_img({"astig_x": 0.2}), psf_img({"astig_x": -0.2})
    cfgp = MicroscopyPSFConfig(wavelength_nm=WL, na=NA, pixel_size_nm=PX, kernel_size=KS,
                               aberrations_waves={"astig_x": 0.2})
    cfgm = MicroscopyPSFConfig(wavelength_nm=WL, na=NA, pixel_size_nm=PX, kernel_size=KS,
                               aberrations_waves={"astig_x": -0.2})
    r2d = np.abs(p2dp - p2dm)
    r3d = np.abs(make_psf_3d(cfgp, nz=9, dz_nm=250.0).numpy()[:, 20, :]
                 - make_psf_3d(cfgm, nz=9, dz_nm=250.0).numpy()[:, 20, :])
    ax = fig.add_subplot(gs[1, 0])


    show(ax, r2d * 1e10, cmap="viridis", clabel="|Δ PSF| (1e−10)", px_nm=PX, bar_nm=500)
    lab(ax, "e", "|PSF(+θ)−PSF(−θ)|", sub="2D, ≈ 0 (num.)")
    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(r3d ** 0.4, cmap="viridis", interpolation="nearest",
                   extent=[(-20.5) * PX / 1000, 20.5 * PX / 1000, 4 * 0.25, -4 * 0.25], aspect="auto")
    ax.tick_params(labelsize=8.0); ax.set_xlabel("x (µm)", fontsize=8.0)
    cax = cbar_axes(ax)
    cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=8.0, length=1.5)
    lab(ax, "f", "|PSF(+θ)−PSF(−θ)|", sub=f"3D, peak {r3d.max():.0e}")
    axh = fig.add_subplot(gs[1, 2])
    axh.plot(scan, (nll1 - nll1.min()) / (nll1.max() - nll1.min()), color="#D55E00", lw=1.1,
             label="1 frame")
    axh.plot(scan, (nll3 - nll3.min()) / (nll3.max() - nll3.min()), color="#0072B2", lw=1.1,
             ls="--", label="3 frames")
    axh.axvline(true_a, color="k", lw=0.5, ls=":")
    axh.set_xlabel("astig_x (waves)"); axh.set_ylabel("NLL (norm.)")


    axh.set_xticks([-0.2, 0.0, 0.2])
    axh.set_xticklabels(["−0.2", "0", "+0.2"])
    axh.set_ylim(0, 1.6)


    lab(axh, "g", "double → single well",
        sub="solid 1f; dashed 3f\ntrue astig_x +0.2 w")
    axf2 = fig.add_subplot(gs[1, 3])
    axf2.plot(scales, law1, "o-", ms=3.5, lw=1.1, color="#D55E00", label="even, 1f")
    axf2.plot(scales, law1c, "^-", ms=3.5, lw=1.1, color="#009E73", label="odd, 1f")
    axf2.plot(scales, law5, "s-", ms=3.5, lw=1.1, color="#0072B2", label="even, 5f")
    xs = np.array(scales)
    axf2.plot(xs, law1[0] * xs**2, ":", color="#D55E00", lw=0.8, alpha=0.6, label="$\\propto$ |$\\theta$|²")
    axf2.set_xscale("log"); axf2.set_yscale("log")
    axf2.set_xlabel("|θ| scale (× base mix)")
    axf2.set_ylabel("Fisher information")


    axf2.legend(fontsize=8.0, loc="upper center", bbox_to_anchor=(0.5, -0.30),
                ncol=2, frameon=False, handlelength=1.3, labelspacing=0.25)
    lab(axf2, "h", "parity-resolved Fisher law", sub="ER, 9 ph/px")
    for k, (tag, letter, est_name) in enumerate([("1f", "i", "single frame"), ("3f", "j", "3-frame diversity")]):
        ax = fig.add_subplot(gs[2, k])
        surf = A(f"nllsurf_{tag}")
        im = ax.imshow(nll_display(surf), cmap="viridis", extent=[-0.3, 0.3, -0.3, 0.3],
                       origin="lower", aspect="auto")
        ax.plot(0.15, -0.10, "w+", ms=8, mew=1.4)
        amin = np.unravel_index(np.argmin(surf), surf.shape)
        ax.plot(np.linspace(-0.3, 0.3, 17)[amin[1]], np.linspace(-0.3, 0.3, 17)[amin[0]],
                "x", ms=5.5, mew=1.3, color="#D55E00")
        ax.set_xlabel("astig_x (waves)", fontsize=8.0)
        if k == 0:
            ax.set_ylabel("astig_y (waves)", fontsize=8.0)
        ax.tick_params(labelsize=8.0)
        cax = cbar_axes(ax)
        cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=8.0, length=1.5)
        lab(ax, letter, f"NLL, {est_name}", sub="s=30, astig x/y" if k == 0 else "s=30")
    axk = fig.add_subplot(gs[2, 2])
    for off, c, ls in [(0.0, "#D55E00", "-"), (0.5, "#0072B2", "--"), (-0.5, "#009E73", "-.")]:
        psf = psf_img({"astig_x": true_a, "defocus": off})
        axk.plot((np.arange(41) - 20) * PX / 1000, psf[20, :] / psf.max(), lw=1.0, color=c, ls=ls,
                 label=f"{off:+.1f} w")
    axk.set_xlabel("offset from center (µm)"); axk.set_ylabel("PSF (norm.)")
    axk.set_ylim(0, 2.5)

    axk.legend(fontsize=8.0, loc="upper left", ncol=2, title="defocus",
               handlelength=1.2, handletextpad=0.4, title_fontsize=8.0,
               borderpad=0.3, labelspacing=0.3)
    lab(axk, "k", "kernel wings vs offset", sub="astig 0.2 w + defocus")
    cp, cm = psf_img({"coma_x": 0.2}), psf_img({"coma_x": -0.2})
    direct = np.abs(cp - cm).max()
    axl = fig.add_subplot(gs[2, 3])
    show(axl, np.abs(cp - cm), cmap="viridis", clabel="|Δ PSF|", px_nm=PX, bar_nm=500)
    lab(axl, "l", "coma: h(−θ) ≠ h(+θ)", sub="mirror-exact")
    savefig(fig, "Fig03.png")


def fig04():
    gt = torch.tensor(load_gt("ER", 1, 1))[None, None].to(DEV)
    true = {"defocus": -0.11, "astig_x": 0.06, "coma_x": 0.17, "spherical": 0.19}
    coeffs = {m: torch.tensor(true[m]) for m in MODES}
    F = parameter_fisher(coeffs, gt, wavelength_nm=WL, na=NA, pixel_size_nm=PX,
                         signal_photons=50.0, background_photons=7.5, read_noise_std=0.15)
    crb, _ = crb_identifiable(F, MODES)
    budget = pd.read_csv(ROOT / "results/information_budget/budget.csv")
    fig = plt.figure(figsize=(6.5, 4.64))
    gs = fig.add_gridspec(2, 3, hspace=0.9, wspace=0.8)
    ax = fig.add_subplot(gs[0, 0])
    eigs = np.linalg.eigvalsh(F)[::-1]
    ax.bar(range(4), eigs, 0.55, color="#0072B2")
    ax.set_yscale("log"); ax.set_xticks(range(4)); ax.set_xticklabels(["d1", "d2", "d3", "d4"])
    ax.set_ylim(bottom=1e2)
    ax.set_ylabel("Fisher eigenvalue")
    lab(ax, "a", "parameter FIM spectrum", sub="single frame, 9 ph/px, ER;\nθ = (−0.11, 0.06, 0.17, 0.19) w")
    ax = fig.add_subplot(gs[0, 1])
    vals = [np.sqrt(crb[m]) for m in MODES]
    ax.bar(range(4), vals, 0.55, color=[MCOL[m] for m in MODES])
    ax.set_xticks(range(4)); ax.set_xticklabels(MODES, fontsize=8.0, rotation=30, ha="right")
    ax.set_ylabel("√CRB (waves)")
    lab(ax, "b", "per-mode precision limit", sub="same operating point as (a)")
    ax = fig.add_subplot(gs[0, 2])
    for mag, c, ls in [(0.0, "k", "-"), (0.1, "#0072B2", "--"), (0.2, "#D55E00", "-.")]:
        cfg = MicroscopyPSFConfig(wavelength_nm=WL, na=NA, pixel_size_nm=PX, kernel_size=KS,
                                  aberrations_waves={"coma_x": mag} if mag else {})
        otf = psf_to_otf(make_psf(cfg), (256, 256)).abs().numpy() ** 2
        prof = np.fft.fftshift(otf / otf.max())
        h, w = prof.shape
        y, x = np.indices(prof.shape)
        r = np.sqrt((y - h // 2) ** 2 + (x - w // 2) ** 2).astype(int)
        prof = np.bincount(r.ravel(), prof.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
        ax.plot(np.arange(len(prof)) / 128.0, prof, lw=1.1, color=c, ls=ls, label=f"coma {mag:.1f} w")
    ax.axhline(0.05, color="r", lw=0.5, ls="--"); ax.set_xlim(0, 0.8); ax.set_ylim(0, 1.02)
    ax.set_xlabel("|q| (rel. to cutoff)")
    ax.set_ylabel("|OTF|² (norm.)")
    ax.legend(fontsize=8.0, loc="upper right")
    lab(ax, "c", "content side: |OTF|²", sub="coma 0/0.1/0.2 w, NA 1.2")
    ax = fig.add_subplot(gs[1, 0])
    for s, c, mk, ls, xb in [("ER", "#D55E00", "o", "-", 0.18), ("MT", "#0072B2", "s", "--", 0.45)]:
        d = budget[budget.structure == s].groupby("photons_per_pixel").agg(
            cf=("content_fisher_diag_mean", "mean")).reset_index()
        ax.plot(d["photons_per_pixel"] * xb, d["cf"], marker=mk, ls=ls, ms=3.5, lw=1.1, color=c, label=s)
    ax.set_xscale("log"); ax.set_yscale("log")


    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterMathtext(base=10,
                                                                       labelOnlyBase=True))
    ax.set_xlabel("mean signal photons/pixel"); ax.set_ylabel("mean content Fisher")
    ax.legend(fontsize=8.0)
    lab(ax, "d", "content information $\\propto$ photons", sub="θ = 0.1 w/mode, single frame")
    ax = fig.add_subplot(gs[1, 1])
    th0 = {m: torch.tensor(0.1) for m in MODES}
    for s, c, mk, ls, xb in [("ER", "#D55E00", "o", "-", 0.18), ("MT", "#0072B2", "s", "--", 0.45)]:
        gts = torch.tensor(load_gt(s, 1, 1))[None, None].to(DEV)
        phs = np.array([1, 2, 5, 10, 20, 50], dtype=float)
        prec = []
        for ph in phs:
            Fp = parameter_fisher(th0, gts, wavelength_nm=WL, na=NA, pixel_size_nm=PX,
                                  signal_photons=float(ph), background_photons=max(0.2, 0.15 * ph),
                                  read_noise_std=0.15)
            eps = 1.0 / 0.5**2
            prec.append(float(np.sqrt(np.trace(np.linalg.inv(Fp + eps * np.eye(len(MODES)))))))
        prec = np.array(prec)
        ax.plot(phs * xb, prec, marker=mk, ls=ls, ms=3.5, lw=1.1, color=c, label=s)
        ax.plot(phs * xb, prec[0] * (phs[0] / phs) ** 0.5, ":", lw=0.8, color=c, alpha=0.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("mean signal photons/pixel"); ax.set_ylabel("√tr (F+prior)$^{-1}$ (waves)")
    ax.legend(fontsize=8.0, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2,
              frameon=False)
    lab(ax, "e", "parameter precision $\\propto$ 1/$\\sqrt{N}$", sub="θ = 0.1 w/mode, single frame")
    savefig(fig, "Fig04.png")


def fig05():
    gt = torch.tensor(load_gt("ER", 1, 1))[None, None].to(DEV)
    base = {"defocus": 0.0, "astig_x": 0.06, "coma_x": 0.17, "spherical": 0.19}
    offsets = [-1.0, -0.5, 0.0, 0.5, 1.0]
    tabs = []
    for ph in [5.0, 20.0, 50.0]:
        tab = np.full((len(MODES), len(offsets)), np.nan)
        for j, off in enumerate(offsets):
            th = dict(base); th["defocus"] += off
            coeffs = {m: torch.tensor(th[m]) for m in MODES}
            F = parameter_fisher(coeffs, gt, wavelength_nm=WL, na=NA, pixel_size_nm=PX,
                                 signal_photons=ph, background_photons=max(0.2, 0.15 * ph), read_noise_std=0.15)
            crb, _ = crb_identifiable(F, MODES)
            for i, m in enumerate(MODES):
                tab[i, j] = np.sqrt(crb.get(m, np.nan))
        tabs.append(tab)


    allv = np.concatenate([np.log10(t.ravel() * 1000.0) for t in tabs])
    vmin, vmax = np.nanmin(allv), np.nanmax(allv)

    fig = plt.figure(figsize=(6.5, 4.64))
    gs = fig.add_gridspec(2, 4, hspace=1.0, wspace=0.55, height_ratios=[1, 0.9],
                          width_ratios=[1, 1, 1, 0.08])
    phs = [5.0, 20.0, 50.0]
    for k, (ph, tab) in enumerate(zip(phs, tabs)):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(np.log10(tab * 1000.0), cmap="viridis_r", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(5)); ax.set_xticklabels(["−1", "−0.5", "0", "+0.5", "+1"], fontsize=8.0)


        ax.set_yticks(range(4))
        ax.set_yticklabels(MODES if k == 0 else [], fontsize=8.0)
        ax.set_xlabel("defocus offset (waves)", fontsize=8.0)
        lab(ax, chr(97 + k), f"√CRB, {ph*0.18:.1f} ph/px",
            sub="$\\theta_0$: astig 0.06, coma 0.17,\nsph 0.19 w" if k == 0 else None)
        if k:
            continue
        for i in range(4):
            for j in range(5):
                if not np.isnan(tab[i, j]):
                    v = (np.log10(tab[i, j] * 1000.0) - vmin) / (vmax - vmin)


                    r, g, b, _ = matplotlib.colormaps["viridis_r"](float(np.clip(v, 0.0, 1.0)))
                    lum = 0.299 * r + 0.587 * g + 0.114 * b
                    ax.text(j, i, f"{tab[i,j]*1000.0:.2g}", ha="center", va="center", fontsize=8.0,
                            color="k" if lum > 0.5 else "w")
    cax = fig.add_subplot(gs[0, 3])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax), cmap="viridis_r"),
        cax=cax)
    cb.set_label("log$_{10}$ √CRB (milli-waves)", fontsize=8.0)
    cb.ax.tick_params(labelsize=8.0)
    axd = fig.add_subplot(gs[1, 1:3])
    for ph, c, mk, ls in [(5.0, "#D55E00", "o", "-"), (20.0, "#0072B2", "s", "--"), (50.0, "#009E73", "^", "-.")]:
        eigs = []
        for off in offsets:
            th = dict(base); th["defocus"] += off
            coeffs = {m: torch.tensor(th[m]) for m in MODES}
            F = parameter_fisher(coeffs, gt, wavelength_nm=WL, na=NA, pixel_size_nm=PX,
                                 signal_photons=ph, background_photons=max(0.2, 0.15 * ph), read_noise_std=0.15)
            eigs.append(np.linalg.eigvalsh(F)[0])
        axd.plot(offsets, np.array(eigs) / ph, marker=mk, ls=ls, ms=3.5, lw=1.1, color=c,
                 label=f"{ph*0.18:.1f} ph/px")
    axd.set_yscale("log"); axd.legend(fontsize=8.0)
    axd.set_xlabel("defocus offset (waves)")
    axd.set_ylabel("min FIM eigenvalue / photon")
    lab(axd, "d", "worst-direction information per photon",
        sub="normalized curves overlap: F $\\propto$ photons")
    savefig(fig, "Fig05.png")


def fig06():


    emp = pd.read_csv(ROOT / "results/crb_cutoff_totalbudget/empirical_all_block.csv")
    emp = emp.rename(columns={"rmse_block": "rmse"})
    fis = pd.read_csv(ROOT / "results/crb_cutoff_totalbudget/fisher_full.csv")
    cut = pd.read_csv(ROOT / "results/crb_cutoff_totalbudget/crossings_bootstrap_block.csv")
    cut = cut.rename(columns={"cross_median": "emp_cross"})
    _th = pd.read_csv(ROOT / "results/paper_figures/photons_needed_total.csv")
    cut = cut.merge(_th.rename(columns={"photons_needed": "crb_cross"}),
                    on=["structure", "protocol", "mode"], how="left")
    diag_path = ROOT / "results" / "init_diagnostic" / "summary.csv"
    have_diag = diag_path.exists()

    XBAR = {"ER": 0.18, "MT": 0.45}
    fig = plt.figure(figsize=(6.5, 5.76))
    gs = fig.add_gridspec(2, 4, hspace=0.75, wspace=0.6, height_ratios=[1, 1.1])
    for j, m in enumerate(MODES):
        ax = fig.add_subplot(gs[0, j])
        for p, c, ls in [("1f", "#D55E00", "-"), ("5f", "#0072B2", "--")]:
            e = emp.query("structure=='MT' and protocol==@p and mode==@m").sort_values("total_photons")
            f = fis.query("structure=='MT' and protocol==@p and mode==@m and identifiable").sort_values("total_photons")
            ax.plot(e["total_photons"] * XBAR["MT"], e["rmse"], "o", ms=2.8, ls=ls, lw=0.9, color=c, label=f"emp {p}")
            ax.plot(f["total_photons"] * XBAR["MT"], f["sqrt_crb"], ls=":", lw=1.0, color=c, alpha=0.75, label=f"√CRB {p}")
        ax.axhline(0.05, color="k", lw=0.5, alpha=0.4)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("total signal ph/px")
        lab(ax, chr(97 + j), f"{m} (MT)", sub="θ draw ±0.2 w, 6 seeds" if j == 0 else None)
        if j == 0:
            ax.set_ylabel("RMSE (waves)")
            ax.legend(fontsize=8.0, loc="upper center", bbox_to_anchor=(0.5, -0.22),
                      ncol=2, frameon=False)
    axe = fig.add_subplot(gs[1, 0:3 if have_diag else 4])
    sub = cut.dropna(subset=["crb_cross"]).copy()
    sub = sub.sort_values(["structure", "mode", "protocol"]).reset_index(drop=True)
    conv = sub["structure"].map(XBAR).astype(float)
    for col in ["emp_cross", "ci_lo", "ci_hi", "crb_cross"]:
        sub[col] = sub[col] * conv
    x = np.arange(len(sub))
    emp_v = sub["emp_cross"].fillna(0.7 * conv)
    hatch_na = sub["emp_cross"].isna()
    hatch_ex = (~hatch_na) & (sub["emp_cross"] <= 2.0 * conv)
    for i, (v, hn, hx) in enumerate(zip(emp_v, hatch_na, hatch_ex)):
        if hn:
            axe.bar(i - 0.18, v, 0.34, color="#D55E00", hatch="//", alpha=0.5)
        elif hx:
            axe.bar(i - 0.18, v, 0.34, color="#D55E00", hatch="..", alpha=0.75)
        else:
            axe.bar(i - 0.18, v, 0.34, color="#D55E00", alpha=0.9)
    yerr = np.stack([(sub["emp_cross"] - sub["ci_lo"]).fillna(0).values,
                     (sub["ci_hi"] - sub["emp_cross"]).fillna(0).values])
    axe.errorbar(x - 0.18, emp_v, yerr=yerr, fmt="none", ecolor="k", elinewidth=0.7, capsize=1.5)
    axe.bar(x + 0.18, sub["crb_cross"], 0.34, color="k")
    handles = [Patch(facecolor="#D55E00", label="empirical (median, 95% CI)"),
               Patch(facecolor="k", label="CRB prediction"),
               Patch(facecolor="#D55E00", hatch="//", alpha=0.5, label="emp.: never exceeds threshold"),
               Patch(facecolor="#D55E00", hatch="..", alpha=0.75, label="emp.: extrapolated < scan floor")]
    axe.legend(handles=handles, fontsize=8.0, loc="upper center",
               bbox_to_anchor=(0.5, -0.78), ncol=2, frameon=False)
    axe.set_yscale("log"); axe.set_ylim(0.15, 30)
    axe.set_xticks(x)
    axe.set_xticklabels([f'{m[:4]} {s}-{p}' for m, s, p in
                         zip(sub["mode"], sub["structure"], sub["protocol"])],
                        fontsize=8.0, rotation=90, ha="center")
    axe.set_ylabel("mean signal ph/px @ 0.05 w")
    lab(axe, "e", "breakdown budget: empirical vs CRB", sub="hatch: below scan floor (see legend)")
    if have_diag:
        diag = pd.read_csv(diag_path)
        axf = fig.add_subplot(gs[1, 3])
        series = [("defocus", "random", "rmse_signed", "#D55E00", "-", "o", "defocus, signed"),
                  ("defocus", "random", "rmse_signaware", "#D55E00", "--", "s", "defocus, sign-aware"),
                  ("coma_x", "random", "rmse_signed", "#009E73", "-", "o", "coma, signed"),
                  ("coma_x", "oracle", "rmse_signed", "#009E73", ":", "^", "coma, oracle init")]
        for m, init, col, c, ls, mk, lb in series:
            d = diag[(diag["protocol"] == "1f") & (diag["init"] == init) & (diag["mode"] == m)]
            d = d.sort_values("total")
            axf.plot(d["total"] * XBAR["MT"], d[col], ls=ls, marker=mk, ms=3, lw=1.0,
                     color=c, label=lb)
        axf.axhline(0.05, color="k", lw=0.5, alpha=0.4)
        axf.set_xscale("log"); axf.set_yscale("log")
        from matplotlib.ticker import NullLocator
        axf.xaxis.set_minor_locator(NullLocator())
        axf.set_xticks([30, 100, 450])
        axf.set_xticklabels(["30", "100", "450"])
        axf.set_xlabel("total signal ph/px")
        axf.set_ylabel("RMSE (waves)")
        axf.legend(fontsize=8.0, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
        lab(axf, "f", "origin of the 1f scatter", sub="sign-aware smooths even modes;\ncoma: random = oracle")
    savefig(fig, "Fig06.png")


def fig07():
    need = pd.read_csv(ROOT / "results/paper_figures/photons_needed_total.csv")
    adapt_big = pd.read_csv(ROOT / "results/adaptive_protocol_plugin/adaptive_vs_fixed_large_aopt_block.csv")
    adapt_small = pd.read_csv(ROOT / "results/adaptive_protocol_plugin/adaptive_vs_fixed_small_aopt_block.csv")
    sweep = pd.read_csv(ROOT / "results/phase_diversity_sweep/sweep_ER_fixedtotal.csv")
    fig = plt.figure(figsize=(6.5, 4.64))
    gs = fig.add_gridspec(2, 2, hspace=0.9, wspace=0.5)
    XBAR = {"ER": 0.18, "MT": 0.45}
    axa = fig.add_subplot(gs[0, 0])
    w = 0.2
    for k, (s, p) in enumerate([("ER", "1f"), ("ER", "5f"), ("MT", "1f"), ("MT", "5f")]):
        sub = need[(need.structure == s) & (need.protocol == p)]
        vals = [max(sub[sub['mode'] == m]["photons_needed"].iloc[0], 0.5) * XBAR[s] for m in MODES]
        axa.bar(np.arange(4) + (k - 1.5) * w, vals, w, label=f"{s}-{p}")
    axa.set_yscale("log")
    axa.set_xticks(range(4)); axa.set_xticklabels(MODES, fontsize=8.0)
    axa.legend(fontsize=8.0, ncol=2)
    axa.set_ylabel("ph/px for √CRB = 0.05 w")
    lab(axa, "a", "total photon requirement",
        sub="same θ draw as Fig. 6 (±0.2 w)")
    axd = fig.add_subplot(gs[0, 1])
    gb = adapt_big[[c for c in adapt_big.columns if c.startswith("gain_")]].mean()
    gsm = adapt_small[[c for c in adapt_small.columns if c.startswith("gain_")]].mean()
    x = np.arange(4); w = 0.36
    axd.bar(x - w / 2, [gb[f"gain_{m}"] for m in MODES], w, color="0.4", label="large |θ| (~±0.25 w)")
    axd.bar(x + w / 2, [gsm[f"gain_{m}"] for m in MODES], w, color="#0072B2", label="small |θ| (~±0.05 w)")
    axd.axhline(0, color="k", lw=0.5)
    axd.set_xticks(x); axd.set_xticklabels(MODES, fontsize=8.0)
    axd.legend(fontsize=8.0, loc="upper center", bbox_to_anchor=(0.5, -0.28),
               ncol=2, frameon=False)
    axd.set_ylabel("gain over fixed (waves)")
    lab(axd, "b", "adaptive gain over fixed protocol",
        sub="4 frames, s=20/frame (3.6 ph/px), ER")
    axe = fig.add_subplot(gs[1, 0])
    th = {m: torch.tensor(0.02) for m in MODES}
    gt = torch.tensor(load_gt("ER", 1, 1))[None, None].to(DEV)
    cands = candidate_offsets(step=0.25)
    crb_traces = [-expected_info_gain(th, gt, c, wavelength_nm=WL, na=NA, pixel_size_nm=PX,
                                      signal_photons=20.0, background_photons=3.0,
                                      read_noise_std=0.15, criterion="aopt")
                  for c in cands]
    axe.plot(cands, crb_traces, "o-", ms=3.5, lw=1.1, color="#0072B2")
    axe.set_yscale("log")
    axe.set_xlabel("candidate next offset (waves)")
    axe.set_ylabel("tr (F+prior)$^{-1}$ (waves²)")
    ratio = max(crb_traces) / max(min(crb_traces), 1e-12)
    lab(axe, "c", f"A-opt objective at θ̂≈0.02 w",
        sub=f"focal frame ~{ratio:.0f}× worse; 3.6 ph/px, ER")
    axf = fig.add_subplot(gs[1, 1])
    for m in MODES:
        g = sweep[sweep.step == 0.3].groupby("n_frames").agg(e=(f"err_{m}", "mean")).reset_index()
        axf.plot(g["n_frames"], g["e"], "o-", ms=3.5, lw=1.0, color=MCOL[m], label=m)
    axf.set_xticks([1, 2, 3, 5]); axf.set_xlabel("diversity frames")
    axf.set_ylabel("estimation error (waves)")
    axf.legend(fontsize=8.0, ncol=2, loc="upper right")
    lab(axf, "d", "frames → accuracy at fixed total photons",
        sub="blind, 10.8 ph/px total, ER")
    savefig(fig, "Fig07.png")


def fig08():
    df = pd.read_csv(ROOT / "results/blind_calibration/stageA_photons20_multi1.csv")
    sweep = pd.read_csv(ROOT / "results/phase_diversity_sweep/sweep_ER_fixedtotal.csv")
    fig = plt.figure(figsize=(6.5, 4.64))
    gs = fig.add_gridspec(2, 2, hspace=0.9, wspace=0.5)
    ax = fig.add_subplot(gs[0, 0])
    for m in MODES:
        ax.plot(df[f"true_{m}"], df[f"est_{m}"], "o", ms=3.5, color=MCOL[m], label=m, alpha=0.85)
    ax.plot([-0.3, 0.3], [-0.3, 0.3], "k-", lw=0.6, alpha=0.5)
    ax.set_xlabel("truth (waves)"); ax.set_ylabel("estimated (waves)")
    ax.legend(fontsize=8.0, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.30),
              frameon=False)
    lab(ax, "a", "blind 3-frame estimate vs truth",
        sub="BioSR ER+MT, s=20 (3.6/9.0 ph/px), 3 seeds")
    ax = fig.add_subplot(gs[0, 1])
    for m in MODES:
        ax.plot(df["seed"], df[f"err_{m}"], "o-", ms=3, lw=0.8, color=MCOL[m], label=m)
    ax.set_xlabel("seed"); ax.set_ylabel("|err| (waves)")
    ax.legend(fontsize=8.0, loc="upper center", bbox_to_anchor=(0.5, -0.30),
              ncol=4, frameon=False)
    lab(ax, "b", "per-seed error structure", sub="same runs as (a)")
    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(4); w = 0.36
    ax.bar(x - w / 2, [A(f"ER_true_{m}") for m in MODES], w, color="#D55E00", label="truth")
    ax.bar(x + w / 2, [A(f"ER_estnb_{m}") for m in MODES], w, color="#0072B2", label="non-blind 5f est")
    ax.set_xticks(x); ax.set_xticklabels(MODES, fontsize=8.0)
    ax.set_ylabel("coefficient (waves)")


    ax.set_ylim(0, max(A(f"ER_true_{m}") for m in MODES) * 1.5)
    ax.legend(fontsize=8.0, loc="upper right")
    lab(ax, "c", "non-blind 5-frame recovery of a strong mix",
        sub="0.09 w RMS, 9 ph/px, BioSR ER")
    ax = fig.add_subplot(gs[1, 1])
    for m in MODES:
        g = sweep[sweep.step == 0.3].groupby("n_frames").agg(e=(f"err_{m}", "mean")).reset_index()
        ax.plot(g["n_frames"], g["e"], "o-", ms=3.5, lw=1.0, color=MCOL[m], label=m)
    ax.set_xticks([1, 2, 3, 5]); ax.set_xlabel("diversity frames")
    ax.set_ylabel("error (waves)")
    ax.legend(fontsize=8.0, ncol=2, loc="upper right")
    lab(ax, "d", "same content, more frames",
        sub="fixed total budget, blind, BioSR ER")
    savefig(fig, "Fig08.png")


BEAD_REF = (353, 292)
BEAD_ABE = (288, 340)
BEAD_HALF = 16


def bead_stacks():
    try:
        import tifffile
        ref = tifffile.imread(ROOT / "data/cocoa_beads/ref_no_aberration.tif")
        abe = tifffile.imread(ROOT / "data/cocoa_beads/aberrated_75nm.tif")
    except Exception:
        from skimage import io
        ref = io.imread(ROOT / "data/cocoa_beads/ref_no_aberration.tif")
        abe = io.imread(ROOT / "data/cocoa_beads/aberrated_75nm.tif")
    return np.asarray(ref, dtype=np.float32), np.asarray(abe, dtype=np.float32)


def _bead_crop(img, yx, half=BEAD_HALF):
    y, x = yx
    c = img[y - half:y + half + 1, x - half:x + half + 1]
    return c - np.median(c)


def _fwhm_nm(prof, px_nm=86.0):
    p = prof / prof.max()
    above = np.where(p >= 0.5)[0]
    return (above[-1] - above[0]) * px_nm


def pupil_map(coeffs, n=201):

    y, x = np.mgrid[-1:1:1j * n, -1:1:1j * n]
    rho, phi = np.sqrt(x**2 + y**2), np.arctan2(y, x)
    rho_c = np.clip(rho, 0, 1)
    modes = {
        "defocus": 2 * rho_c**2 - 1,
        "astig_x": rho_c**2 * np.cos(2 * phi),
        "astig_y": rho_c**2 * np.sin(2 * phi),
        "coma_x": (3 * rho_c**3 - 2 * rho_c) * np.cos(phi),
        "coma_y": (3 * rho_c**3 - 2 * rho_c) * np.sin(phi),
        "trefoil_x": rho_c**3 * np.cos(3 * phi),
        "trefoil_y": rho_c**3 * np.sin(3 * phi),
        "spherical": 6 * rho_c**4 - 6 * rho_c**2 + 1,
    }
    w = np.zeros((n, n))
    for k, v in coeffs.items():
        w = w + v * modes[k]
    w[rho > 1] = np.nan
    return w


def show_pupil(ax, w, lim, letter, title, sub=None):
    im = ax.imshow(w, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest",
                   extent=[-1, 1, -1, 1], origin="lower")
    ax.set_xticks([]); ax.set_yticks([])
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, ec="k", lw=0.6))
    cax = cbar_axes(ax)
    cb = ax.figure.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=8.0, length=1.5)
    cb.set_label("waves", fontsize=8.0)
    cb.outline.set_linewidth(0.4)
    lab(ax, letter, title, sub=sub)


def fig09():
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ml = pd.read_csv(ROOT / "results/real_beads_multilevel/multilevel.csv")
    c3d = pd.read_csv(ROOT / "results/calibration_3d/calibration_3d.csv")
    ref_st, abe_st = bead_stacks()
    ref, abe = ref_st[50], abe_st[50]
    n = lambda x: np.clip((x - np.percentile(x, 1)) / (np.percentile(x, 99.8) - np.percentile(x, 1)), 0, 1)
    refn, aben = n(ref), n(abe)

    fig = plt.figure(figsize=(6.5, 7.0))
    gs = fig.add_gridspec(3, 3, hspace=0.9, wspace=0.8, height_ratios=[1, 1, 0.85])

    for k, (img, yx, name, sub) in enumerate([(refn, BEAD_REF, "CoCoA reference field", "no aberration"),
                                              (aben, BEAD_ABE, "CoCoA 75 nm DM field", "DM: astig/tref/coma-y")]):  # noqa: E501
        ax = fig.add_subplot(gs[0, k])

        show(ax, img, gamma=0.6, px_nm=86.0, bar_nm=10000, clabel="intensity",
             clabelpad=0.5)
        y, x = yx
        rect = plt.Rectangle((x - BEAD_HALF, y - BEAD_HALF), 2 * BEAD_HALF + 1, 2 * BEAD_HALF + 1,
                             fill=False, ec="w", lw=0.7)
        ax.add_patch(rect)
        axins = inset_axes(ax, width="42%", height="42%", loc="upper right", borderpad=1.1)
        bead = _bead_crop(ref if k == 0 else abe, yx)
        axins.imshow(np.clip(bead / bead.max(), 0, 1) ** 0.7, cmap="inferno", interpolation="nearest")
        axins.set_xticks([]); axins.set_yticks([])
        nb = 1000.0 / 86.0
        axins.plot([2 * BEAD_HALF - nb - 1, 2 * BEAD_HALF - 1], [2 * BEAD_HALF - 2, 2 * BEAD_HALF - 2],
                   color="w", lw=1.8, solid_capstyle="butt")
        for spine in axins.spines.values():
            spine.set_edgecolor("w"); spine.set_linewidth(0.7)
        lab(ax, chr(97 + k), name, sub=sub + "; bar 1 µm" if k == 0 else sub)

    axc = fig.add_subplot(gs[0, 2])
    axnm = (np.arange(2 * BEAD_HALF + 1) - BEAD_HALF) * 86.0
    for img, yx, col, labtxt in [(ref, BEAD_REF, "k", "reference"), (abe, BEAD_ABE, "#D55E00", "aberrated")]:
        c = _bead_crop(img, yx)
        h = BEAD_HALF
        xcut, ycut = c[h, :], c[:, h]
        axc.plot(axnm, xcut / xcut.max(), color=col, lw=1.2,
                 label=f"{labtxt} x ({_fwhm_nm(xcut):.0f} nm)")
        axc.plot(axnm, ycut / ycut.max(), color=col, lw=0.9, ls="--", alpha=0.75,
                 label=f"{labtxt} y ({_fwhm_nm(ycut):.0f} nm)")
    axc.set_xlabel("offset from bead center (nm)")
    axc.set_ylabel("intensity (norm.)", labelpad=0.5)
    axc.legend(fontsize=8.0, loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2,
               frameon=False, handlelength=1.3, borderpad=0.05, labelspacing=0.25)
    lab(axc, "c", "single-bead profiles",
        sub="point source; FWHM in legend")

    v2 = pd.read_csv(ROOT / "results/real_aberration_validation_v3/cocoa_v3_comparison.csv")
    axd = fig.add_subplot(gs[1, 0])
    order = ["astig_x", "trefoil_y", "coma_y", "defocus", "astig_y", "coma_x", "spherical"]
    sub = v2.set_index("mode").loc[order].reset_index()
    x = np.arange(len(sub)); w = 0.36
    axd.bar(x - w / 2, sub["gt_waves"], w, color="#D55E00", label="mirror-applied GT")
    axd.bar(x + w / 2, sub["est_differential"], w, color="#0072B2",
            label="differential blind est")
    axd.set_xticks(x)


    axd.set_xticklabels([mm.replace("_", "-") for mm in order], fontsize=8.0, rotation=90, ha="center")
    axd.axhline(0, color="k", lw=0.5)
    axd.set_ylabel("coefficient (waves)")
    axd.set_ylim(-0.16, 0.16)


    lab(axd, "d", "differential estimate vs DM truth",
        sub="3 z-frames; orange=truth, blue=est")


    import json
    MOD4 = ["defocus", "astig_x", "coma_x", "spherical"]
    SG = dict(zip(MOD4, [1 / np.sqrt(3), 1 / np.sqrt(6), 1 / np.sqrt(8), 1 / np.sqrt(5)]))

    def rms_stat(df, err_cols=True):
        if err_cols:
            return np.sqrt(sum(df[f"err_{m}"] ** 2 * SG[m] ** 2 for m in MOD4))
        return df["err_rms"]

    ml2 = ml.copy()
    ml2["stat"] = rms_stat(ml2)
    ml2["nom"] = ml2["magnitude"].map(
        {0.05: 0.028, 0.10: 0.056, 0.15: 0.084, 0.20: 0.112, 0.25: 0.140, 0.30: 0.168})
    axp = pd.read_csv(ROOT / "results/axial_prospective/empirical_sweep.csv")
    axp["stat"] = np.sqrt(sum(axp[f"signed_sq_{m}"] * SG[m] ** 2 for m in MOD4))
    tr = pd.read_csv(ROOT / "results/axial_basin/transition_scan.csv")
    tr["stat"] = rms_stat(tr)
    axial = pd.concat([axp[["rms", "stat", "seed"]], tr[["rms", "stat", "seed"]]])

    axe = fig.add_subplot(gs[1, 1])
    g2 = ml2.groupby("nom").stat.median().reset_index()
    axe.plot(ml2["nom"], ml2["stat"], "o", ms=2.6, mfc="none", mec=PAL["red"], mew=0.5,
             alpha=0.75, zorder=3)
    axe.plot(g2["nom"], g2["stat"], "o-", ms=3.4, lw=1.2, color=PAL["red"],
             label="2D blind", zorder=4)
    axe.plot(axial["rms"], axial["stat"], "s", ms=2.6, mfc="none", mec=PAL["blue"],
             mew=0.5, alpha=0.75, zorder=3)

    bins = pd.concat([axp.assign(src="p"), tr.assign(src="t")])
    med = (bins.groupby(["src", "magnitude"])
           .agg(r=("rms", "median"), s=("stat", "median")).reset_index()
           .sort_values("r"))
    axe.plot(med["r"], med["s"], "s--", ms=3.4, lw=1.2, color=PAL["blue"],
             label="3D axial", zorder=4)
    axe.axhline(0.05, color="k", lw=0.9, ls=":", zorder=2)
    axe.axvspan(0.147, 0.189, color="0.55", alpha=0.16, zorder=1)
    axe.set_xscale("log"); axe.set_yscale("log")
    axe.set_xlim(0.02, 0.62); axe.set_ylim(1.2e-3, 2.0)
    axe.set_xticks([0.03, 0.1, 0.2, 0.4])
    axe.set_xticklabels(["0.03", "0.1", "0.2", "0.4"])
    axe.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axe.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axe.tick_params(axis="both", which="minor", length=0)
    axe.set_xlabel("applied wavefront RMS (waves)")
    axe.set_ylabel("error, Eq. (9) RMS (waves)")

    axe.legend(fontsize=8.0, loc="upper left", frameon=False, handlelength=1.3,
               borderpad=0.1, labelspacing=0.3, borderaxespad=0.25)
    lab(axe, "e", "working range, both geometries",
        sub="dotted: 0.05 w target; band: onset")

    axf = fig.add_subplot(gs[1, 2])
    ini = pd.read_csv(ROOT / "results/axial_basin/init_scan.csv")
    ini = ini[ini.magnitude == 0.5].copy()
    ini["dnll"] = ini["terminal_nll"] - ini.groupby("seed")["terminal_nll"].transform("min")
    good = ini[ini.err_rms < 0.05]; bad = ini[ini.err_rms > 0.25]
    axf.plot(good["err_rms"], good["dnll"], "o", ms=2.6, color=PAL["blue"],
             label="correct basin", alpha=0.85)
    axf.plot(bad["err_rms"], bad["dnll"], "^", ms=2.8, color=PAL["red"],
             label="sign-reversed basin", alpha=0.85)
    axf.set_xscale("log")
    axf.set_xlim(2e-3, 1.2); axf.set_ylim(-0.0008, 0.015)
    axf.set_xlabel("coefficient error (waves, RMS)")

    axf.set_ylabel("NLL excess")
    axf.legend(fontsize=8.0, loc="upper center", frameon=False, handlelength=1.3,
               borderpad=0.15, labelspacing=0.28)
    lab(axf, "f", "the wrong answer fits as well",
        sub=f"{len(ini)} inits; excess {bad['dnll'].median():.4f}")

    axg = fig.add_subplot(gs[2, 0])
    prof = json.loads((ROOT / "results/axial_basin/profile.json").read_text(encoding="utf-8"))
    t = np.asarray(prof["nll_on_fixed_counts"]["t"])
    yf = np.asarray(prof["nll_on_fixed_counts"]["nll"]); yf = yf - yf.min()
    ym = np.asarray(prof["nll_on_fresh_realizations"]["nll_mean"])
    ys = np.asarray(prof["nll_on_fresh_realizations"]["nll_std"])
    ym = ym - ym.min()
    axg.fill_between(t, ym - ys, ym + ys, color=PAL["sky"], alpha=0.45, lw=0)
    axg.plot(t, ym, "--", color=PAL["blue"], lw=1.1)
    axg.plot(t, yf, "-", color="0.15", lw=1.3)
    axg.plot([0], [0], "o", ms=3.4, color=PAL["green"])
    axg.plot([1], [yf[-1]], "s", ms=3.4, color=PAL["red"])


    axg.annotate("barrier", xy=(0.5, yf.max() * 1.01), xytext=(0.5, yf.max() * 1.10),
                 fontsize=8.0, ha="center", va="bottom",
                 arrowprops=dict(arrowstyle="-", lw=0.6, color="0.3"))
    axg.set_xlim(-0.02, 1.02); axg.set_ylim(-0.002, yf.max() * 1.52)
    axg.set_xlabel("interpolation: correct $\\to$ sign-reversed")
    axg.set_ylabel("NLL excess (per pixel)")
    lab(axg, "g", "two wells, one barrier",
        sub=f"green correct; red wrong (+{yf[-1]:.3f})\n"
            f"blue band: 8 fresh draws")

    axh = fig.add_subplot(gs[2, 1])
    cold = axp[axp.magnitude == 0.5].set_index("seed")["stat"].reindex(range(5))
    sb = pd.read_csv(ROOT / "results/axial_prospective/step_budget_check.csv")
    cold2 = sb[sb["mag"] == 0.5].set_index("seed")["plain_rmse"].reindex(range(5))
    cont = pd.read_csv(ROOT / "results/axial_basin/continuation.csv")
    warm = cont[cont.arm == "warm_from_0.30"].set_index("seed")["err_rms"].reindex(range(5))
    for s in range(5):
        axh.plot([s - 0.18, s + 0.18], [cold[s], warm[s]], "-", color="0.75", lw=0.7, zorder=1)
    axh.plot(range(5), cold2, "+", ms=6.0, mew=1.2, color="0.35", label="cold, 2000 steps")
    axh.plot(range(5), cold, "x", ms=4.6, mew=1.1, color=PAL["red"], label="cold, 400 steps")
    axh.plot(range(5), warm, "o", ms=4.2, color=PAL["green"], label="warm start, 400 steps")
    axh.axhline(0.05, color="k", lw=0.9, ls=":", zorder=0)
    axh.set_yscale("log"); axh.set_ylim(1.5e-3, 40.0)
    axh.set_xticks(range(5)); axh.set_xticklabels([f"s{s}" for s in range(5)])
    axh.set_xlabel("random draw (fixed photon counts)")
    axh.set_ylabel("error (waves, RMS)")


    axh.legend(fontsize=8.0, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3,
               frameon=False, handlelength=1.2, columnspacing=1.0,
               borderpad=0.1, labelspacing=0.3, borderaxespad=0.25)
    lab(axh, "h", "initialization, not iteration",
        sub="5x steps: still fails; warm start OK")


    gt_c = dict(zip(sub["mode"], sub["gt_waves"]))
    est_c = dict(zip(sub["mode"], sub["est_differential"]))
    wres = pupil_map(est_c) - pupil_map(gt_c)
    show_pupil(fig.add_subplot(gs[2, 2]), wres, 0.06, "i", "residual wavefront",
               sub=f"est−applied, {np.nanstd(wres):.3f} w RMS")
    savefig(fig, "Fig09.png")


def fig10():
    from skimage.metrics import structural_similarity as ssim
    fig = plt.figure(figsize=(6.5, 4.46))
    gs = fig.add_gridspec(3, 6, hspace=0.5, wspace=0.30, height_ratios=[1, 1, 0.9])
    keys = [("ER_gt", "GT"), ("ER_obs", "obs"), ("ER_rl_nom", "RL\nnominal"),
            ("ER_rl_cal", "RL\ncalibrated"), ("ER_rl_true", "RL\noracle"), ("ER_blind_x", "blind\nrecon")]
    for k, (key, t) in enumerate(keys):
        ax = fig.add_subplot(gs[0, k])
        last = k == len(keys) - 1
        show(ax, A(key), gamma=0.7, px_nm=PX, bar_nm=2000, cbar=last,
             clabel="norm. intensity" if last else "")
        lab(ax, chr(97 + k), t)
    for k, (key, t) in enumerate(keys):
        img = A(key)[96:160, 96:160]
        ax = fig.add_subplot(gs[1, k])
        last = k == len(keys) - 1
        show(ax, img, gamma=0.8, px_nm=PX, bar_nm=1000, cbar=last,
             clabel="norm. intensity" if last else "")
        lab(ax, chr(103 + k), f"zoom" if k else "zoom of (a-f)")
    gt_img = A("ER_gt")
    names = ["obs", "RL nom", "RL cal", "RL oracle", "blind"]
    ims = [A("ER_obs"), A("ER_rl_nom"), A("ER_rl_cal"), A("ER_rl_true"), A("ER_blind_x")]
    ss = [ssim(gt_img, im, data_range=1.0) for im in ims]
    ps = [10 * np.log10(1.0 / max(np.mean((gt_img - im) ** 2), 1e-12)) for im in ims]
    axi = fig.add_subplot(gs[2, 0:3])
    axi.bar(np.arange(5), ps, 0.55, color=["0.5", "#D55E00", "#0072B2", "#009E73", "#CC79A7"])
    axi.set_xticks(range(5))
    axi.set_xticklabels(names, fontsize=8.0, rotation=25, ha="right")
    axi.set_ylabel("PSNR (dB)")
    lab(axi, "m", "fidelity vs GT", sub="BioSR ER, 0.09 w RMS, 5.4 ph/px")
    axj = fig.add_subplot(gs[2, 3:6])
    axj.bar(np.arange(5), ss, 0.55, color=["0.5", "#D55E00", "#0072B2", "#009E73", "#CC79A7"])
    axj.set_xticks(range(5))
    axj.set_xticklabels(names, fontsize=8.0, rotation=25, ha="right")
    axj.set_ylabel("SSIM"); axj.set_ylim(0, 1)
    lab(axj, "n", "structure vs GT", sub="same data as (m)")


    pm, pn = axi.get_position(), axj.get_position()
    sh = 0.09 * pm.width
    axi.set_position([pm.x0, pm.y0, pm.width - sh, pm.height])
    axj.set_position([pn.x0 + sh, pn.y0, pn.width - sh, pn.height])
    savefig(fig, "Fig10.png")


def fig11():
    ds = pd.read_csv(ROOT / "results/downstream_endpoint/downstream_endpoint.csv")
    sweeps = {s: pd.read_csv(ROOT / f"results/biosr_sweep_{s}/per_run_metrics.csv") for s in ["ER", "MT"]}
    fig = plt.figure(figsize=(6.5, 4.83))
    gs = fig.add_gridspec(2, 3, hspace=0.9, wspace=0.8)
    ccp = ds[ds.task == "spots"]
    axa = fig.add_subplot(gs[0, 0])
    x = np.arange(len(ccp)); w = 0.27
    axa.bar(x - w, ccp["f1_obs"], w, color="0.5", label="obs")
    axa.bar(x, ccp["f1_uncal"], w, color="#D55E00", label="RL nominal")
    axa.bar(x + w, ccp["f1_cal"], w, color="#0072B2", label="RL calibrated")
    axa.set_xticks(x); axa.set_xticklabels([f"seed {s}" for s in ccp["seed"]], fontsize=8.0)
    axa.set_ylabel("spot-detection F1"); axa.set_ylim(0, 0.9)
    axa.legend(fontsize=8.0, loc="upper center", bbox_to_anchor=(0.5, -0.32),
               ncol=3, frameon=False)
    lab(axa, "a", "CCP spot counting (BioTISR)",
        sub="0.13 w RMS injected; blind 5f + RL")
    axb = fig.add_subplot(gs[0, 1])
    axb.bar(x - w, ccp["counterr_obs"], w, color="0.5", label="obs")
    axb.bar(x, ccp["counterr_uncal"], w, color="#D55E00", label="RL nominal")
    axb.bar(x + w, ccp["counterr_cal"], w, color="#0072B2", label="RL calibrated")
    axb.set_yscale("log"); axb.set_ylim(8, 5e3 * 10); axb.set_xticks(x)
    axb.set_xticklabels([f"seed {s}" for s in ccp["seed"]], fontsize=8.0)
    axb.set_ylabel("|count error|")

    axb.legend(fontsize=8.0, loc="upper right")
    lab(axb, "b", "count error (log scale)", sub="seed 2 outside working range")
    axc = fig.add_subplot(gs[0, 2])
    allr = []
    for s, df in sweeps.items():
        df = df.dropna(subset=["hallucination_auc"]).copy(); df["s"] = s
        allr.append(df)
    alldf = pd.concat(allr)
    g = alldf.groupby(["s", "na", "defocus_waves", "photons_per_pixel"]).agg(
        cg=("c_gamma", "mean"), auc=("hallucination_auc", "mean")).reset_index()
    pcols = {2.0: "#D55E00", 5.0: "#E69F00", 10.0: "#0072B2"}
    for (s, ph), d in g.groupby(["s", "photons_per_pixel"]):
        axc.scatter(d["cg"], d["auc"], s=10, marker="o" if s == "ER" else "s",
                    color=pcols.get(ph, "k"), alpha=0.75, linewidths=0)
    axc.set_xlabel("C$_\\gamma$ (operator content pass-through)")
    axc.set_ylabel("hallucination-detection AUC")
    handles = [plt.Line2D([], [], marker="o", ls="", color=pcols[p], label=f"{p:g} ph/px",
                          markersize=4) for p in [2.0, 5.0, 10.0]]
    handles += [plt.Line2D([], [], marker="o", ls="", color="0.4", label="ER", markersize=4),
                plt.Line2D([], [], marker="s", ls="", color="0.4", label="MT", markersize=4)]
    axc.legend(handles=handles, fontsize=8.0, loc="upper center",
               bbox_to_anchor=(0.5, -0.34), ncol=3, frameon=False)
    lab(axc, "c", "content identifiability → AUC",
        sub="48 unseen operators; ρ ≥ 0.70")
    axd = fig.add_subplot(gs[1, 0])
    for mag, c, ls in [(0.0, "k", "-"), (0.1, "#0072B2", "--"), (0.2, "#D55E00", "-.")]:
        cfg = MicroscopyPSFConfig(wavelength_nm=WL, na=NA, pixel_size_nm=PX, kernel_size=KS,
                                  aberrations_waves={"coma_x": mag} if mag else {})
        otf = psf_to_otf(make_psf(cfg), (256, 256)).abs().numpy() ** 2
        prof = np.fft.fftshift(otf / otf.max())
        h, w = prof.shape
        y, x = np.indices(prof.shape)
        r = np.sqrt((y - h // 2) ** 2 + (x - w // 2) ** 2).astype(int)
        prof = np.bincount(r.ravel(), prof.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
        axd.plot(np.arange(len(prof)) / 128.0, prof, lw=1.1, color=c, ls=ls, label=f"coma {mag:.1f} w")
    axd.axhline(0.05, color="r", lw=0.5, ls="--"); axd.set_xlim(0, 0.8); axd.set_ylim(0, 1.02)
    axd.set_xlabel("|q| (rel. to cutoff)")
    axd.set_ylabel("|OTF|² (norm.)")
    axd.legend(fontsize=8.0)
    lab(axd, "d", "aberration eats the content band", sub="coma 0/0.1/0.2 w, NA 1.2")
    axe = fig.add_subplot(gs[1, 1])
    defs = np.linspace(0, 1.5, 16)
    for na, c, ls in [(1.2, "#0072B2", "-"), (0.8, "#009E73", "--")]:
        cg = []
        for d in defs:
            cfg = MicroscopyPSFConfig(wavelength_nm=WL, na=na, pixel_size_nm=PX, kernel_size=KS,
                                      aberrations_waves={"defocus": float(d)})
            otf = psf_to_otf(make_psf(cfg), (256, 256)).abs().numpy()
            cg.append(float(((otf / otf.max()).flatten() >= 0.05).mean()))
        axe.plot(defs, cg, marker="o", ls=ls, ms=3, lw=1.0, color=c, label=f"NA {na}")
    axe.set_xlabel("defocus (waves)"); axe.set_ylabel("C$_{0.05}$")
    axe.legend(fontsize=8.0, loc="upper right")
    lab(axe, "e", "C$_\\gamma$ vs defocus", sub="2D operator, γ = 0.05")
    savefig(fig, "Fig11.png")


def fig12():
    mm = pd.read_csv(ROOT / "results/model_mismatch/model_mismatch_summary.csv")
    ml = pd.read_csv(ROOT / "results/real_beads_multilevel/multilevel.csv")
    c3d = pd.read_csv(ROOT / "results/calibration_3d/calibration_3d.csv")
    gt = torch.tensor(load_gt("ER", 1, 1))[None, None].to(DEV)
    true = {"defocus": -0.11, "astig_x": 0.06, "coma_x": 0.17, "spherical": 0.19}
    from phyguard.estimation.calibration import parameterized_psf
    coeffs = {m: torch.tensor(true[m], requires_grad=True) for m in MODES}
    psf = parameterized_psf(coeffs, wavelength_nm=WL, na=NA, pixel_size_nm=PX)
    mic = PhotonMicroscope(psf=psf, signal_photons=50.0, background_photons=7.5, read_noise_std=0.15)
    lam = mic.expected_counts(gt).detach().clamp_min(1e-6)
    v = []
    for m in MODES:
        g = torch.autograd.grad(mic.expected_counts(gt).sum(), coeffs[m], retain_graph=True)[0]
        v.append(float(g))
    v = np.array(v)
    Fbug = np.outer(v, v) * float((1.0 / lam.cpu().numpy()).sum())
    Ffix = parameter_fisher({m: torch.tensor(true[m]) for m in MODES}, gt,
                            wavelength_nm=WL, na=NA, pixel_size_nm=PX,
                            signal_photons=50.0, background_photons=7.5, read_noise_std=0.15)

    fig = plt.figure(figsize=(6.5, 4.64))
    gs = fig.add_gridspec(2, 2, hspace=0.9, wspace=0.5)
    axa = fig.add_subplot(gs[0, 0])
    for param, mk, ls, labtxt in [("na", "o", "-", "NA (≡ pixel size)"), ("wavelength_nm", "s", "--", "wavelength")]:
        g = mm[mm.param == param]
        axa.plot(g["factor"], g["rms_total"], marker=mk, ls=ls, ms=3.5, lw=1.1, label=labtxt)
    axa.axvline(1.0, color="k", lw=0.5, alpha=0.4)


    yl = axa.get_ylim()
    axa.set_ylim(yl[0], yl[1] * 1.5)
    axa.legend(fontsize=8.0, loc="upper right")
    axa.set_xlabel("assumed / true system parameter")
    axa.set_ylabel("total bias (waves RMS)")
    lab(axa, "a", "system-model mismatch",
        sub="non-blind 5f, 9 ph/px; ±1% → bias <0.006 w")
    axb = fig.add_subplot(gs[0, 1])
    g = mm[mm.param == "na"]
    for m in MODES:
        axb.plot(g["factor"], g[f"bias_{m}"], "o-", ms=3, lw=1.0, color=MCOL[m], label=m)
    axb.axhline(0, color="k", lw=0.5); axb.axvline(1.0, color="k", lw=0.5, alpha=0.4)
    axb.set_xlabel("assumed / true NA"); axb.set_ylabel("bias (waves)")
    axb.legend(fontsize=8.0, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.30),
               frameon=False)
    lab(axb, "b", "defocus absorbs the mismatch", sub="same runs as (a); NA·pixel/λ degeneracy")
    axc = fig.add_subplot(gs[1, 0])
    g2 = ml.groupby("magnitude").agg(v=("mean_err", "mean"), sd=("mean_err", "std")).reset_index()
    g3 = c3d.groupby("magnitude").agg(v=("mean_err", "mean"), sd=("mean_err", "std")).reset_index()
    axc.errorbar(g2["magnitude"], g2["v"], yerr=g2["sd"], fmt="o-", ms=3.5, lw=1.0, capsize=2.5,
                 color="#D55E00", label="2D blind (crosses target at 0.09 w RMS)")
    axc.errorbar(g3["magnitude"], g3["v"], yerr=g3["sd"], fmt="s--", ms=3.5, lw=1.0, capsize=2.5,
                 color="#0072B2", label="3D (works to 0.15 w RMS)")
    axc.set_xlabel("applied aberration (waves RMS)")
    axc.set_ylabel("estimation error (waves)")
    axc.legend(fontsize=8.0, loc="upper center", bbox_to_anchor=(0.5, -0.30),
               ncol=2, frameon=False)
    lab(axc, "c", "working range, quantified",
        sub="2D blind crosses target at 0.09 w; 3D works to 0.15 w")
    axd = fig.add_subplot(gs[1, 1])
    e1 = np.linalg.eigvalsh(Fbug)[::-1]
    e2 = np.linalg.eigvalsh(Ffix)[::-1]
    axd.bar(np.arange(4) - 0.18, np.abs(e1), 0.34, color="0.5", label="first version (rank one)")
    axd.bar(np.arange(4) + 0.18, e2, 0.34, color="#0072B2", label="fixed (per-pixel Jacobian)")
    axd.set_yscale("log"); axd.set_xticks(range(4))
    axd.set_xticklabels(["λ1", "λ2", "λ3", "λ4"], fontsize=8.0)
    axd.set_ylabel("Fisher eigenvalues")
    axd.legend(fontsize=8.0, loc="upper right")
    lab(axd, "d", "Fisher eigenvalue spectrum", sub="implementation check")
    savefig(fig, "Fig12.png")


def main():
    fig01()
    fig02()
    fig03()
    fig04()
    fig05()
    fig06()
    fig07()
    fig08()
    fig09()
    fig10()
    fig11()
    fig12()
    print("done ->", OUT)


if __name__ == "__main__":
    main()
