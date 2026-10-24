from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch

from ..optics.psf import MicroscopyPSFConfig, make_psf
from ..optics.forward import PhotonMicroscope
from ..likelihood import poisson_gaussian_nll

ABERRATION_MODES = ("defocus", "astig_x", "astig_y", "coma_x", "coma_y", "spherical")


def parameterized_psf(
    coeffs: Dict[str, torch.Tensor],
    *,
    wavelength_nm: float,
    na: float,
    pixel_size_nm: float,
    kernel_size: int = 41,
    n_obj: float = 1.0,
    axial_offset_um: float = 0.0,
) -> torch.Tensor:

    cfg = MicroscopyPSFConfig(
        wavelength_nm=wavelength_nm,
        na=na,
        pixel_size_nm=pixel_size_nm,
        kernel_size=kernel_size,
        aberrations_waves=coeffs,
        n_obj=n_obj,
        axial_offset_um=axial_offset_um,
    )
    return make_psf(cfg)


@dataclass
class CalibrationResult:
    true_coeffs: Dict[str, float]
    estimated_coeffs: Dict[str, float]
    loss_history: List[float]
    fisher_info: np.ndarray | None = None
    crb: Dict[str, float] = field(default_factory=dict)
    identifiable_magnitude: Dict[str, float] = field(default_factory=dict)


def estimate_aberrations(
    counts: torch.Tensor,
    structure: torch.Tensor,
    *,
    modes: List[str],
    wavelength_nm: float,
    na: float,
    pixel_size_nm: float,
    signal_photons: float,
    background_photons: float,
    read_noise_std: float,
    init: Dict[str, float] | None = None,
    n_steps: int = 800,
    lr: float = 0.08,
    kernel_size: int = 41,
    n_restarts: int = 6,
    seed: int = 0,
) -> CalibrationResult:

    rng = np.random.default_rng(seed)
    best = None
    for r in range(n_restarts):
        if init is not None and r == 0:
            start = {m: float(init.get(m, 0.0)) for m in modes}
        else:
            start = {m: float(rng.uniform(-0.25, 0.25)) for m in modes}
        coeffs = {
            m: torch.tensor(start[m], requires_grad=True) for m in modes
        }
        opt = torch.optim.Adam(list(coeffs.values()), lr=lr)
        history = []
        for _ in range(n_steps):
            opt.zero_grad()
            psf = parameterized_psf(
                coeffs, wavelength_nm=wavelength_nm, na=na,
                pixel_size_nm=pixel_size_nm, kernel_size=kernel_size,
            )
            mic = PhotonMicroscope(
                psf=psf, signal_photons=signal_photons,
                background_photons=background_photons, read_noise_std=read_noise_std,
            )
            mu = mic.expected_counts(structure)
            loss = poisson_gaussian_nll(counts, mu, read_noise_std)
            loss.backward()
            opt.step()
            history.append(float(loss.detach()))
        final_loss = history[-1]
        if best is None or final_loss < best[0]:
            best = (final_loss, {m: float(coeffs[m].detach()) for m in modes}, history)

    _, est, history = best
    coeffs = {m: torch.tensor(est[m], requires_grad=True) for m in modes}
    fisher = fisher_information(
        coeffs, structure,
        wavelength_nm=wavelength_nm, na=na, pixel_size_nm=pixel_size_nm,
        signal_photons=signal_photons, background_photons=background_photons,
        read_noise_std=read_noise_std, kernel_size=kernel_size,
    )
    crb = {}
    n_degenerate = 0
    if fisher is not None:


        w, V = np.linalg.eigh(fisher)
        tol = max(np.abs(w).max() * 1e-8, 1e-12)
        keep = w > tol
        n_degenerate = int((~keep).sum())
        inv = (V[:, keep] * (1.0 / np.maximum(w[keep], 1e-300))) @ V[:, keep].T
        crb_vals = np.diag(inv)
        crb = {m: float(crb_vals[i]) for i, m in enumerate(modes)}
    result = CalibrationResult(true_coeffs={}, estimated_coeffs=est,
                               loss_history=history, fisher_info=fisher, crb=crb)
    result.n_degenerate_directions = n_degenerate
    return result


def estimate_aberrations_multiframe(
    counts: torch.Tensor,
    structure: torch.Tensor,
    *,
    modes: List[str],
    defocus_offsets: List[float],
    wavelength_nm: float,
    na: float,
    pixel_size_nm: float,
    signal_photons: float,
    background_photons: float,
    read_noise_std: float,
    n_steps: int = 500,
    lr: float = 0.08,
    kernel_size: int = 41,
    n_restarts: int = 4,
    seed: int = 0,
) -> CalibrationResult:

    if counts.ndim != 4 or counts.shape[0] != len(defocus_offsets):
        raise ValueError("counts must be [F,1,H,W] with F == len(defocus_offsets)")
    rng = np.random.default_rng(seed)

    def _nll(theta: Dict[str, torch.Tensor]) -> torch.Tensor:
        tot = 0.0
        for i, off in enumerate(defocus_offsets):
            th = dict(theta)
            th["defocus"] = th["defocus"] + float(off)
            psf = parameterized_psf(
                th, wavelength_nm=wavelength_nm, na=na,
                pixel_size_nm=pixel_size_nm, kernel_size=kernel_size,
            )
            mic = PhotonMicroscope(
                psf=psf, signal_photons=signal_photons,
                background_photons=background_photons, read_noise_std=read_noise_std,
            )
            tot = tot + poisson_gaussian_nll(
                counts[i], mic.expected_counts(structure), read_noise_std
            )
        return tot / len(defocus_offsets)

    best = None
    for r in range(n_restarts):
        start = {m: float(rng.uniform(-0.25, 0.25)) for m in modes}
        theta = {m: torch.tensor(start[m], requires_grad=True) for m in modes}
        opt = torch.optim.Adam(list(theta.values()), lr=lr)
        history = []
        for _ in range(n_steps):
            opt.zero_grad()
            loss = _nll(theta)
            loss.backward()
            opt.step()
            history.append(float(loss.detach()))
        if best is None or history[-1] < best[0]:
            best = (history[-1], {m: float(theta[m].detach()) for m in modes}, history)

    _, est, history = best
    return CalibrationResult(true_coeffs={}, estimated_coeffs=est, loss_history=history)


def fisher_information(
    coeffs: Dict[str, torch.Tensor],
    structure: torch.Tensor,
    *,
    wavelength_nm: float,
    na: float,
    pixel_size_nm: float,
    signal_photons: float,
    background_photons: float,
    read_noise_std: float,
    kernel_size: int = 41,
) -> np.ndarray | None:

    from .info import parameter_fisher

    return parameter_fisher(
        coeffs, structure, wavelength_nm=wavelength_nm, na=na,
        pixel_size_nm=pixel_size_nm, signal_photons=signal_photons,
        background_photons=background_photons, read_noise_std=read_noise_std,
        kernel_size=kernel_size,
    )
