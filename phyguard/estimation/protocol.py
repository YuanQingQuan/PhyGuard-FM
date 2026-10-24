from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from .info import parameter_fisher, crb_identifiable


def candidate_offsets(step: float = 0.25, n: int = 5) -> List[float]:
    return [float(i) * step for i in range(-(n // 2), n // 2 + 1)]


def expected_info_gain(
    theta: Dict[str, torch.Tensor],
    structure: torch.Tensor,
    candidate: float,
    *,
    wavelength_nm: float,
    na: float,
    pixel_size_nm: float,
    signal_photons: float,
    background_photons: float,
    read_noise_std: float,
    kernel_size: int = 41,
    criterion: str = "aopt",
    prior_std: float = 0.5,
    accumulated: np.ndarray | None = None,
) -> float:

    th = dict(theta)
    th["defocus"] = th["defocus"] + candidate
    F = parameter_fisher(
        th, structure,
        wavelength_nm=wavelength_nm, na=na, pixel_size_nm=pixel_size_nm,
        signal_photons=signal_photons, background_photons=background_photons,
        read_noise_std=read_noise_std, kernel_size=kernel_size,
    )
    if F is None:
        return 0.0
    if accumulated is not None:
        F = F + accumulated
    if criterion == "trace":

        crb, _ = crb_identifiable(F, list(theta.keys()))
        return float(np.sum(1.0 / np.maximum(list(crb.values()), 1e-12)))
    eps = 1.0 / prior_std**2
    Freg = F + eps * np.eye(F.shape[0])
    if criterion == "aopt":

        return -float(np.trace(np.linalg.inv(Freg)))
    if criterion == "eopt":

        return float(np.linalg.eigvalsh(Freg).min())
    raise ValueError(f"unknown criterion {criterion!r}")


def select_next_offset(
    theta: Dict[str, torch.Tensor],
    structure: torch.Tensor,
    candidates: List[float],
    *,
    wavelength_nm: float,
    na: float,
    pixel_size_nm: float,
    signal_photons: float,
    background_photons: float,
    read_noise_std: float,
    kernel_size: int = 41,
    criterion: str = "aopt",
    prior_std: float = 0.5,
    accumulated: np.ndarray | None = None,
) -> tuple[float, Dict[float, float]]:

    gains = {}
    for cand in candidates:
        gains[cand] = expected_info_gain(
            theta, structure, cand,
            wavelength_nm=wavelength_nm, na=na, pixel_size_nm=pixel_size_nm,
            signal_photons=signal_photons, background_photons=background_photons,
            read_noise_std=read_noise_std, kernel_size=kernel_size,
            criterion=criterion, prior_std=prior_std, accumulated=accumulated,
        )
    best = max(gains, key=gains.get)
    return best, gains


def fixed_offsets(n_frames: int, step: float) -> List[float]:

    if n_frames == 1:
        return [0.0]
    if n_frames == 2:
        return [-step, step]
    if n_frames == 3:
        return [-step, 0.0, step]
    if n_frames == 4:
        return [-1.5 * step, -0.5 * step, 0.5 * step, 1.5 * step]
    if n_frames == 5:
        return [-2 * step, -step, 0.0, step, 2 * step]
    raise ValueError(n_frames)
