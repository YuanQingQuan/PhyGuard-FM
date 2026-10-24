from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as tF

from ..optics.psf import (
    MicroscopyPSFConfig,
    make_psf,
    psf_to_otf,
    psf_aberration_derivatives,
)
from ..optics.forward import PhotonMicroscope


@dataclass
class InformationBudget:

    param_crb: Dict[str, float]
    param_fisher_eigs: List[float]
    n_degenerate_directions: int
    content_c_gamma: float
    content_eff_rank_fraction: float
    content_fisher_diag_mean: float
    budget_ratio_param_vs_content: float

    def to_dict(self):
        return asdict(self)


def parameter_fisher(
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

    modes = list(coeffs.keys())
    if not modes:
        return None
    values = {m: float(coeffs[m].detach() if isinstance(coeffs[m], torch.Tensor) else coeffs[m])
              for m in modes}
    cfg = MicroscopyPSFConfig(
        wavelength_nm=wavelength_nm, na=na, pixel_size_nm=pixel_size_nm,
        kernel_size=kernel_size, aberrations_waves=values,
    )
    device = structure.device
    psf = make_psf(cfg, device=device, dtype=torch.float64)
    derivs = psf_aberration_derivatives(cfg, tuple(modes), device=device, dtype=torch.float64)

    x = structure.to(torch.float64).clamp(0.0, 1.0)
    if x.ndim == 2:
        x = x[None, None]
    elif x.ndim == 3:
        x = x[None]
    kh, kw = psf.shape
    weight = psf[None, None]
    hx = tF.conv2d(x, weight, padding=(kh // 2, kw // 2))
    mask = (hx > 0).to(torch.float64)
    lam = (hx.clamp_min(0.0) * float(signal_photons) + float(background_photons)).clamp_min(1e-6)

    k = len(modes)
    dmu = []
    for m in modes:
        dh = derivs[m][None, None]
        d = tF.conv2d(x, dh, padding=(kh // 2, kw // 2)) * mask * float(signal_photons)
        dmu.append(d[0, 0].cpu().numpy())
    lam_np = lam[0, 0].cpu().numpy()
    Fm = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            Fm[i, j] = Fm[j, i] = float(np.sum(dmu[i] * dmu[j] / lam_np))
    return Fm


def parameter_fisher_multiframe(
    coeffs: Dict[str, torch.Tensor],
    structure: torch.Tensor,
    *,
    defocus_offsets: List[float],
    wavelength_nm: float,
    na: float,
    pixel_size_nm: float,
    signal_photons: float,
    background_photons: float,
    read_noise_std: float,
    kernel_size: int = 41,
) -> np.ndarray | None:

    total = None
    for off in defocus_offsets:
        th = dict(coeffs)
        th["defocus"] = th["defocus"] + float(off)
        F = parameter_fisher(
            th, structure, wavelength_nm=wavelength_nm, na=na,
            pixel_size_nm=pixel_size_nm, signal_photons=signal_photons,
            background_photons=background_photons, read_noise_std=read_noise_std,
            kernel_size=kernel_size,
        )
        if F is None:
            return None
        total = F if total is None else total + F
    return total


def crb_identifiable(fisher: np.ndarray | None, modes: List[str],
                     crb_max: float = 0.0025) -> tuple[Dict[str, float], int]:

    if fisher is None:
        return {}, 0
    w, V = np.linalg.eigh(fisher)
    keep = w > 1.0 / max(crb_max, 1e-12)
    n_degenerate = int((~keep).sum())
    inv = (V[:, keep] * (1.0 / np.maximum(w[keep], 1e-300))) @ V[:, keep].T
    return {m: float(np.diag(inv)[i]) for i, m in enumerate(modes)}, n_degenerate


def content_identifiability(
    psf: torch.Tensor,
    image_shape: tuple[int, int],
    gamma: float = 0.05,
) -> dict:

    otf = psf_to_otf(psf, image_shape)
    s = otf.abs().flatten()
    s = s / s.max().clamp_min(1e-12)
    c_gamma = (s >= gamma).float().mean().item()
    p = s / s.sum().clamp_min(1e-12)
    entropy = -(p * torch.log(p.clamp_min(1e-12))).sum()
    eff_rank = torch.exp(entropy).item() / s.numel()
    return {"c_gamma": float(c_gamma), "effective_rank_fraction": float(eff_rank),
            "null_fraction": float((s < gamma).float().mean().item())}


def information_budget(
    coeffs: Dict[str, torch.Tensor],
    structure: torch.Tensor,
    *,
    wavelength_nm: float,
    na: float,
    pixel_size_nm: float,
    signal_photons: float,
    background_photons: float,
    read_noise_std: float,
    gamma: float = 0.05,
    kernel_size: int = 41,
) -> InformationBudget:

    modes = list(coeffs.keys())
    F = parameter_fisher(coeffs, structure, wavelength_nm=wavelength_nm, na=na,
                         pixel_size_nm=pixel_size_nm, signal_photons=signal_photons,
                         background_photons=background_photons,
                         read_noise_std=read_noise_std, kernel_size=kernel_size)
    crb, n_degen = crb_identifiable(F, modes)
    psf = make_psf(MicroscopyPSFConfig(
        wavelength_nm=wavelength_nm, na=na, pixel_size_nm=pixel_size_nm,
        kernel_size=kernel_size, aberrations_waves=coeffs,
    ), device=structure.device)
    content = content_identifiability(psf, tuple(structure.shape[-2:]), gamma=gamma)


    otf = psf_to_otf(psf, tuple(structure.shape[-2:]))
    content_fisher_diag_mean = float(
        (otf.abs() ** 2 * signal_photons).mean().item()
    )
    crb_norm = float(np.sqrt(sum(v**2 for v in crb.values()))) if crb else float("nan")


    return InformationBudget(
        param_crb=crb,
        param_fisher_eigs=[float(e) for e in np.linalg.eigvalsh(F)] if F is not None else [],
        n_degenerate_directions=n_degen,
        content_c_gamma=content["c_gamma"],
        content_eff_rank_fraction=content["effective_rank_fraction"],
        content_fisher_diag_mean=content_fisher_diag_mean,
        budget_ratio_param_vs_content=crb_norm / max(content_fisher_diag_mean, 1e-12),
    )
