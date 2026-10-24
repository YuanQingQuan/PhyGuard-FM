from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import torch

from phyguard.optics.zernike_ansi import pupil_grid_exact


@dataclass
class MicroscopyPSFConfig:


    wavelength_nm: float = 520.0
    na: float = 1.2
    pixel_size_nm: float = 100.0
    kernel_size: int = 65
    aberrations_waves: Dict[str, float] = field(default_factory=dict)


    n_obj: float = 1.0
    axial_offset_um: float = 0.0


    pupil_grid: str = "fftfreq"


def _axial_phase_waves(rho: torch.Tensor, aperture: torch.Tensor,
                       cfg: MicroscopyPSFConfig) -> torch.Tensor:

    n = max(float(cfg.n_obj), float(cfg.na) * 1.0001)
    z_nm = float(cfg.axial_offset_um) * 1000.0
    inside = (n**2 - (float(cfg.na) * rho) ** 2).clamp_min(0.0)
    w = (z_nm / float(cfg.wavelength_nm)) * (torch.sqrt(inside) - n)
    piston = (w * aperture).sum() / aperture.sum().clamp_min(1.0)
    return (w - piston) * aperture


def _zernike_phase(rho: torch.Tensor, theta: torch.Tensor, coeffs: Dict[str, float]) -> torch.Tensor:
    phase_waves = torch.zeros_like(rho)


    modes = {
        "defocus": 2.0 * rho**2 - 1.0,
        "astig_x": rho**2 * torch.cos(2.0 * theta),
        "astig_y": rho**2 * torch.sin(2.0 * theta),
        "coma_x": (3.0 * rho**3 - 2.0 * rho) * torch.cos(theta),
        "coma_y": (3.0 * rho**3 - 2.0 * rho) * torch.sin(theta),
        "trefoil_x": rho**3 * torch.cos(3.0 * theta),
        "trefoil_y": rho**3 * torch.sin(3.0 * theta),
        "spherical": 6.0 * rho**4 - 6.0 * rho**2 + 1.0,


        "quad4_x": rho**4 * torch.cos(4.0 * theta),
        "quad4_y": rho**4 * torch.sin(4.0 * theta),
        "astig2_x": (4.0 * rho**4 - 3.0 * rho**2) * torch.cos(2.0 * theta),
        "astig2_y": (4.0 * rho**4 - 3.0 * rho**2) * torch.sin(2.0 * theta),
        "penta5_x": rho**5 * torch.cos(5.0 * theta),
        "penta5_y": rho**5 * torch.sin(5.0 * theta),
        "tref3_x": (5.0 * rho**5 - 4.0 * rho**3) * torch.cos(3.0 * theta),
        "tref3_y": (5.0 * rho**5 - 4.0 * rho**3) * torch.sin(3.0 * theta),
        "coma2_x": (10.0 * rho**5 - 12.0 * rho**3 + 3.0 * rho) * torch.cos(theta),
        "coma2_y": (10.0 * rho**5 - 12.0 * rho**3 + 3.0 * rho) * torch.sin(theta),
    }
    for name, value in coeffs.items():
        if name not in modes:
            raise ValueError(f"Unsupported aberration mode: {name}")


        c = (
            value
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(value, dtype=phase_waves.dtype, device=phase_waves.device)
        )
        phase_waves = phase_waves + c * modes[name]
    return 2.0 * torch.pi * phase_waves


def _pupil_coords(
    cfg: MicroscopyPSFConfig,
    n: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:

    if cfg.pupil_grid == "exact":


        rho, theta, aperture = pupil_grid_exact(
            n,
            pixel_size_um=cfg.pixel_size_nm / 1000.0,
            wavelength_um=cfg.wavelength_nm / 1000.0,
            na=cfg.na,
            dtype=dtype,
            device=device,
        )
        return rho, theta, aperture, True


    fy = torch.fft.fftfreq(n, d=cfg.pixel_size_nm, device=device, dtype=dtype)
    fx = torch.fft.fftfreq(n, d=cfg.pixel_size_nm, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    cutoff = float(cfg.na) / float(cfg.wavelength_nm)
    rho = torch.sqrt(xx**2 + yy**2) / max(cutoff, 1e-12)
    theta = torch.atan2(yy, xx)
    return rho, theta, (rho <= 1.0).to(dtype), False


def make_psf(
    cfg: MicroscopyPSFConfig,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:


    n = int(cfg.kernel_size)
    if cfg.pupil_grid not in ("fftfreq", "exact"):
        raise ValueError(f"unsupported pupil_grid: {cfg.pupil_grid!r}")
    if cfg.pupil_grid == "fftfreq" and n % 2 == 0:
        raise ValueError("kernel_size must be odd on the fftfreq grid")
    if cfg.na <= 0 or cfg.wavelength_nm <= 0 or cfg.pixel_size_nm <= 0:
        raise ValueError("NA, wavelength and pixel size must be positive")

    rho, theta, aperture, centred_pupil = _pupil_coords(cfg, n, device=device, dtype=dtype)

    phase = _zernike_phase(rho, theta, cfg.aberrations_waves)
    if cfg.axial_offset_um != 0.0:
        phase = phase + 2.0 * torch.pi * _axial_phase_waves(rho, aperture, cfg).to(phase.dtype)
    pupil = aperture.to(torch.complex64) * torch.exp(1j * phase.to(torch.complex64))

    if centred_pupil:
        amp = torch.fft.ifft2(torch.fft.ifftshift(pupil))
    else:
        amp = torch.fft.ifft2(pupil)
    intensity = torch.fft.fftshift(torch.abs(amp) ** 2)
    intensity = intensity / intensity.sum().clamp_min(1e-12)
    return intensity.to(dtype)


_ZERNIKE_MODE_NAMES = ("defocus", "astig_x", "astig_y", "coma_x", "coma_y",
    "trefoil_x", "trefoil_y", "spherical",
    "quad4_x", "quad4_y", "astig2_x", "astig2_y",
    "penta5_x", "penta5_y", "tref3_x", "tref3_y", "coma2_x", "coma2_y")


def _zernike_basis(rho: torch.Tensor, theta: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "defocus": 2.0 * rho**2 - 1.0,
        "astig_x": rho**2 * torch.cos(2.0 * theta),
        "astig_y": rho**2 * torch.sin(2.0 * theta),
        "coma_x": (3.0 * rho**3 - 2.0 * rho) * torch.cos(theta),
        "coma_y": (3.0 * rho**3 - 2.0 * rho) * torch.sin(theta),
        "trefoil_x": rho**3 * torch.cos(3.0 * theta),
        "trefoil_y": rho**3 * torch.sin(3.0 * theta),
        "spherical": 6.0 * rho**4 - 6.0 * rho**2 + 1.0,
        "quad4_x": rho**4 * torch.cos(4.0 * theta),
        "quad4_y": rho**4 * torch.sin(4.0 * theta),
        "astig2_x": (4.0 * rho**4 - 3.0 * rho**2) * torch.cos(2.0 * theta),
        "astig2_y": (4.0 * rho**4 - 3.0 * rho**2) * torch.sin(2.0 * theta),
        "penta5_x": rho**5 * torch.cos(5.0 * theta),
        "penta5_y": rho**5 * torch.sin(5.0 * theta),
        "tref3_x": (5.0 * rho**5 - 4.0 * rho**3) * torch.cos(3.0 * theta),
        "tref3_y": (5.0 * rho**5 - 4.0 * rho**3) * torch.sin(3.0 * theta),
        "coma2_x": (10.0 * rho**5 - 12.0 * rho**3 + 3.0 * rho) * torch.cos(theta),
        "coma2_y": (10.0 * rho**5 - 12.0 * rho**3 + 3.0 * rho) * torch.sin(theta),
    }


def psf_aberration_derivatives(
    cfg: MicroscopyPSFConfig,
    modes: tuple[str, ...] = _ZERNIKE_MODE_NAMES,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Dict[str, torch.Tensor]:

    for m in modes:
        if m not in _ZERNIKE_MODE_NAMES:
            raise ValueError(f"Unsupported aberration mode: {m}")
    n = int(cfg.kernel_size)
    rho, theta, aperture, centred = _pupil_coords(cfg, n, device=device, dtype=dtype)
    cplx = torch.complex128 if dtype == torch.float64 else torch.complex64
    phase = _zernike_phase(rho, theta, cfg.aberrations_waves).to(dtype)
    pupil = aperture.to(cplx) * torch.exp(1j * phase.to(cplx))


    to_fft_order = torch.fft.ifftshift if centred else (lambda t: t)
    amp = torch.fft.ifft2(to_fft_order(pupil))
    raw = torch.abs(amp) ** 2
    s = raw.sum().clamp_min(1e-300)
    basis = _zernike_basis(rho, theta)
    out = {}
    for m in modes:
        modulated = (1j * 2.0 * torch.pi * basis[m]).to(cplx) * pupil
        damp = torch.fft.ifft2(to_fft_order(modulated))
        draw = 2.0 * torch.real(torch.conj(amp) * damp)
        dh = (draw * s - raw * draw.sum()) / s**2
        dh = torch.fft.fftshift(dh)
        out[m] = dh.to(dtype)
    return out


def psf_to_otf(psf: torch.Tensor, output_shape: tuple[int, int] | None = None) -> torch.Tensor:

    if psf.ndim != 2:
        raise ValueError("psf must be 2-D")
    if output_shape is None:
        output_shape = tuple(psf.shape)
    h, w = output_shape
    kh, kw = psf.shape
    if kh > h or kw > w:
        raise ValueError("PSF cannot be larger than requested OTF grid")
    pad = torch.zeros((h, w), dtype=psf.dtype, device=psf.device)


    y0 = h // 2 - kh // 2
    x0 = w // 2 - kw // 2
    pad[y0 : y0 + kh, x0 : x0 + kw] = psf
    pad = torch.fft.ifftshift(pad)
    return torch.fft.fft2(pad)
