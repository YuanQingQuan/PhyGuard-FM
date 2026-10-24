from __future__ import annotations

import torch

from .psf import MicroscopyPSFConfig, _zernike_phase


def make_psf_3d(
    cfg: MicroscopyPSFConfig,
    *,
    nz: int,
    dz_nm: float,
    refractive_index: float = 1.518,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:

    n = int(cfg.kernel_size)
    if n % 2 == 0:
        raise ValueError("kernel_size must be odd")
    if cfg.na <= 0 or cfg.wavelength_nm <= 0 or cfg.pixel_size_nm <= 0:
        raise ValueError("NA, wavelength and pixel size must be positive")

    fy = torch.fft.fftfreq(n, d=cfg.pixel_size_nm, device=device, dtype=dtype)
    fx = torch.fft.fftfreq(n, d=cfg.pixel_size_nm, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    cutoff = float(cfg.na) / float(cfg.wavelength_nm)
    rr = torch.sqrt(xx**2 + yy**2)
    rho = rr / max(cutoff, 1e-12)
    theta = torch.atan2(yy, xx)
    aperture = (rho <= 1.0).to(dtype)
    phase = _zernike_phase(rho, theta, cfg.aberrations_waves)
    pupil = aperture.to(torch.complex64) * torch.exp(1j * phase.to(torch.complex64))


    zs = (torch.arange(nz, device=device, dtype=dtype) - nz // 2) * float(dz_nm)
    lam = float(cfg.wavelength_nm)

    rho_cyc = rr

    inside = float(refractive_index) ** 2 - (lam * rho_cyc) ** 2
    inside = inside.clamp_min(0.0)
    kz = (2.0 * torch.pi / lam) * torch.sqrt(inside)

    psf3 = torch.zeros((nz, n, n), dtype=dtype, device=device)
    for i, z in enumerate(zs):
        prop = torch.exp(1j * (kz * float(z)).to(torch.complex64))
        field = torch.fft.ifft2(pupil * prop)
        inten = torch.abs(field) ** 2
        inten = torch.fft.fftshift(inten)
        psf3[i] = inten / inten.sum().clamp_min(1e-12)
    return psf3


def forward_3d_stack(
    structure_3d: torch.Tensor,
    psf3d: torch.Tensor,
) -> torch.Tensor:

    if structure_3d.ndim != 3 or psf3d.ndim != 3:
        raise ValueError("structure_3d [nz,H,W] and psf3d [nz,kh,kw] required")
    import torch.nn.functional as F
    nz = structure_3d.shape[0]
    out = torch.zeros_like(structure_3d)
    for i in range(nz):
        psf = psf3d[i]
        kh, kw = psf.shape
        w = psf[None, None]
        out[i] = F.conv2d(structure_3d[i][None, None], w, padding=(kh // 2, kw // 2))[0, 0]
    return out
