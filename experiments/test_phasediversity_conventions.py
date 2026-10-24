from __future__ import annotations

import math

import numpy as np
import torch

from phyguard.optics.psf import (
    MicroscopyPSFConfig,
    _zernike_basis,
    _ZERNIKE_MODE_NAMES,
    make_psf,
)
from phyguard.optics.zernike_ansi import (
    ANSI_ESTIMATED,
    ANSI_INDEX,
    ANSI_NAME,
    LAMBDA_UM,
    pupil_max_radius,
    pupil_radius_samples,
    radial_polynomial,
)


def test_ansi_index_table_is_self_consistent() -> None:
    for j, (n, m) in ANSI_INDEX.items():
        assert (n * (n + 2) + m) == 2 * j, (j, n, m)
        assert abs(m) <= n and (n - abs(m)) % 2 == 0, (j, n, m)


def test_ansi_names_cover_modes_3_to_20_uniquely() -> None:
    names = [ANSI_NAME[j] for j in ANSI_ESTIMATED]
    assert len(names) == 18
    assert len(set(names)) == 18, "duplicate mode name: an |m|-keyed table aliases orders"


def test_psf_modes_match_analytic_radial_polynomials() -> None:
    torch.manual_seed(0)
    rho = torch.rand(4000, dtype=torch.float64) * 1.4
    th = torch.rand(4000, dtype=torch.float64) * 2 * math.pi - math.pi
    basis = _zernike_basis(rho, th)
    assert set(basis) == set(_ZERNIKE_MODE_NAMES)

    for j in ANSI_ESTIMATED:
        n, m = ANSI_INDEX[j]
        ref = radial_polynomial(n, m, rho)
        if m != 0:
            ref = ref * (torch.cos(abs(m) * th) if m > 0 else torch.sin(abs(m) * th))
        err = (basis[ANSI_NAME[j]] - ref).abs().max().item()
        assert err < 1e-12, (ANSI_NAME[j], n, m, err)


def test_matlab_pupil_radius_constants() -> None:

    assert abs(pupil_max_radius(512) - 2.1314) < 5e-5
    assert abs(pupil_max_radius(768) - 2.1314) < 5e-5
    assert abs(pupil_radius_samples(512) - 119.87) < 0.05
    assert abs(pupil_radius_samples(768) - 179.93) < 0.05


def _original_make_psf(cfg: MicroscopyPSFConfig, dtype=torch.float32) -> torch.Tensor:

    n = int(cfg.kernel_size)
    assert n % 2 == 1
    fy = torch.fft.fftfreq(n, d=cfg.pixel_size_nm, dtype=dtype)
    fx = torch.fft.fftfreq(n, d=cfg.pixel_size_nm, dtype=dtype)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    cutoff = float(cfg.na) / float(cfg.wavelength_nm)
    rho = torch.sqrt(xx**2 + yy**2) / cutoff
    theta = torch.atan2(yy, xx)
    aperture = (rho <= 1.0).to(dtype)
    modes = {
        "defocus": 2.0 * rho**2 - 1.0,
        "astig_x": rho**2 * torch.cos(2 * theta),
        "coma_x": (3.0 * rho**3 - 2.0 * rho) * torch.cos(theta),
        "spherical": 6.0 * rho**4 - 6.0 * rho**2 + 1.0,
    }
    phase = torch.zeros_like(rho)
    for k, v in cfg.aberrations_waves.items():
        phase = phase + v * modes[k]
    pupil = aperture.to(torch.complex64) * torch.exp(1j * (2.0 * math.pi * phase).to(torch.complex64))
    inten = torch.fft.fftshift(torch.abs(torch.fft.ifft2(pupil)) ** 2)
    return (inten / inten.sum()).to(dtype)


def test_default_pupil_grid_path_is_unchanged() -> None:

    for aberr in ({}, {"defocus": 0.3}, {"defocus": -0.5, "astig_x": 0.2}):
        cfg = MicroscopyPSFConfig(wavelength_nm=520.0, na=1.2, pixel_size_nm=62.6,
                                  kernel_size=41, aberrations_waves=aberr)
        new = make_psf(cfg)
        old = _original_make_psf(cfg)
        assert torch.equal(new, old), f"default path changed for {aberr}"


def test_exact_grid_psf_is_centred_and_normalised() -> None:
    cfg = MicroscopyPSFConfig(wavelength_nm=532.0, na=1.2, pixel_size_nm=104.0,
                              kernel_size=512, pupil_grid="exact",
                              aberrations_waves={"astig_x": 1.0 / 0.532})
    psf = make_psf(cfg)
    assert abs(psf.sum().item() - 1.0) < 1e-5


    n = psf.shape[0]
    yr = torch.arange(n, dtype=torch.float64)
    yy, xx = torch.meshgrid(yr, yr, indexing="ij")
    w = psf.double()
    assert abs((w * yy).sum().item() / w.sum().item() - n / 2.0) < 0.1
    assert abs((w * xx).sum().item() / w.sum().item() - n / 2.0) < 0.1


    n = psf.shape[0]
    rot180 = torch.roll(torch.flip(psf, dims=(0, 1)), shifts=(1, 1), dims=(0, 1))
    asym = (psf - rot180).abs().max() / psf.max()
    assert asym < 1e-5, f"PSF is not centrosymmetric about index N/2: {asym}"
    wrong_centre = (psf - torch.flip(psf, dims=(0, 1))).abs().max() / psf.max()
    assert wrong_centre > 0.1, (
        "the aperture-centre flip unexpectedly looks symmetric; the half-sample "
        "offset between pupil and PSF centres may have changed"
    )


    flat = torch.sort(psf.flatten(), descending=True).values
    assert abs(flat[0].item() - flat[1].item()) / flat[0].item() < 1e-5


def test_wavefront_rms_anchor_matches_published_value() -> None:

    try:
        from experiments.run_phasediversity_external import load_iteration, wavefront_rms
        from phyguard.optics.zernike_ansi import ansi_coeffs_to_waves
        _, coeffs, _ = load_iteration("cell", 1)
    except FileNotFoundError:
        return
    truth = ansi_coeffs_to_waves({k: float(v) for k, v in enumerate(coeffs[0], 1) if v != 0})
    rms_nm = wavefront_rms(truth)
    assert abs(rms_nm - 219.0) < 1.0, rms_nm


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:                                # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
