from __future__ import annotations

import numpy as np
import torch

from phyguard.optics.psf import MicroscopyPSFConfig, make_psf, psf_to_otf
from phyguard.optics.forward import PhotonMicroscope
from phyguard.likelihood import poisson_gaussian_nll
from phyguard.estimation.info import parameter_fisher
from phyguard.identifiability import convolution_identifiability


def test_psf_normalized():
    p = make_psf(MicroscopyPSFConfig(kernel_size=33))
    assert p.shape == (33, 33)
    assert torch.isclose(p.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.all(p >= 0)


def test_forward_photon_conservation():
    torch.manual_seed(0)
    psf = make_psf(MicroscopyPSFConfig(kernel_size=33))
    x = torch.rand(1, 1, 96, 96)
    mic = PhotonMicroscope(psf, signal_photons=20.0, background_photons=1.0)
    lam = mic.expected_counts(x)
    expected = 20.0 * float(x.mean()) + 1.0

    m = 20
    interior = lam[..., m:-m, m:-m].mean()
    assert abs(float(interior) - expected) / expected < 0.05


def test_otf_alignment_circular_matches_convolution():

    torch.manual_seed(1)
    psf = make_psf(MicroscopyPSFConfig(kernel_size=41))
    x = torch.rand(1, 1, 256, 256)
    mic = PhotonMicroscope(psf, signal_photons=1.0, background_photons=0.0)
    y_zero = mic._conv(x)
    otf = psf_to_otf(psf, (256, 256))
    y_circ = torch.fft.ifft2(torch.fft.fft2(x) * otf).real
    d = (y_zero[0, 0] - y_circ[0, 0]).abs()
    m = 25
    assert float(d[m:-m, m:-m].mean()) < 1e-4


def test_nll_minimized_at_truth_or_conjugate():
    torch.manual_seed(2)
    true_d = 0.25
    psf_true = make_psf(MicroscopyPSFConfig(kernel_size=41, aberrations_waves={"defocus": true_d}))
    gt = torch.rand(1, 1, 96, 96)
    mic = PhotonMicroscope(psf_true, signal_photons=50.0, background_photons=2.0, read_noise_std=0.1)
    counts = mic.sample(gt)["counts"]

    def nll_at(d):
        psf = make_psf(MicroscopyPSFConfig(kernel_size=41, aberrations_waves={"defocus": d}))
        mic2 = PhotonMicroscope(psf, signal_photons=50.0, background_photons=2.0, read_noise_std=0.1)
        return float(poisson_gaussian_nll(counts, mic2.expected_counts(gt), 0.1))


    assert abs(nll_at(true_d) - nll_at(-true_d)) < 1e-4
    assert nll_at(true_d) <= min(nll_at(0.0), nll_at(0.1), nll_at(-0.1))


def test_fisher_symmetric_and_near_psd():
    torch.manual_seed(3)
    structure = torch.rand(1, 1, 96, 96)
    coeffs = {m: torch.tensor(0.1, requires_grad=True) for m in ["defocus", "astig_x"]}
    F = parameter_fisher(coeffs, structure, wavelength_nm=520.0, na=1.2,
                         pixel_size_nm=100.0, signal_photons=10.0,
                         background_photons=1.5, read_noise_std=0.15)
    assert np.allclose(F, F.T)
    eig = np.linalg.eigvalsh(F)

    assert eig.min() > -abs(eig.max()) * 1e-5


def test_fisher_matches_finite_differences():

    torch.manual_seed(4)
    modes = ["defocus", "coma_x"]
    true = {"defocus": -0.11, "coma_x": 0.17}
    structure = torch.rand(1, 1, 96, 96, dtype=torch.float64)
    photons, bg = 50.0, 7.5

    def lam_of(vals):
        c = {m: torch.as_tensor(vals[m], dtype=torch.float64) for m in vals}
        psf = make_psf(MicroscopyPSFConfig(
            wavelength_nm=520.0, na=1.2, pixel_size_nm=100.0, kernel_size=33,
            aberrations_waves=c), dtype=torch.float64)
        mic = PhotonMicroscope(psf=psf, signal_photons=photons, background_photons=bg)
        return mic.expected_counts(structure).detach()

    coeffs = {m: torch.tensor(true[m]) for m in modes}
    F = parameter_fisher(coeffs, structure, wavelength_nm=520.0, na=1.2,
                         pixel_size_nm=100.0, signal_photons=photons,
                         background_photons=bg, read_noise_std=0.0, kernel_size=33)
    lam_t = lam_of(true)
    lam_np = lam_t[0, 0].clamp_min(1e-6).numpy()
    for i, m in enumerate(modes):
        pert = dict(true); pert[m] = true[m] + 1e-4
        fd = ((lam_of(pert) - lam_t) / 1e-4)[0, 0].numpy()
        fd_f = float(np.sum(fd**2 / lam_np))
        assert abs(F[i, i] / fd_f - 1.0) < 0.02, (m, F[i, i], fd_f)


def test_fisher_kl_curvature_consistency():

    torch.manual_seed(5)
    modes = ["defocus", "astig_x", "coma_x"]
    true = {"defocus": -0.08, "astig_x": 0.05, "coma_x": 0.12}
    structure = torch.rand(1, 1, 96, 96, dtype=torch.float64)
    photons, bg = 50.0, 7.5

    def lam_of(vals):
        c = {m: torch.as_tensor(vals[m], dtype=torch.float64) for m in vals}
        psf = make_psf(MicroscopyPSFConfig(
            wavelength_nm=520.0, na=1.2, pixel_size_nm=100.0, kernel_size=33,
            aberrations_waves=c), dtype=torch.float64)
        mic = PhotonMicroscope(psf=psf, signal_photons=photons, background_photons=bg)
        return mic.expected_counts(structure).detach()

    def kl_of(vals):
        lm, lt = lam_of(vals), lam_of(true)
        lm = lm.clamp_min(1e-9)
        return float((lt * (lt.clamp_min(1e-9) / lm).log() - (lt - lm)).sum())

    coeffs = {m: torch.tensor(true[m]) for m in modes}
    F = parameter_fisher(coeffs, structure, wavelength_nm=520.0, na=1.2,
                         pixel_size_nm=100.0, signal_photons=photons,
                         background_photons=bg, read_noise_std=0.0, kernel_size=33)
    w, V = np.linalg.eigh(F)
    t0 = np.array([true[m] for m in modes])
    for idx in [0, len(modes) - 1]:
        v = V[:, idx]
        d = 1e-3
        kl = kl_of({m: float(t0[i] + d * v[i]) for i, m in enumerate(modes)})
        quad = 0.5 * d**2 * w[idx]
        assert abs(kl / quad - 1.0) < 0.05, (idx, kl, quad)


def test_c_gamma_matches_direct_otf():
    psf = make_psf(MicroscopyPSFConfig(kernel_size=41))
    shape = (128, 128)
    report = convolution_identifiability(psf, shape, gamma=0.05)
    otf = psf_to_otf(psf, shape)
    s = otf.abs().flatten()
    s = s / s.max()
    direct = float((s >= 0.05).float().mean())
    assert abs(report.c_gamma - direct) < 1e-6
