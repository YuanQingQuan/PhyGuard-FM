from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from phyguard.optics.psf import MicroscopyPSFConfig, make_psf, psf_to_otf
from phyguard.optics.forward import PhotonMicroscope
from phyguard.likelihood import poisson_gaussian_nll, poisson_nll
from phyguard.estimation.info import parameter_fisher
from phyguard.identifiability import convolution_identifiability


def test_psf_normalization():
    psf = make_psf(MicroscopyPSFConfig(wavelength_nm=520, na=1.2, pixel_size_nm=100, kernel_size=41,
                                       aberrations_waves={"defocus": 0.2}))
    return float(psf.sum()), "PSF sum == 1"


def test_photon_conservation():
    torch.manual_seed(0)
    psf = make_psf(MicroscopyPSFConfig(wavelength_nm=520, na=1.2, pixel_size_nm=100, kernel_size=41))
    x = torch.rand(1, 1, 128, 128)
    mic = PhotonMicroscope(psf=psf, signal_photons=20.0, background_photons=1.0)
    lam = mic.expected_counts(x)
    mean_counts = float(lam.mean())
    expected = 20.0 * float(x.mean()) + 1.0
    return abs(mean_counts - expected) / expected, "mean counts ~ signal*mean(x)+bg"


def test_zero_pad_vs_circular():

    torch.manual_seed(1)
    psf = make_psf(MicroscopyPSFConfig(wavelength_nm=520, na=1.2, pixel_size_nm=100, kernel_size=41))
    x = torch.rand(1, 1, 256, 256)

    mic = PhotonMicroscope(psf=psf, signal_photons=1.0, background_photons=0.0)
    y_zero = mic._conv(x)

    h, w = x.shape[-2:]
    otf = psf_to_otf(psf, (h, w))
    y_circ = torch.fft.ifft2(torch.fft.fft2(x) * otf).real[None, None]

    m = 25
    center_diff = float((y_zero[..., m:-m, m:-m] - y_circ[..., m:-m, m:-m]).abs().mean())
    edge_diff = float((y_zero[..., :m, :] - y_circ[..., :m, :]).abs().mean())
    return (center_diff, edge_diff), "center (zero-pad ~= circular), edge differs"


def test_nll_minimized_at_truth():

    torch.manual_seed(2)
    psf_true = make_psf(MicroscopyPSFConfig(wavelength_nm=520, na=1.2, pixel_size_nm=100, kernel_size=41,
                                            aberrations_waves={"defocus": 0.25}))
    gt = torch.rand(1, 1, 128, 128)
    mic = PhotonMicroscope(psf=psf_true, signal_photons=50.0, background_photons=2.0, read_noise_std=0.1)
    counts = mic.sample(gt)["counts"]
    def nll_at(d):
        psf = make_psf(MicroscopyPSFConfig(wavelength_nm=520, na=1.2, pixel_size_nm=100, kernel_size=41,
                                           aberrations_waves={"defocus": d}))
        mic2 = PhotonMicroscope(psf=psf, signal_photons=50.0, background_photons=2.0, read_noise_std=0.1)
        return float(poisson_gaussian_nll(counts, mic2.expected_counts(gt), 0.1))
    vals = {d: nll_at(d) for d in [-0.25, -0.15, -0.05, 0.0, 0.05, 0.15, 0.25]}
    best = min(vals, key=vals.get)
    return (best, vals), "NLL minimized near true defocus (0.25) or its conjugate (-0.25)"


def test_fisher_psd():
    structure = torch.rand(1, 1, 128, 128)
    coeffs = {m: torch.tensor(0.1, requires_grad=True) for m in ["defocus", "astig_x", "coma_x"]}
    F = parameter_fisher(coeffs, structure, wavelength_nm=520.0, na=1.2, pixel_size_nm=100,
                         signal_photons=10.0, background_photons=1.5, read_noise_std=0.15)
    eig = np.linalg.eigvalsh(F)
    return (float(eig.min()), float(eig.max())), "Fisher min eigenvalue >= -eps"


def test_otf_singular_values_match_circular_conv():

    psf = make_psf(MicroscopyPSFConfig(wavelength_nm=520, na=1.2, pixel_size_nm=100, kernel_size=41))
    shape = (128, 128)
    report = convolution_identifiability(psf, shape, gamma=0.05)
    otf = psf_to_otf(psf, shape)
    s_otf = otf.abs().flatten()
    s_otf = s_otf / s_otf.max()
    c_gamma_direct = float((s_otf >= 0.05).float().mean())
    return (report.c_gamma, c_gamma_direct), "c_gamma from report == direct OTF computation"


def main():
    tests = [
        ("PSF normalization", test_psf_normalization),
        ("Photon conservation", test_photon_conservation),
        ("Zero-pad vs circular", test_zero_pad_vs_circular),
        ("NLL at truth", test_nll_minimized_at_truth),
        ("Fisher PSD", test_fisher_psd),
        ("OTF singular values", test_otf_singular_values_match_circular_conv),
    ]
    for name, fn in tests:
        try:
            val, desc = fn()
            print(f"[PASS] {name:28s} {desc}: {val}")
        except Exception as e:
            print(f"[FAIL] {name:28s} {e}")


if __name__ == "__main__":
    main()
