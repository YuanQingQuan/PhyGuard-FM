import torch
from phyguard.optics.psf import MicroscopyPSFConfig, make_psf
from phyguard.optics.forward import PhotonMicroscope
from phyguard.identifiability import convolution_identifiability
from phyguard.verification import measurement_support_map


def test_psf_normalized():
    p=make_psf(MicroscopyPSFConfig(kernel_size=33))
    assert p.shape==(33,33)
    assert torch.isclose(p.sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.all(p>=0)


def test_forward_shape_and_counts():
    p=make_psf(MicroscopyPSFConfig(kernel_size=33))
    m=PhotonMicroscope(p, signal_photons=5, background_photons=2)
    x=torch.zeros(1,1,64,64); x[:,:,32,32]=1
    y=m.expected_counts(x)
    assert y.shape==x.shape
    assert y.min()>=2


def test_identifiability_range():
    p=make_psf(MicroscopyPSFConfig(kernel_size=33))
    r=convolution_identifiability(p,(64,64),0.05)
    assert 0<=r.c_gamma<=1
    assert 0<=r.null_fraction<=1


def test_support_prefers_matching_candidate():
    p=make_psf(MicroscopyPSFConfig(kernel_size=33))
    m=PhotonMicroscope(p, signal_photons=100, background_photons=1)
    x=torch.zeros(1,1,64,64); x[:,:,24:40,24:40]=1
    obs=m.expected_counts(x)
    good=measurement_support_map(obs,x,m).mean()
    bad=measurement_support_map(obs,torch.zeros_like(x),m).mean()
    assert good>bad
