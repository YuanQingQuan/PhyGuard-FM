from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepImagePrior(nn.Module):


    def __init__(self, num_input_channels: int = 32, num_output_channels: int = 1,
                 num_channels_down: int = 64, num_channels_up: int = 64,
                 num_channels_skip: int = 32):
        super().__init__()
        self.head = self._conv_block(num_input_channels, num_channels_up)

        self.enc1 = self._conv_block(num_channels_up, num_channels_up)
        self.enc2 = self._conv_block(num_channels_up, num_channels_down)
        self.enc3 = self._conv_block(num_channels_down, num_channels_down)
        self.dec3 = self._conv_block(num_channels_down + num_channels_skip, num_channels_down)
        self.dec2 = self._conv_block(num_channels_down + num_channels_skip, num_channels_up)
        self.dec1 = self._conv_block(num_channels_up + num_channels_skip, num_channels_up)
        self.tail = nn.Conv2d(num_channels_up, num_output_channels, 1)
        self.skip = nn.Conv2d(num_channels_up, num_channels_skip, 1)

    @staticmethod
    def _conv_block(cin: int, cout: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.head(z)
        s1 = self.skip(x)
        x1 = F.max_pool2d(x, 2)
        x1 = self.enc1(x1)
        s2 = self.skip(x1)
        x2 = F.max_pool2d(x1, 2)
        x2 = self.enc2(x2)
        s3 = self.skip(x2)
        x3 = F.max_pool2d(x2, 2)
        x3 = self.enc3(x3)
        u = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        u = torch.cat([u, s3], dim=1)
        u = self.dec3(u)
        u = F.interpolate(u, scale_factor=2, mode="bilinear", align_corners=False)
        u = torch.cat([u, s2], dim=1)
        u = self.dec2(u)
        u = F.interpolate(u, scale_factor=2, mode="bilinear", align_corners=False)
        u = torch.cat([u, s1], dim=1)
        u = self.dec1(u)
        return torch.sigmoid(self.tail(u))


def make_dip_noise(h: int, w: int, channels: int = 32, device: torch.device = None,
                   seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.randn((1, channels, h, w), generator=g, device=device)


def dip_blind_estimate(
    counts: torch.Tensor,
    *,
    modes: List[str],
    wavelength_nm: float,
    na: float,
    pixel_size_nm: float,
    signal_photons: float,
    background_photons: float,
    read_noise_std: float,
    n_steps: int = 800,
    lr_net: float = 0.01,
    lr_theta: float = 0.1,
    seed: int = 0,
    n_restarts: int = 3,
    defocus_offsets: List[float] | None = None,
    h: int | None = None,
    w: int | None = None,
) -> dict:

    from ..optics.psf import MicroscopyPSFConfig, make_psf
    from ..optics.forward import PhotonMicroscope
    from ..likelihood import poisson_gaussian_nll

    h = h or counts.shape[-2]
    w = w or counts.shape[-1]
    offsets = defocus_offsets or [0.0]
    multi = len(offsets) > 1
    rng = torch.Generator()
    rng.manual_seed(seed)
    best = None
    for r in range(n_restarts):
        net = DeepImagePrior().to(counts.device)
        net.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)
        z = make_dip_noise(h, w, device=counts.device, seed=seed + r)
        theta = {m: torch.tensor(
            float(torch.rand(1, generator=rng).item() * 0.4 - 0.2) if r > 0 else 0.0,
            requires_grad=True, device=counts.device,
        ) for m in modes}
        opt_net = torch.optim.Adam(net.parameters(), lr=lr_net)
        opt_th = torch.optim.Adam(list(theta.values()), lr=lr_theta)
        history = []

        def nll(x_in):
            tot = 0.0
            for i, off in enumerate(offsets):
                th = dict(theta)
                th["defocus"] = th["defocus"] + off
                psf = make_psf(MicroscopyPSFConfig(
                    wavelength_nm=wavelength_nm, na=na, pixel_size_nm=pixel_size_nm,
                    kernel_size=41, aberrations_waves=th,
                ), device=counts.device)
                mic = PhotonMicroscope(psf=psf, signal_photons=signal_photons,
                                       background_photons=background_photons,
                                       read_noise_std=read_noise_std)
                c = counts[i] if multi else counts
                tot = tot + poisson_gaussian_nll(c, mic.expected_counts(x_in), read_noise_std)
            return tot / len(offsets)

        for step in range(n_steps):
            opt_net.zero_grad()
            x = net(z)
            loss = nll(x)
            loss.backward()
            opt_net.step()
            opt_th.zero_grad()
            loss = nll(net(z).detach())
            loss.backward()
            opt_th.step()
            if step % 50 == 0:
                history.append(float(loss.detach()))

        with torch.no_grad():
            x_out = net(z)[0, 0].clamp(0, 1).detach().cpu().numpy()
        final = float(nll(net(z)).detach())
        est = {m: float(theta[m].detach()) for m in modes}
        if best is None or final < best[0]:
            best = (final, est, x_out, history)
    return {"theta": best[1], "x": best[2], "loss": best[0], "loss_history": best[3]}


def blind_rl_estimate(
    counts: torch.Tensor,
    *,
    modes: List[str],
    wavelength_nm: float,
    na: float,
    pixel_size_nm: float,
    signal_photons: float,
    background_photons: float,
    read_noise_std: float,
    n_outer: int = 25,
    rl_iters: int = 15,
    lr_theta: float = 0.1,
    n_theta_steps: int = 10,
    seed: int = 0,
    defocus_offsets: List[float] | None = None,
) -> dict:

    from ..optics.psf import MicroscopyPSFConfig, make_psf
    from ..optics.forward import PhotonMicroscope
    from ..likelihood import poisson_gaussian_nll
    from skimage.restoration import richardson_lucy

    offsets = defocus_offsets or [0.0]
    multi = len(offsets) > 1
    rng = np.random.default_rng(seed)
    best = None
    for r in range(3):
        theta = {m: torch.tensor(float(rng.uniform(-0.2, 0.2)) if r > 0 else 0.0, requires_grad=True) for m in modes}

        mid = len(offsets) // 2
        c0 = counts[mid] if multi else counts
        obs_img = torch.clamp((c0 - background_photons) / signal_photons, 0, 1)
        obs_img = obs_img[0, 0].cpu().numpy().astype(np.float32)
        nom = make_psf(MicroscopyPSFConfig(
            wavelength_nm=wavelength_nm, na=na, pixel_size_nm=pixel_size_nm, kernel_size=41,
        )).detach().cpu().numpy()
        x_np = richardson_lucy(np.clip(obs_img, 0, 1), nom, num_iter=rl_iters, clip=False)
        history = []
        for _ in range(n_outer):

            opt_t = torch.optim.Adam(list(theta.values()), lr=lr_theta)
            x_t = torch.tensor(np.clip(x_np, 0, 1), device=counts.device)[None, None]
            for _ in range(n_theta_steps):
                opt_t.zero_grad()
                loss = 0.0
                for i, off in enumerate(offsets):
                    th = dict(theta); th["defocus"] = th["defocus"] + off
                    psf = make_psf(MicroscopyPSFConfig(
                        wavelength_nm=wavelength_nm, na=na, pixel_size_nm=pixel_size_nm,
                        kernel_size=41, aberrations_waves=th,
                    ))
                    mic = PhotonMicroscope(psf=psf, signal_photons=signal_photons,
                                           background_photons=background_photons,
                                           read_noise_std=read_noise_std)
                    c = counts[i] if multi else counts
                    loss = loss + poisson_gaussian_nll(c, mic.expected_counts(x_t), read_noise_std)
                loss.backward()
                opt_t.step()

            th0 = {m: float(theta[m].detach()) for m in modes}
            psf0 = make_psf(MicroscopyPSFConfig(
                wavelength_nm=wavelength_nm, na=na, pixel_size_nm=pixel_size_nm,
                kernel_size=41, aberrations_waves=th0,
            )).detach().cpu().numpy()
            x_np = richardson_lucy(np.clip(obs_img, 0, 1), psf0, num_iter=rl_iters, clip=False)
            history.append(float(loss.detach()))
        final = history[-1]
        est = {m: float(theta[m].detach()) for m in modes}
        x_est = np.clip(x_np, 0, 1)
        if best is None or final < best[0]:
            best = (final, est, x_est, history)
    return {"theta": best[1], "x": best[2], "loss": best[0], "loss_history": best[3]}
