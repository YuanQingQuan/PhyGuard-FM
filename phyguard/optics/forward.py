from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class PhotonMicroscope:


    psf: torch.Tensor
    signal_photons: float = 20.0
    background_photons: float = 1.0
    read_noise_std: float = 0.0
    downsample: int = 1

    def __post_init__(self) -> None:
        if self.psf.ndim != 2:
            raise ValueError("psf must be a 2-D tensor")
        if self.signal_photons <= 0:
            raise ValueError("signal_photons must be > 0")
        if self.background_photons < 0 or self.read_noise_std < 0:
            raise ValueError("background/read noise must be nonnegative")
        if int(self.downsample) < 1:
            raise ValueError("downsample must be >= 1")

    def _conv(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x[None, None]
        elif x.ndim == 3:
            x = x[None]
        if x.ndim != 4:
            raise ValueError("x must be HxW, CxHxW or BxCxHxW")
        b, c, _, _ = x.shape
        k = self.psf.to(device=x.device, dtype=x.dtype)
        kh, kw = k.shape
        weight = k[None, None].repeat(c, 1, 1, 1)
        y = F.conv2d(x, weight, padding=(kh // 2, kw // 2), groups=c)
        if self.downsample > 1:
            y = F.avg_pool2d(y, kernel_size=self.downsample, stride=self.downsample)
        return y

    def expected_counts(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0.0, 1.0)
        hx = self._conv(x).clamp_min(0.0)
        return hx * float(self.signal_photons) + float(self.background_photons)

    def sample(self, x: torch.Tensor, *, generator: torch.Generator | None = None) -> Dict[str, torch.Tensor]:
        lam = self.expected_counts(x)

        if generator is not None:
            state = torch.random.get_rng_state()
            torch.random.set_rng_state(generator.get_state())
        counts = torch.poisson(lam)
        if generator is not None:
            generator.set_state(torch.random.get_rng_state())
            torch.random.set_rng_state(state)
        if self.read_noise_std > 0:
            counts = counts + torch.randn_like(counts) * float(self.read_noise_std)
        norm = ((counts - float(self.background_photons)) / float(self.signal_photons)).clamp(0.0, 1.0)
        return {"expected_counts": lam, "counts": counts, "normalized": norm}
