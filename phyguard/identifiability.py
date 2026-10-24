from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch

from .optics.psf import psf_to_otf


@dataclass
class IdentifiabilityReport:
    c_gamma: float
    effective_rank_fraction: float
    null_fraction: float
    condition_proxy: float
    gamma: float

    def to_dict(self):
        return asdict(self)


def convolution_identifiability(psf: torch.Tensor, image_shape: tuple[int, int], gamma: float = 0.05) -> IdentifiabilityReport:

    otf = psf_to_otf(psf, image_shape)
    s = otf.abs().flatten()
    s = s / s.max().clamp_min(1e-12)
    c_gamma = (s >= gamma).float().mean().item()
    null_fraction = (s < gamma).float().mean().item()

    p = s / s.sum().clamp_min(1e-12)
    entropy = -(p * torch.log(p.clamp_min(1e-12))).sum()
    eff_rank = torch.exp(entropy).item() / s.numel()

    supported = s[s >= gamma]
    condition = (supported.max() / supported.min().clamp_min(1e-12)).item() if supported.numel() else float("inf")
    return IdentifiabilityReport(float(c_gamma), float(eff_rank), float(null_fraction), float(condition), float(gamma))
