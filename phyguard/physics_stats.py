from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class BlindPhysicsStats:
    background_level: float
    high_frequency_noise_std: float
    snr_proxy_db: float
    high_frequency_energy_ratio: float

    def to_dict(self):
        return asdict(self)


def estimate_blind_stats(image: torch.Tensor, background_quantile: float = 0.1) -> BlindPhysicsStats:

    x = image.detach().float()
    if x.ndim > 2:
        x = x.squeeze()
    x = x.clamp_min(0)
    flat = x.flatten()
    background = torch.quantile(flat, background_quantile).item()


    xx = x[None, None]
    low = F.avg_pool2d(xx, 3, stride=1, padding=1)[0, 0]
    hp = x - low
    mad = torch.median(torch.abs(hp - torch.median(hp)))
    sigma = (mad / 0.67448975).item()

    centered = x - x.mean()
    spec = torch.fft.fftshift(torch.fft.fft2(centered))
    power = spec.abs() ** 2
    h, w = power.shape
    yy, xxg = torch.meshgrid(
        torch.linspace(-1, 1, h, device=x.device),
        torch.linspace(-1, 1, w, device=x.device),
        indexing="ij",
    )
    rr = torch.sqrt(xxg**2 + yy**2)
    hf = power[rr >= 0.65].mean().item()
    total = power.mean().item() + 1e-12
    hf_ratio = hf / total

    signal_std = torch.std(torch.clamp(x - background, min=0)).item()
    snr_db = 20.0 * np.log10((signal_std + 1e-12) / (sigma + 1e-12))
    return BlindPhysicsStats(background, sigma, float(snr_db), float(hf_ratio))
