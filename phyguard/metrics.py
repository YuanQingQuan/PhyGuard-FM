from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _np(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x).squeeze()
    return x


def psnr(true, test, data_range=1.0) -> float:
    return float(peak_signal_noise_ratio(_np(true), _np(test), data_range=data_range))


def ssim(true, test, data_range=1.0) -> float:
    return float(structural_similarity(_np(true), _np(test), data_range=data_range))


def hallucination_energy(true, test, threshold=0.08) -> float:

    a, b = _np(true), _np(test)
    d = np.maximum(b - a - threshold, 0)
    return float(np.mean(d**2))
