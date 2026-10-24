from __future__ import annotations

import torch
import torch.nn.functional as F


def poisson_nll(observed_counts: torch.Tensor, expected_counts: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    expected_counts = expected_counts.clamp_min(1e-8)
    loss = expected_counts - observed_counts * torch.log(expected_counts)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(reduction)


def poisson_gaussian_nll(
    observed_counts: torch.Tensor,
    expected_counts: torch.Tensor,
    read_noise_std: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:

    var = expected_counts.clamp_min(1e-6) + float(read_noise_std) ** 2
    loss = 0.5 * ((observed_counts - expected_counts) ** 2 / var + torch.log(var))
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(reduction)


def local_standardized_residual(
    observed_counts: torch.Tensor,
    expected_counts: torch.Tensor,
    read_noise_std: float = 0.0,
    window: int = 9,
) -> torch.Tensor:

    if window % 2 == 0:
        raise ValueError("window must be odd")
    var = expected_counts.clamp_min(1e-6) + float(read_noise_std) ** 2
    z2 = (observed_counts - expected_counts) ** 2 / var
    if z2.ndim == 2:
        z2 = z2[None, None]
    elif z2.ndim == 3:
        z2 = z2[None]
    pooled = F.avg_pool2d(z2, kernel_size=window, stride=1, padding=window // 2)
    return pooled
