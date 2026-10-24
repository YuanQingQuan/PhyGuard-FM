from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

from .likelihood import local_standardized_residual
from .optics.forward import PhotonMicroscope


@dataclass
class VerificationResult:
    support_foundation: torch.Tensor
    support_physics: torch.Tensor
    fused: torch.Tensor
    residual_foundation: torch.Tensor
    residual_physics: torch.Tensor


def _resize_support(s: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if s.shape[-2:] == size:
        return s
    return F.interpolate(s, size=size, mode="bilinear", align_corners=False)


def verify_and_fuse(
    observation_counts: torch.Tensor,
    foundation_candidate: torch.Tensor,
    physics_candidate: torch.Tensor,
    microscope: PhotonMicroscope,
    *,
    local_window: int = 9,
    temperature: float = 1.0,
    conservative_bias: float = 0.15,
) -> VerificationResult:

    mu_f = microscope.expected_counts(foundation_candidate)
    mu_p = microscope.expected_counts(physics_candidate)
    r_f = local_standardized_residual(
        observation_counts, mu_f, microscope.read_noise_std, window=local_window
    )
    r_p = local_standardized_residual(
        observation_counts, mu_p, microscope.read_noise_std, window=local_window
    )
    score_f = -r_f / max(float(temperature), 1e-8)
    score_p = -r_p / max(float(temperature), 1e-8) + float(conservative_bias)
    weights = torch.softmax(torch.cat([score_f, score_p], dim=1), dim=1)
    wf, wp = weights[:, 0:1], weights[:, 1:2]

    target_size = foundation_candidate.shape[-2:]
    wf_img = _resize_support(wf, target_size)
    wp_img = _resize_support(wp, target_size)
    fused = wf_img * foundation_candidate + wp_img * physics_candidate
    return VerificationResult(wf_img, wp_img, fused, r_f, r_p)


def measurement_support_map(
    observation_counts: torch.Tensor,
    candidate: torch.Tensor,
    microscope: PhotonMicroscope,
    *,
    local_window: int = 9,
    temperature: float = 1.0,
) -> torch.Tensor:
    mu = microscope.expected_counts(candidate)
    residual = local_standardized_residual(
        observation_counts, mu, microscope.read_noise_std, window=local_window
    )
    support = torch.exp(-0.5 * residual / max(float(temperature), 1e-8))
    return _resize_support(support, candidate.shape[-2:])
