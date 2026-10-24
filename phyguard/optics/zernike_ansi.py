from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import torch


ANSI_INDEX: Dict[int, Tuple[int, int]] = {
    0: (0, 0),
    1: (1, -1),
    2: (1, 1),
    3: (2, -2),
    4: (2, 0),
    5: (2, 2),
    6: (3, -3),
    7: (3, -1),
    8: (3, 1),
    9: (3, 3),
    10: (4, -4),
    11: (4, -2),
    12: (4, 0),
    13: (4, 2),
    14: (4, 4),
    15: (5, -5),
    16: (5, -3),
    17: (5, -1),
    18: (5, 1),
    19: (5, 3),
    20: (5, 5),
}


ANSI_NAME: Dict[int, str] = {
    3: "astig_y",   4: "defocus",   5: "astig_x",
    6: "trefoil_y", 7: "coma_y",    8: "coma_x",    9: "trefoil_x",
    10: "quad4_y",  11: "astig2_y", 12: "spherical", 13: "astig2_x",
    14: "quad4_x",
    15: "penta5_y", 16: "tref3_y",  17: "coma2_y",  18: "coma2_x",
    19: "tref3_x",  20: "penta5_x",
}


ANSI_ESTIMATED = tuple(range(3, 21))


ANSI_MODE_NAMES: Tuple[str, ...] = tuple(
    ANSI_NAME[j] for j in ANSI_ESTIMATED
)

LAMBDA_UM = 0.532
PIXEL_SIZE_UM = 0.104
NA = 1.2


def radial_polynomial(n: int, m: int, rho: torch.Tensor) -> torch.Tensor:

    m = abs(int(m))
    n = int(n)
    if n < 0 or m > n or (n - m) % 2:
        raise ValueError(f"invalid Zernike order (n={n}, m={m})")
    out = torch.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        coeff = (
            (-1) ** k
            * math.factorial(n - k)
            / (
                math.factorial(k)
                * math.factorial((n + m) // 2 - k)
                * math.factorial((n - m) // 2 - k)
            )
        )
        out = out + coeff * rho ** (n - 2 * k)
    return out


def zernike_mode(n: int, m: int, rho: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:

    radial = radial_polynomial(n, m, rho)
    if m == 0:
        return radial
    if m > 0:
        return radial * torch.cos(m * theta)
    return radial * torch.sin(abs(m) * theta)


def mode_values(rho: torch.Tensor, theta: torch.Tensor) -> Dict[str, torch.Tensor]:

    return {
        ANSI_NAME[j]: zernike_mode(*ANSI_INDEX[j], rho, theta)
        for j in ANSI_ESTIMATED
    }


def ansi_coeffs_to_waves(
    coeffs_um: Dict[int, float],
    *,
    wavelength_um: float = LAMBDA_UM,
    indices: Iterable[int] | None = None,
) -> Dict[str, float]:

    waves: Dict[str, float] = {}
    for j, value in coeffs_um.items():
        j = int(j)
        if indices is not None and j not in set(indices):
            continue
        if j not in ANSI_NAME:
            raise ValueError(
                f"ANSI index {j} has no PhyGuard mode (piston/tilt are not modelled)"
            )
        waves[ANSI_NAME[j]] = float(value) / float(wavelength_um)
    return waves


def pupil_max_radius(n: int, *, pixel_size_um: float = PIXEL_SIZE_UM,
                     wavelength_um: float = LAMBDA_UM, na: float = NA) -> float:

    fre_max = na / wavelength_um
    fre_sampling_phase = n * pixel_size_um
    pixel_size_phase = 1.0 / fre_sampling_phase
    return round(n / 2) * pixel_size_phase / fre_max


def pupil_grid_exact(
    n: int,
    *,
    pixel_size_um: float = PIXEL_SIZE_UM,
    wavelength_um: float = LAMBDA_UM,
    na: float = NA,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    r_max = pupil_max_radius(n, pixel_size_um=pixel_size_um,
                             wavelength_um=wavelength_um, na=na)
    x = torch.linspace(-r_max, r_max, n, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(x, x, indexing="ij")
    rho = torch.sqrt(xx**2 + yy**2)
    theta = torch.atan2(yy, xx)
    aperture = (rho <= 1.0).to(dtype)
    return rho, theta, aperture


def pupil_radius_samples(n: int, *, pixel_size_um: float = PIXEL_SIZE_UM,
                         wavelength_um: float = LAMBDA_UM, na: float = NA) -> float:

    return (n - 1) * pixel_size_um * na / wavelength_um
