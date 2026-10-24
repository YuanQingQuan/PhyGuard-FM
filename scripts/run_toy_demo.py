from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from skimage.draw import line_aa, disk

from phyguard.identifiability import convolution_identifiability
from phyguard.metrics import psnr, ssim, hallucination_energy
from phyguard.optics.psf import MicroscopyPSFConfig, make_psf
from phyguard.optics.forward import PhotonMicroscope
from phyguard.physics_stats import estimate_blind_stats
from phyguard.verification import verify_and_fuse


def phantom(n=128, seed=7):
    rng = np.random.default_rng(seed)
    x = np.zeros((n, n), np.float32)
    for _ in range(18):
        y0, x0, y1, x1 = rng.integers(8, n - 8, size=4)
        rr, cc, val = line_aa(y0, x0, y1, x1)
        x[rr, cc] += val * rng.uniform(0.4, 1.0)
    for _ in range(10):
        cy, cx = rng.integers(10, n - 10, size=2)
        rr, cc = disk((cy, cx), rng.integers(2, 5), shape=x.shape)
        x[rr, cc] += rng.uniform(0.4, 1.0)
    x = gaussian_filter(x, 0.8)
    return x / (x.max() + 1e-8)


root = Path(__file__).resolve().parents[1]
outdir = root / "results" / "toy_demo"
outdir.mkdir(parents=True, exist_ok=True)

torch.manual_seed(3)
gt = phantom()
psf = make_psf(MicroscopyPSFConfig(wavelength_nm=520, na=1.2, pixel_size_nm=100, kernel_size=33))
mic = PhotonMicroscope(psf=psf, signal_photons=20.0, background_photons=1.0, read_noise_std=0.25)
obs = mic.sample(torch.tensor(gt)[None, None])
obs_img = obs["normalized"][0, 0].numpy()
obs_counts = obs["counts"]


physics_candidate = np.clip(gaussian_filter(obs_img, 1.2), 0, 1).astype(np.float32)


fm = np.clip(gaussian_filter(gt, 0.25), 0, 1).astype(np.float32)
rr, cc, val = line_aa(20, 90, 70, 115)
fm[rr, cc] = np.maximum(fm[rr, cc], 0.95 * val.astype(np.float32))
fm = gaussian_filter(fm, 0.45)
fm = np.clip(fm, 0, 1).astype(np.float32)

ver = verify_and_fuse(
    obs_counts,
    torch.tensor(fm)[None, None],
    torch.tensor(physics_candidate)[None, None],
    mic,
    local_window=7,
    temperature=0.5,
    conservative_bias=0.0,
)
fused = ver.fused[0, 0].detach().numpy()
support = ver.support_foundation[0, 0].detach().numpy()

stats = estimate_blind_stats(obs["normalized"][0, 0])
ident = convolution_identifiability(psf, gt.shape, gamma=0.05)
metrics = {
    "observation": {"psnr": psnr(gt, obs_img), "ssim": ssim(gt, obs_img)},
    "physics_candidate": {"psnr": psnr(gt, physics_candidate), "ssim": ssim(gt, physics_candidate), "hallucination_energy": hallucination_energy(gt, physics_candidate)},
    "foundation_candidate": {"psnr": psnr(gt, fm), "ssim": ssim(gt, fm), "hallucination_energy": hallucination_energy(gt, fm)},
    "fused": {"psnr": psnr(gt, fused), "ssim": ssim(gt, fused), "hallucination_energy": hallucination_energy(gt, fused)},
    "blind_stats": stats.to_dict(),
    "identifiability": ident.to_dict(),
}
with open(outdir / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

fig = plt.figure(figsize=(15, 8))
items = [
    (gt, "Ground truth"),
    (obs_img, "Photon-limited observation"),
    (physics_candidate, "Conservative measurement candidate"),
    (fm, "Foundation-like candidate\n(with unsupported structure)"),
    (support, "Measurement support\nfor foundation candidate"),
    (fused, "Physics-verified fusion"),
]
for i, (img, title) in enumerate(items, 1):
    ax = fig.add_subplot(2, 3, i)
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
fig.tight_layout()
fig.savefig(outdir / "toy_demo.png", dpi=180)
plt.close(fig)
print(json.dumps(metrics, indent=2))
print("saved:", outdir)
