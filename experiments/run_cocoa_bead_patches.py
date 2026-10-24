from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import tifffile
from scipy.ndimage import maximum_filter, gaussian_filter

from experiments.run_cocoa_validation_v2 import (
    NA, WAVEL_NM, PIXEL_NM, N_OBJ, DZ_UM, Z_FRAMES, MODES7, GT,
)
from experiments.run_cocoa_validation_v3 import estimate
from experiments.run_real_optical_pilot import normalize

PATCH = 96
MIN_DIST = 80
MARGIN = 64
N_PATCHES = 8


def bead_centers(frame):
    g = gaussian_filter(frame, 2.0)
    thr = np.percentile(g, 97.0)
    mx = maximum_filter(g, size=MIN_DIST)
    ys, xs = np.where((g == mx) & (g > thr))
    order = np.argsort(-g[ys, xs])
    pts = []
    for idx in order:
        y, x = int(ys[idx]), int(xs[idx])
        if y < MARGIN or x < MARGIN or y > frame.shape[0] - MARGIN or x > frame.shape[1] - MARGIN:
            continue
        if all((y - py) ** 2 + (x - px) ** 2 >= MIN_DIST ** 2 for py, px in pts):
            pts.append((y, x))
        if len(pts) >= N_PATCHES:
            break
    return pts


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    outdir = ROOT / "results" / "real_aberration_validation_v3"
    outdir.mkdir(parents=True, exist_ok=True)
    stacks = {tag: tifffile.imread(ROOT / "data" / "cocoa_beads" / f).astype(np.float32)
              for tag, f in [("reference", "ref_no_aberration.tif"),
                             ("aberrated", "aberrated_75nm.tif")]}
    zi = stacks["reference"].shape[0] // 2
    centers = bead_centers(stacks["reference"][zi])
    print("bead centers:", centers, flush=True)

    rows = []
    h = PATCH // 2
    for tag, st in stacks.items():
        for j, (y, x) in enumerate(centers):
            frames = [normalize(st[zi + dz][y - h:y + h, x - h:x + h],
                                p_low=0.5, p_high=99.5) for dz in Z_FRAMES]
            theta = estimate(frames, device, seed=0)
            row = {"stack": tag, "patch": j, "y": y, "x": x}
            row.update({m: float(theta[m]) for m in MODES7})
            rows.append(row)
            print(row, flush=True)
            pd.DataFrame(rows).to_csv(outdir / "bead_patches.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "bead_patches.csv", index=False)
    print("\n=== per-stack mean +/- std over patches (waves) ===")
    g = df.groupby("stack")[MODES7].agg(["mean", "std"])
    print(g.round(4).to_string())
    print("\n=== between-stack shift (abe - ref) vs within-stack spread ===")
    for m in MODES7:
        mu_r, sd_r = df[df["stack"] == "reference"][m].mean(), df[df["stack"] == "reference"][m].std()
        mu_a, sd_a = df[df["stack"] == "aberrated"][m].mean(), df[df["stack"] == "aberrated"][m].std()
        shift = mu_a - mu_r
        spread = float(np.hypot(sd_r, sd_a))
        gt = GT.get(m, 0.0)
        print(f"  {m:10s}: shift {shift:+.4f} (GT {gt:+.4f}), pooled spread {spread:.4f}, "
              f"|shift|/spread {abs(shift) / max(spread, 1e-9):.2f}")
    print("saved:", outdir / "bead_patches.csv")


if __name__ == "__main__":
    main()
