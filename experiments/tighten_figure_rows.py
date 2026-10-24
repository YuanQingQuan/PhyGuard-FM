from __future__ import annotations

import argparse
import pathlib

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
MIN_BAND_IN = 0.30
KEEP_IN = 0.14
WHITE = 253


def find_bands(rgb: np.ndarray, dpi: int) -> list[tuple[int, int]]:

    ink = rgb.min(axis=2) < WHITE
    rows_blank = ~ink.any(axis=1)
    bands: list[tuple[int, int]] = []
    start = None
    for k, blank in enumerate(rows_blank):
        if blank and start is None:
            start = k
        elif not blank and start is not None:
            bands.append((start, k))
            start = None
    if start is not None:
        bands.append((start, len(rows_blank)))
    min_px = int(round(MIN_BAND_IN * dpi))
    return [b for b in bands if b[1] - b[0] >= min_px]


def tighten(path: pathlib.Path, dpi: int, dry_run: bool = False) -> dict:
    im = Image.open(path)
    rgb = np.asarray(im.convert("RGB"))
    keep_px = int(round(KEEP_IN * dpi))
    bands = find_bands(rgb, dpi)
    if not bands:
        return {"path": path.name, "bands": 0, "before": rgb.shape[0], "after": rgb.shape[0]}

    keep_rows = list(range(rgb.shape[0]))
    for a, b in bands:

        half = keep_px // 2
        keep_rows = [y for y in keep_rows if not (a + half <= y < b - (keep_px - half))]

    out = rgb[keep_rows]
    assert (out.min(axis=2) < WHITE).sum() == (rgb.min(axis=2) < WHITE).sum(), (
        f"{path.name}: ink count changed, refusing to write"
    )
    if not dry_run:
        Image.fromarray(out).save(path)
    return {
        "path": path.name,
        "bands": len(bands),
        "before": rgb.shape[0],
        "after": out.shape[0],
        "band_in": [round((b - a) / dpi, 2) for a, b in bands],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "results" / "paper_figures" / "v7"))
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    total_before = total_after = 0
    for p in sorted(pathlib.Path(args.dir).glob("Fig*.png")):
        r = tighten(p, args.dpi, args.dry_run)
        total_before += r["before"]
        total_after += r["after"]
        if r["bands"]:
            print(
                f"{r['path']}: {r['bands']} band(s) {r['band_in']} -> "
                f"{r['before']/args.dpi:.2f}in tall becomes {r['after']/args.dpi:.2f}in"
            )
        else:
            print(f"{r['path']}: clean")
    print(
        f"\ntotal height {total_before/args.dpi:.2f}in -> {total_after/args.dpi:.2f}in "
        f"(saved {(total_before-total_after)/args.dpi:.2f}in)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
