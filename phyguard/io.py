from __future__ import annotations

from pathlib import Path
import numpy as np
import tifffile
from PIL import Image


def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        x = tifffile.imread(path)
    elif path.suffix.lower() == ".npy":
        x = np.load(path)
    else:
        x = np.asarray(Image.open(path))
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3 and x.shape[-1] in (3, 4):
        x = x[..., :3].mean(axis=-1)
    if x.ndim > 2:
        x = np.squeeze(x)
    x -= x.min()
    m = x.max()
    if m > 0:
        x /= m
    return x


def save_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(image, dtype=np.float32)
    if path.suffix.lower() in {".tif", ".tiff"}:
        tifffile.imwrite(path, x)
    elif path.suffix.lower() == ".npy":
        np.save(path, x)
    else:
        Image.fromarray(np.clip(x * 255.0, 0, 255).astype(np.uint8)).save(path)
