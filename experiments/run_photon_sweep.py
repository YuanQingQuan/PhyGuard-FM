from __future__ import annotations

import argparse, csv
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from phyguard.io import read_image
from phyguard.metrics import psnr, ssim
from phyguard.optics.psf import MicroscopyPSFConfig, make_psf
from phyguard.optics.forward import PhotonMicroscope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--photons", default="0.5,1,2,5,10,20,50")
    ap.add_argument("--background", type=float, default=1.0)
    args = ap.parse_args()
    files = sorted([p for p in Path(args.gt_dir).iterdir() if p.suffix.lower() in {'.tif','.tiff','.png','.jpg','.npy'}])
    psf = make_psf(MicroscopyPSFConfig())
    rows=[]
    for photons in [float(x) for x in args.photons.split(',')]:
        mic = PhotonMicroscope(psf, photons, args.background, 0.25)
        for p in files:
            gt=read_image(p)
            obs=mic.sample(torch.tensor(gt)[None,None])["normalized"][0,0].numpy()
            rows.append(dict(file=p.name, photons=photons, psnr=psnr(gt,obs), ssim=ssim(gt,obs)))
    out=Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

if __name__ == '__main__': main()
