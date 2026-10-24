from __future__ import annotations
import argparse, csv
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from phyguard.identifiability import convolution_identifiability
from phyguard.optics.psf import MicroscopyPSFConfig, make_psf


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--size',type=int,default=256)
    args=ap.parse_args(); rows=[]
    for na in [0.8,1.0,1.2,1.4]:
      for d in [0,0.1,0.25,0.5,0.75]:
        psf=make_psf(MicroscopyPSFConfig(na=na,aberrations_waves={'defocus':d}))
        r=convolution_identifiability(psf,(args.size,args.size),0.05).to_dict(); r.update(na=na,defocus_waves=d); rows.append(r)
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    with open(out,'w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

if __name__=='__main__': main()
