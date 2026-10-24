from __future__ import annotations

from pathlib import Path
import json, sys, argparse
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from skimage.restoration import richardson_lucy, denoise_tv_chambolle

from phyguard.identifiability import convolution_identifiability
from phyguard.optics.forward import PhotonMicroscope
from phyguard.optics.psf import MicroscopyPSFConfig, make_psf
from phyguard.verification import verify_and_fuse
from experiments.run_real_optical_pilot import get_real_microscopy, inject_plausible_duplicate, normalize, auc_rank, masked_mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--photons', type=float, default=5.0)
    args = ap.parse_args()
    tag = str(args.photons).replace('.', 'p')
    outdir = ROOT / 'results' / f'real_identifiability_sweep_{tag}photons'
    outdir.mkdir(parents=True, exist_ok=True)
    gt, meta = get_real_microscopy()
    effective_pixel_nm = meta['native_pixel_spacing_um'] * 1000.0 * (384 / 256)

    foundation_clean = gt.copy()
    foundation_hall, halluc_mask, inj = inject_plausible_duplicate(foundation_clean, patch=28, strength=0.95)

    nas = [0.55, 0.70, 0.85, 1.00, 1.20, 1.40]
    defocuses = [0.0, 0.10, 0.20, 0.35]
    photons = float(args.photons)
    seeds = [3, 11, 29, 47, 71]
    rows = []

    for na in nas:
        for defocus in defocuses:
            cfg = MicroscopyPSFConfig(
                wavelength_nm=520.0, na=float(na), pixel_size_nm=effective_pixel_nm,
                kernel_size=41, aberrations_waves={'defocus': float(defocus)}
            )
            psf = make_psf(cfg)
            ident = convolution_identifiability(psf, gt.shape, gamma=0.05).to_dict()
            psf_np = psf.detach().cpu().numpy()
            for seed in seeds:
                torch.manual_seed(seed)
                mic = PhotonMicroscope(psf=psf, signal_photons=photons,
                                       background_photons=1.5, read_noise_std=0.15)
                obs = mic.sample(torch.tensor(gt)[None, None])
                counts = obs['counts']
                obs_img = obs['normalized'][0,0].detach().numpy().astype(np.float32)
                rl = richardson_lucy(np.clip(obs_img,0,1), psf_np, num_iter=12, clip=False)
                rl = normalize(np.clip(rl,0,None), p_low=0.1, p_high=99.7)
                physics = denoise_tv_chambolle(rl, weight=0.08, channel_axis=None).astype(np.float32)
                ver = verify_and_fuse(counts,
                                      torch.tensor(foundation_hall)[None,None],
                                      torch.tensor(physics)[None,None], mic,
                                      local_window=9, temperature=0.70, conservative_bias=0.05)
                support = np.clip(ver.support_foundation[0,0].detach().numpy(),0,1)
                tau = float(np.quantile(support, 0.005))
                keep = 1.0/(1.0+np.exp(-(support-tau)/0.02))
                fused = np.clip(keep*foundation_hall + (1-keep)*physics,0,1)
                f_mse = masked_mse(gt, foundation_hall, halluc_mask)
                fused_mse = masked_mse(gt, fused, halluc_mask)
                rows.append({
                    'na': na, 'defocus_waves': defocus, 'photons_per_pixel': photons, 'seed': seed,
                    'c_gamma': ident['c_gamma'],
                    'effective_rank_fraction': ident['effective_rank_fraction'],
                    'null_fraction': ident['null_fraction'],
                    'condition_proxy': ident['condition_proxy'],
                    'hallucination_auc': auc_rank(halluc_mask, 1-support),
                    'support_drop': float(support[~halluc_mask].mean()-support[halluc_mask].mean()),
                    'mse_reduction_factor': float(f_mse/max(fused_mse,1e-12)),
                })

    df = pd.DataFrame(rows)
    df.to_csv(outdir/'per_run_metrics.csv', index=False)
    summary = df.groupby(['na','defocus_waves'], as_index=False).agg(
        c_gamma=('c_gamma','first'),
        effective_rank_fraction=('effective_rank_fraction','first'),
        null_fraction=('null_fraction','first'),
        condition_proxy=('condition_proxy','first'),
        hallucination_auc_mean=('hallucination_auc','mean'),
        hallucination_auc_std=('hallucination_auc','std'),
        support_drop_mean=('support_drop','mean'),
        mse_reduction_factor_mean=('mse_reduction_factor','mean'),
    )
    summary.to_csv(outdir/'summary_by_operator.csv', index=False)

    metrics = ['c_gamma','effective_rank_fraction','null_fraction','condition_proxy']
    corr = {}
    for m in metrics:
        r_auc,p_auc = spearmanr(summary[m], summary['hallucination_auc_mean'])
        r_red,p_red = spearmanr(summary[m], summary['mse_reduction_factor_mean'])
        corr[m] = {'rho_auc': float(r_auc), 'p_auc': float(p_auc),
                   'rho_mse_reduction': float(r_red), 'p_mse_reduction': float(p_red)}
    with open(outdir/'correlations.json','w') as f:
        json.dump({'n_operators': len(summary), 'seeds_per_operator': len(seeds),
                   'photons_per_pixel': photons, 'spearman': corr,
                   'controlled_degradation': True,
                   }, f, indent=2)


    fig, ax = plt.subplots(figsize=(6,4.5))
    ax.scatter(summary['effective_rank_fraction'], summary['hallucination_auc_mean'])
    ax.set_xlabel('OTF effective-rank fraction')
    ax.set_ylabel('Unsupported-structure detection AUC')
    ax.set_ylim(0,1.02)
    fig.tight_layout(); fig.savefig(outdir/'effective_rank_vs_auc.png', dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6,4.5))
    ax.scatter(summary['c_gamma'], summary['hallucination_auc_mean'])
    ax.set_xlabel('C_gamma (OTF fraction above threshold)')
    ax.set_ylabel('Unsupported-structure detection AUC')
    ax.set_ylim(0,1.02)
    fig.tight_layout(); fig.savefig(outdir/'cgamma_vs_auc.png', dpi=180); plt.close(fig)

    print(summary.to_string(index=False))
    print(json.dumps(corr, indent=2))
    print('saved', outdir)

if __name__ == '__main__':
    main()
