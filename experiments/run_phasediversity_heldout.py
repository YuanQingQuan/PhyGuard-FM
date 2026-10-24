from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from phyguard.optics.zernike_ansi import (
    ANSI_ESTIMATED,
    ANSI_INDEX,
    ANSI_NAME,
    LAMBDA_UM,
    ansi_coeffs_to_waves,
)

from experiments.run_phasediversity_external import (
    DATASETS,
    build_otf,
    diversity_offsets,
    expected_counts,
    load_iteration,
    wavefront_rms,
)

ROOT = Path(__file__).resolve().parents[1]
ALL_MODES = tuple(ANSI_NAME[j] for j in ANSI_ESTIMATED)


MODE_SETS: dict[int, tuple[str, ...]] = {
    m: tuple(ANSI_NAME[j] for j in ANSI_ESTIMATED[:m]) for m in (4, 7, 10, 18)
}


def run_mode_set(
    modes, *, counts_t, offsets, x_init, s, b, n, pixel_size_nm, na,
    train, holdout, n_outer, n_inner, restarts, lr_x, lr_theta, tv_weight,
    theta_cmd, seed,
):
    rng = np.random.default_rng(seed)

    def tv(x):
        return ((x[..., 1:, :] - x[..., :-1, :]).abs().mean()
                + (x[..., :, 1:] - x[..., :, :-1]).abs().mean())

    def score(theta, x, frames):
        tot = 0.0
        for i in frames:
            otf = build_otf(theta, offsets[i], n=n, wavelength_nm=LAMBDA_UM * 1000.0,
                            pixel_size_nm=pixel_size_nm, na=na)
            r = expected_counts(x, otf, s, b).clamp_min(1e-6)
            tot = tot + (r - counts_t[i] * torch.log(r)).mean()
        return tot / len(frames)


    t_fixed = {m: torch.tensor(theta_cmd.get(m, 0.0)) for m in modes}
    xb = x_init.clone().requires_grad_(True)
    opt = torch.optim.Adam([xb], lr=lr_x)
    for _ in range(n_outer * n_inner):
        opt.zero_grad()
        xc = xb.clamp(0, 1)
        (score(t_fixed, xc, train) + tv_weight * tv(xc)).backward()
        opt.step()
    xb = xb.detach().clamp(0, 1)
    with torch.no_grad():
        base_train = float(score(t_fixed, xb, train))
        base_held = float(score(t_fixed, xb, [holdout]))
        base_wfe = wavefront_rms(t_fixed)


    best = None
    for r in range(restarts):
        x = x_init.clone().requires_grad_(True)
        theta = {m: torch.tensor(float(rng.uniform(-0.3, 0.3)) if r > 0 else 0.0,
                                 requires_grad=True) for m in modes}
        for _ in range(n_outer):
            opt_x = torch.optim.Adam([x], lr=lr_x)
            for _ in range(n_inner):
                opt_x.zero_grad()
                xc = x.clamp(0, 1)
                (score(theta, xc, train) + tv_weight * tv(xc)).backward()
                opt_x.step()
            opt_t = torch.optim.Adam(list(theta.values()), lr=lr_theta)
            for _ in range(n_inner):
                opt_t.zero_grad()
                score(theta, x.detach().clamp(0, 1), train).backward()
                opt_t.step()
        with torch.no_grad():
            xc = x.clamp(0, 1)
            tr = float(score(theta, xc, train))
            he = float(score(theta, xc, [holdout]))
            est = {m: float(theta[m].detach()) for m in modes}
        if best is None or tr < best["train"]:
            best = {"train": tr, "held": he, "theta": est,
                    "wfe": wavefront_rms(est)}
    return {"baseline": {"train": base_train, "held": base_held, "wfe": base_wfe},
            "blind": best}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="beads", choices=sorted(DATASETS))
    ap.add_argument("--iteration", type=int, default=1)
    ap.add_argument("--holdout", type=int, default=0)
    ap.add_argument("--n-outer", type=int, default=40)
    ap.add_argument("--n-inner", type=int, default=10)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--lr-x", type=float, default=0.05)
    ap.add_argument("--lr-theta", type=float, default=0.05)
    ap.add_argument("--tv-weight", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=str(ROOT / "results" / "phasediversity_external"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    counts, coeffs, _ = load_iteration(args.dataset, args.iteration)
    offsets = diversity_offsets(coeffs)
    truth = ansi_coeffs_to_waves(
        {k: float(v) for k, v in enumerate(coeffs[0], start=1) if v != 0})
    theta_cmd = {m: truth.get(m, 0.0) for m in ALL_MODES}

    n = counts.shape[-1]
    counts_t = torch.as_tensor(counts[:, None], dtype=torch.float32)
    raw = torch.as_tensor(counts[0], dtype=torch.float32)
    lo = torch.quantile(raw.flatten(), 0.1)
    hi = torch.quantile(raw.flatten(), 0.997)
    x_init = ((raw - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)[None, None]
    b = float(torch.median(raw))
    s = float(raw.mean() - b) / max(float(x_init.mean()), 1e-6)

    train = [i for i in range(5) if i != args.holdout]
    print(f"{args.dataset} it{args.iteration}: holding out frame {args.holdout}, "
          f"training on {train}")
    print(f"signal_photons={s:.1f} background_photons={b:.1f}; "
          f"truth wavefront RMS {wavefront_rms(truth):.1f} nm")

    rows = []
    for M, modes in sorted(MODE_SETS.items()):

        res = run_mode_set(
            modes, counts_t=counts_t, offsets=offsets, x_init=x_init, s=s, b=b,
            n=n, pixel_size_nm=104.0, na=1.2, train=train, holdout=args.holdout,
            n_outer=args.n_outer, n_inner=args.n_inner, restarts=args.restarts,
            lr_x=args.lr_x, lr_theta=args.lr_theta, tv_weight=args.tv_weight,
            theta_cmd=theta_cmd, seed=args.seed)
        bl, bd = res["baseline"], res["blind"]

        gate = bd["train"] <= bl["train"]
        rows.append({"M": M, "modes": list(modes), "gate": gate,
                     "train_base": bl["train"], "held_base": bl["held"],
                     "train_blind": bd["train"], "held_blind": bd["held"],
                     "wfe_base": bl["wfe"], "wfe_blind": bd["wfe"],
                     "theta": bd["theta"]})
        print(f"\n--- M={M:2d} ({len(modes)} modes)  gate(train blind<=base): {gate}")
        print(f"    train NLL  base {bl['train']:.6f}   blind {bd['train']:.6f}   "
              f"delta {bd['train']-bl['train']:+.6f}")
        print(f"    HELD  NLL  base {bl['held']:.6f}   blind {bd['held']:.6f}   "
              f"delta {bd['held']-bl['held']:+.6f}   <-- the question")
        print(f"    wavefront RMS  base {bl['wfe']:.1f} nm   blind {bd['wfe']:.1f} nm")

    print("\n" + "=" * 78)
    print(f"{'M':>3s} {'train base':>11s} {'train blind':>12s} {'d_train':>10s} "
          f"{'held base':>11s} {'held blind':>11s} {'d_held':>9s}")
    for r in rows:
        print(f"{r['M']:3d} {r['train_base']:11.4f} {r['train_blind']:12.4f} "
              f"{r['train_blind']-r['train_base']:+10.4f} "
              f"{r['held_base']:11.4f} {r['held_blind']:11.4f} "
              f"{r['held_blind']-r['held_base']:+9.4f}")

    d_held = [r["held_blind"] - r["held_base"] for r in rows if r["gate"]]
    if d_held:
        print(f"\nheld-out delta over all gated M: {[round(v,4) for v in d_held]}")
        if all(v <= 0 for v in d_held):
            print("=> the extra modes improve the held-out frame; consistent with "
                  "them recovering real structure theta_cmd does not describe.")
        elif all(v >= 0 for v in d_held):
            print("=> the extra modes do NOT improve the held-out frame; the "
                  "training-set gain is overfitting/object-model slack.")
        else:
            print("=> mixed: inspect per-M signs before claiming anything.")

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.dataset}_it{args.iteration}_holdout{args.holdout}"
    (out / f"{tag}_heldout.json").write_text(json.dumps(
        {"dataset": args.dataset, "iteration": args.iteration,
         "holdout": args.holdout, "train": train,
         "signal_photons": s, "background_photons": b,
         "truth_wavefront_rms_nm": wavefront_rms(truth), "rows": rows,
         "settings": vars(args)}, indent=2))
    print(f"\nwrote {out / (tag + '_heldout.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
