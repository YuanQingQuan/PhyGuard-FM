from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from experiments.run_axial_prospective import MODES, SIGMA, THRESHOLD, draw_true, load_volume, axial_fisher

OUTDIR = ROOT / "results" / "axial_basin"
TARGET = 0.05
SAFE_MAGS = (0.30, 0.36)


def g_basin(thetas: np.ndarray):

    d = np.linalg.norm(thetas[:, None, :] - thetas[None, :, :], axis=-1)
    medoid = int(np.argmin(d.sum(axis=1)))
    diam = float(d.max())
    return diam, float(d[:, medoid].max()), float(np.median(d[:, medoid])), medoid


def signaware_err(est: np.ndarray, true: np.ndarray) -> float:

    est = np.atleast_2d(np.asarray(est, dtype=float))
    true = np.asarray(true, dtype=float)
    e = est - true
    alt = est + true
    sig = np.array([SIGMA[MODES.index(m)] for m in MODES])
    per_run = (np.minimum(e ** 2, alt ** 2) * sig ** 2).sum(axis=-1)
    return float(np.sqrt(per_run[0]))


def main():
    frames = []
    for name in ("init_scan", "gatesweep"):
        p = OUTDIR / f"{name}.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["source"] = name
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    structure = load_volume()

    rows = []
    for (mag, seed), g in df.groupby(["magnitude", "seed"]):
        thetas = g[[f"est_{m}" for m in MODES]].to_numpy(float)
        true = np.array([draw_true(float(mag), int(seed))[m] for m in MODES])
        diam, dmax_medoid, dmed, medoid = g_basin(thetas)
        err_medoid = signaware_err(thetas[medoid], true)
        succ_frac = float((g["err_rms_signaware"] < THRESHOLD).mean())
        F = axial_fisher(structure, draw_true(float(mag), int(seed)))
        glocal = float(np.sqrt(np.max(np.diag(np.linalg.pinv(F)))))
        rows.append({"magnitude": float(mag), "seed": int(seed),
                     "n_restarts": len(g), "rms": float(g["rms"].iloc[0]),
                     "G_basin": diam, "medoid_max": dmax_medoid,
                     "g_basin_medoid": dmed,
                     "G_local": glocal,
                     "err_medoid": err_medoid,
                     "medoid_ok": bool(err_medoid < TARGET),
                     "success_frac": succ_frac})

    per = pd.DataFrame(rows).sort_values(["magnitude", "seed"])
    per.to_csv(OUTDIR / "gates_per_dataset.csv", index=False)


    tau_b = 0.01
    per["consensus_pass"] = per["G_basin"] < tau_b
    per["fisher_pass"] = per["G_local"] < TARGET
    safe = per[per["medoid_ok"]]

    conf = {(t, fp, cp): 0
            for t in (True, False) for fp in (True, False) for cp in (True, False)}
    for _, r in per.iterrows():
        conf[(not bool(r["medoid_ok"]), bool(r["fisher_pass"]),
              bool(r["consensus_pass"]))] += 1

    unimodal = per[per["medoid_ok"] & (per["G_basin"] < tau_b)]
    multi_ok = per[per["medoid_ok"] & (per["G_basin"] >= tau_b)]
    payload = {

        "tau_b": tau_b,

        "separation": {
            "unimodal_max": float(unimodal["G_basin"].max()),
            "multimodal_min": float(per[per["G_basin"] >= tau_b]["G_basin"].min()),
            "n_unimodal": int(len(unimodal)),
            "n_multimodal": int((per["G_basin"] >= tau_b).sum()),
            "n_multimodal_ok": int(len(multi_ok)),
            "n_wrong_consensus": int((~per["medoid_ok"]).sum()),
            "g_basin_medoid_failing_max": float(
                per[~per["medoid_ok"]]["g_basin_medoid"].max()),
        },
        "confusion_counts": {
            f"truth_{'fail' if t else 'ok'}_fisher_{'pass' if fp else 'fail'}"
            f"_consensus_{'pass' if cp else 'fail'}": int(conf[(t, fp, cp)])
            for t in (True, False) for fp in (True, False) for cp in (True, False)
        },
        "per_dataset": per.to_dict(orient="records"),
    }
    (OUTDIR / "gates.json").write_text(json.dumps(payload, indent=2),
                                       encoding="utf-8")
    print(per.to_string(index=False,
                        float_format=lambda v: f"{v:.4g}"))
    print(f"\nTAU_B = {tau_b:.4g} waves")
    print(json.dumps(payload["separation"], indent=2))
    print(json.dumps(payload["confusion_counts"], indent=2))


if __name__ == "__main__":
    main()
