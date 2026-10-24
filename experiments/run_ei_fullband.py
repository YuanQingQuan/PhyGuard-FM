import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from experiments.run_effective_information import (
    MODES, OUT, THETA, load_structure, moments, schur_real_basis, crb,
)

NA, WL, PX, N = 1.2, 520.0, 62.6, 256
R_MAX = (2 * NA / WL * PX) * N
print("incoherent OTF cutoff: %.4f cy/pixel = %.2f bins on the %d-grid"
      % (2 * NA / WL * PX, R_MAX, N), flush=True)


def exact_band_radius():

    return R_MAX


def sweep(s_list=(50.0,), radii=(40, 48, 56, 64), tag="fullband_schur"):
    x = load_structure(N)
    rows = []
    for s in s_list:
        F_tt, dmu, lam, h_full = moments(x, THETA, s)
        for radius in radii:
            t0 = time.time()
            F_eff, K = schur_real_basis(F_tt, dmu, lam, h_full, s, radius,
                                        memmap_dir=OUT)
            w_tt = np.linalg.eigvalsh(F_tt)
            w_eff = np.linalg.eigvalsh(F_eff)
            c_tt_p = np.sqrt(np.diag(np.linalg.pinv(F_tt)))
            c_eff_p = np.sqrt(np.diag(np.linalg.pinv(F_eff)))
            c_tt_t, nd_tt, _ = crb(F_tt)
            c_eff_t, nd_eff, _ = crb(F_eff)
            row = {
                "s": s, "radius": radius, "K": K,
                "lambda_min_ratio": float(w_eff[0] / w_tt[0]),
                "crb_norm_inflation_pinv": float(
                    np.linalg.norm(c_eff_p) / np.linalg.norm(c_tt_p)),
                **{f"crb_inflation_pinv_{m}": float(c_eff_p[i] / c_tt_p[i])
                   for i, m in enumerate(MODES)},
                "crb_norm_inflation_thresholded": float(
                    np.linalg.norm(c_eff_t) / np.linalg.norm(c_tt_t)),
                "n_degenerate_eff": nd_eff,
                "crb_eff_max_mode_waves": float(np.max(c_eff_t)),
                "minutes": round((time.time() - t0) / 60.0, 1),
            }
            rows.append(row)
            print("s=%g R=%d K=%d: lam_min ratio %.4f, CRB(pinv) inflation "
                  "%.3f (per mode %s), thresholded %.3f, blind CRB max %.4f w "
                  "[%.1f min]"
                  % (s, radius, K, row["lambda_min_ratio"],
                     row["crb_norm_inflation_pinv"],
                     " ".join("%.2f" % row[f"crb_inflation_pinv_{m}"]
                              for m in MODES),
                     row["crb_norm_inflation_thresholded"],
                     row["crb_eff_max_mode_waves"], row["minutes"]), flush=True)
            pd.DataFrame(rows).to_csv(OUT / f"{tag}.csv", index=False)
    return rows


def exact_band(s=50.0):

    x = load_structure(N)
    F_tt, dmu, lam, h_full = moments(x, THETA, s)
    radius = exact_band_radius()
    t0 = time.time()
    F_eff, K = schur_real_basis(F_tt, dmu, lam, h_full, s, radius,
                                memmap_dir=OUT)
    w_tt = np.linalg.eigvalsh(F_tt)
    w_eff = np.linalg.eigvalsh(F_eff)
    c_tt_p = np.sqrt(np.diag(np.linalg.pinv(F_tt)))
    c_eff_p = np.sqrt(np.diag(np.linalg.pinv(F_eff)))
    c_tt_t, nd_tt, _ = crb(F_tt)
    c_eff_t, nd_eff, _ = crb(F_eff)
    out = {
        "R_bins": round(radius, 2), "R_max_physical": R_MAX, "K": int(K),
        "lambda_min_ratio": float(w_eff[0] / w_tt[0]),
        "crb_inflation_norm_pinv": float(
            np.linalg.norm(c_eff_p) / np.linalg.norm(c_tt_p)),
        "crb_inflation_per_mode_pinv": [
            float(c_eff_p[i] / c_tt_p[i]) for i in range(len(MODES))],
        "crb_inflation_norm_thresholded": float(
            np.linalg.norm(c_eff_t) / np.linalg.norm(c_tt_t)),
        "crb_tt_waves_thresholded": [float(v) for v in c_tt_t],
        "crb_eff_waves_thresholded": [float(v) for v in c_eff_t],
        "n_degenerate": int(nd_eff),
        "minutes": round((time.time() - t0) / 60.0, 1),
    }
    print(json.dumps(out, indent=2), flush=True)
    (OUT / "exact_band.json").write_text(json.dumps(out, indent=2),
                                         encoding="utf-8")
    return out


def parity_exact(s=50.0, alphas=(1e-3, 1e-2, 0.1, 0.3)):

    x = load_structure(N)
    radius = exact_band_radius()
    rows = []
    for a in alphas:
        theta = {"defocus": 0.0, "astig_x": float(a),
                 "coma_x": 0.17, "spherical": 0.0}
        F_tt, dmu, lam, h_full = moments(x, theta, s)
        F_eff, K = schur_real_basis(F_tt, dmu, lam, h_full, s, radius,
                                    memmap_dir=OUT)
        rows.append({"alpha": float(a), "F_tt_xx": float(F_tt[1, 1]),
                     "F_eff_xx": float(F_eff[1, 1]), "K": int(K)})
        print("alpha=%.4g: F_tt=%.5g F_eff=%.5g ratio=%.4f K=%d"
              % (a, F_tt[1, 1], F_eff[1, 1], F_eff[1, 1] / F_tt[1, 1], K),
              flush=True)
        pd.DataFrame(rows).to_csv(OUT / "fullband_parity.csv", index=False)
    q = [r["F_eff_xx"] / r["alpha"] ** 2 for r in rows]
    print("F_eff/alpha^2 across the sweep:", ["%.4g" % v for v in q], flush=True)
    return rows


def invariance_exact(s_small=5.0):

    x = load_structure(N)
    radius = exact_band_radius()
    out = {}
    for s in (50.0, s_small):
        F_tt, dmu, lam, h_full = moments(x, THETA, s)
        F_eff, K = schur_real_basis(F_tt, dmu, lam, h_full, s, radius,
                                    memmap_dir=OUT)
        c_tt_p = np.sqrt(np.diag(np.linalg.pinv(F_tt)))
        c_eff_p = np.sqrt(np.diag(np.linalg.pinv(F_eff)))
        out["s=%g" % s] = {
            "K": int(K),
            "crb_norm_inflation_pinv": float(
                np.linalg.norm(c_eff_p) / np.linalg.norm(c_tt_p)),
        }
        print("s=%g: inflation %.4f" % (s, out["s=%g" % s]["crb_norm_inflation_pinv"]),
              flush=True)
    (OUT / "fullband_invariance.json").write_text(json.dumps(out, indent=2),
                                                  encoding="utf-8")
    return out


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    print("physical support radius passed to the mask: %.2f bins"
          % exact_band_radius(), flush=True)
    if stage in ("exact", "all"):
        exact_band()
    if stage in ("parity", "all"):
        parity_exact()
    if stage in ("invariance", "all"):
        invariance_exact()
    if stage in ("invariance", "all"):
        invariance_exact()
