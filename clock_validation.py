"""Validation follow-up to Umaid's clock model (PR #1, sim_clock_gamma.py).

Umaid's files are used unchanged. This script adds the two tests that
turn his ZT9 result from a promising fit into evidence:

Part A - reproduction.
    Refit the six ZT9 plants with a free per-plant clock (amplitude a
    and phase free per plant, Umaid's bounds) and compare against the
    official static fits. This reproduces his table by an independent
    fitting path (bounded least_squares from the official fit as the
    starting point, rather than differential evolution).

Part B - shared-clock held-out test (the decisive one).
    A circadian clock should have ONE phase. Fit a single shared
    (a, phase) plus per-plant (k, tau, max_rate, phi0) on the 42
    non-ZT9 plants only, alternating plant fits and shared-clock fits.
    Then freeze (a, phase), fit only the four standard parameters for
    each ZT9 plant, and compare RMSE against (i) the official static
    fits and (ii) Umaid's free-clock fits. If the shared clock,
    calibrated without ever seeing ZT9, still fixes ZT9's shape, the
    circadian-gating interpretation survives its first held-out test.
    If only the per-plant free clock helps, the "clock" is acting as a
    flexible curve-bender.
"""

from __future__ import annotations

import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import fit_real
from data_real import curves, load_long
from sim import static_light
from sim_clock_gamma import simulate as clock_simulate

ZT9_FREE_CLOCK_REPORTED = {  # Umaid, PR #1, docs/CHANGES_15OCT_UMAID.md
    "ZT9-1": 3.20, "ZT9-2": 1.80, "ZT9-3": 0.86,
    "ZT9-4": 1.01, "ZT9-5": 0.84, "ZT9-6": 1.20,
}


def clock_curve(k, tau, mr, phi, a, phase, zt0, te):
    ts, th = clock_simulate(k, tau, mr, static_light(phi), float(te[-1]),
                            gamma=0.0, a=a, phase=phase, ZT_start=zt0)
    return np.interp(te, ts, th)


def fit_plant_clock(te, y, zt0, x0, shared=None):
    """Fit one plant. shared=None -> a, phase free; else (a, phase) fixed."""
    if shared is None:
        lo = np.array([1.0, 0.10, 2.0, 30.0, 0.0, 0.0])
        hi = np.array([30.0, 12.0, 100.0, 90.0, 0.90, 24.0])
        start = np.array([x0[0], max(x0[1], 0.10), x0[2], x0[3], 0.5, 8.0])

        def resid(x):
            return clock_curve(x[0], x[1], x[2], x[3], x[4], x[5], zt0, te) - y
    else:
        a, phase = shared
        lo, hi = fit_real.LO, fit_real.HI
        start = np.array(x0, float)

        def resid(x):
            return clock_curve(x[0], x[1], x[2], x[3], a, phase, zt0, te) - y
    start = np.minimum(np.maximum(start, lo * 1.001), hi * 0.999)
    res = least_squares(resid, start, bounds=(lo, hi), xtol=1e-6, ftol=1e-8)
    rmse = float(np.sqrt(np.mean(res.fun ** 2)))
    return res.x, rmse


def main():
    os.makedirs("results/figures", exist_ok=True)
    t0 = time.time()
    fits = pd.read_csv("results/fit_fig1c.csv").set_index("plant")
    df = load_long(sheet="Fig1C")
    allc = {}
    for key, (t, y) in curves(df).items():
        label = key.split("/")[1]
        allc[label] = (t - t[0], y, float(t[0]), fits.loc[label, "block"])

    def official(label):
        r = fits.loc[label]
        return [float(r.k), float(r.tau), float(r.max_rate), float(r.phi0)]

    # ---------------- Part A: free-clock reproduction on ZT9 -------------
    rowsA = []
    zt9 = sorted([p for p in allc if p.startswith("ZT9")])
    for label in zt9:
        te, y, zt0, _ = allc[label]
        x, rmse = fit_plant_clock(te, y, zt0, official(label), shared=None)
        rowsA.append({"plant": label, "static_rmse": float(fits.loc[label, "rmse"]),
                      "free_clock_rmse_refit": rmse,
                      "free_clock_rmse_um_reported":
                          ZT9_FREE_CLOCK_REPORTED[label],
                      "a": float(x[4]), "phase": float(x[5]),
                      "tau_clock": float(x[1])})
        print("A %s free-clock RMSE %.2f (Umaid %.2f) a=%.2f phase=%.1f (%.0fs)"
              % (label, rmse, ZT9_FREE_CLOCK_REPORTED[label], x[4], x[5],
                 time.time() - t0), flush=True)

    # ---------------- Part B: shared clock on non-ZT9, test on ZT9 -------
    train = [p for p in allc if not p.startswith("ZT9")]
    params = {p: np.array(official(p), float) for p in train}
    shared = (0.5, 8.0)
    for rnd in range(2):
        for p in train:
            te, y, zt0, _ = allc[p]
            x, _ = fit_plant_clock(te, y, zt0, params[p], shared=shared)
            params[p] = x
        # fit shared (a, phase) with plant params frozen
        def shared_resid(s):
            a, phase = float(s[0]), float(s[1])
            out = []
            for p in train:
                te, y, zt0, _ = allc[p]
                k, tau, mr, phi = params[p]
                out.append(clock_curve(k, tau, mr, phi, a, phase, zt0, te) - y)
            return np.concatenate(out)
        res = least_squares(shared_resid, np.array(shared),
                            bounds=([0.0, 0.0], [0.95, 24.0]),
                            xtol=1e-4, ftol=1e-6)
        shared = (float(res.x[0]), float(res.x[1]))
        print("B round %d shared a=%.3f phase=%.2f (%.0fs)"
              % (rnd, shared[0], shared[1], time.time() - t0), flush=True)

    def rmse_of(label, prm, shr):
        te, y, zt0, _ = allc[label]
        pred = clock_curve(prm[0], prm[1], prm[2], prm[3], shr[0], shr[1],
                           zt0, te)
        return float(np.sqrt(np.mean((pred - y) ** 2)))

    train_rows = [{"plant": p,
                   "static_rmse": float(fits.loc[p, "rmse"]),
                   "shared_clock_rmse_train": rmse_of(p, params[p], shared)}
                  for p in train]

    rowsB = []
    for label in zt9:
        te, y, zt0, _ = allc[label]
        x, rmse = fit_plant_clock(te, y, zt0, official(label), shared=shared)
        rowsB.append({"plant": label,
                      "static_rmse": float(fits.loc[label, "rmse"]),
                      "shared_clock_rmse_heldout": rmse})
        print("B %s held-out shared-clock RMSE %.2f vs static %.2f (%.0fs)"
              % (label, rmse, float(fits.loc[label, "rmse"]),
                 time.time() - t0), flush=True)

    A = pd.DataFrame(rowsA)
    B = pd.DataFrame(rowsB)
    T = pd.DataFrame(train_rows)
    A.to_csv("results/clock_validation_freeclock_zt9.csv", index=False)
    B.to_csv("results/clock_validation_shared_heldout.csv", index=False)
    summary = {
        "runtime_seconds": round(time.time() - t0, 1),
        "shared_clock_fitted_without_ZT9": {"a": shared[0], "phase": shared[1]},
        "ZT9_median_rmse": {
            "static": float(B.static_rmse.median()),
            "free_clock_um_reported":
                float(np.median(list(ZT9_FREE_CLOCK_REPORTED.values()))),
            "shared_clock_heldout": float(B.shared_clock_rmse_heldout.median()),
        },
        "training_blocks_median_rmse": {
            "static": float(T.static_rmse.median()),
            "shared_clock_in_sample": float(T.shared_clock_rmse_train.median()),
        },
        "per_plant": B.to_dict("records"),
    }
    with open("results/clock_validation_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(B))
    w = 0.27
    ax.bar(x - w, B.static_rmse, w, label="static (official)")
    ax.bar(x, [ZT9_FREE_CLOCK_REPORTED[p] for p in B.plant], w,
           label="free clock per plant (Umaid)")
    ax.bar(x + w, B.shared_clock_rmse_heldout, w,
           label="shared clock, ZT9 held out")
    ax.set_xticks(x, B.plant)
    ax.set_ylabel("fit RMSE (deg)")
    ax.set_title("Does a shared circadian clock predict ZT9 without seeing it?\n"
                 "(shared a=%.2f, phase=%.1f h fitted on the other 42 plants)"
                 % shared)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/figures/clock_validation_dashboard.png", dpi=170)
    print(json.dumps(summary, indent=2))
    print("saved results/figures/clock_validation_dashboard.png")


if __name__ == "__main__":
    main()
