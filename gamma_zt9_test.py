"""Proprioception (gamma) test on ZT9-1: reproduction of the negative result.

Refits ZT9-1 (the worst-fitting seedling) with the proprioceptive term of
sim_gamma.py, holding gamma fixed on a grid and then free, with all other
parameters fitted by bounded least squares. The official static fit for
this plant has RMSE 6.759 deg with tau at its 0.05 h lower bound.

Expected outcome (documented in docs/CHANGES_2026-10-15.md): the
optimiser returns gamma to ~0 and RMSE is unchanged, i.e. a
theta-proportional straightening term in a single-angle model does not
repair the ZT9 two-phase shape. A negative result, retained because it
constrains the Model v2 design.

Run: python gamma_zt9_test.py
Writes results/gamma_zt9_test.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import fit_real
from data_real import curves, load_long
from sim import static_light
from sim_gamma import simulate as gamma_simulate

LO5 = np.concatenate([fit_real.LO, [0.0]])
HI5 = np.concatenate([fit_real.HI, [5.0]])


def fit_with_gamma(te, y, gamma_fixed=None):
    if gamma_fixed is None:
        x0 = np.array([5.0, 1.0, 30.0, 55.0, 0.5])

        def resid(x):
            ts, th = gamma_simulate(x[0], x[1], x[2], static_light(x[3]),
                                    float(te[-1]), gamma=x[4])
            return np.interp(te, ts, th) - y
        res = least_squares(resid, x0, bounds=(LO5, HI5), xtol=1e-6, ftol=1e-8)
        x = res.x
    else:
        x0 = np.array([5.0, 1.0, 30.0, 55.0])

        def resid(x):
            ts, th = gamma_simulate(x[0], x[1], x[2], static_light(x[3]),
                                    float(te[-1]), gamma=gamma_fixed)
            return np.interp(te, ts, th) - y
        res = least_squares(resid, x0, bounds=(fit_real.LO, fit_real.HI),
                            xtol=1e-6, ftol=1e-8)
        x = np.concatenate([res.x, [gamma_fixed]])
    rmse = float(np.sqrt(np.mean(res.fun ** 2)))
    return x, rmse


def main():
    df = load_long(sheet="Fig1C")
    t, y = curves(df)["Fig1C/ZT9-1"]
    te = t - t[0]
    rows = []
    for g in [0.0, 0.5, 1.0, 1.5, 2.0]:
        x, rmse = fit_with_gamma(te, y, gamma_fixed=g)
        rows.append({"gamma": g, "k": x[0], "tau": x[1], "max_rate": x[2],
                     "phi0": x[3], "rmse": rmse})
        print("gamma=%.1f  RMSE %.3f  (k=%.2f tau=%.3f max_rate=%.1f phi0=%.1f)"
              % (g, rmse, x[0], x[1], x[2], x[3]), flush=True)
    x, rmse = fit_with_gamma(te, y, gamma_fixed=None)
    rows.append({"gamma": float(x[4]), "k": x[0], "tau": x[1],
                 "max_rate": x[2], "phi0": x[3], "rmse": rmse})
    print("gamma FREE -> %.3f  RMSE %.3f" % (x[4], rmse), flush=True)
    pd.DataFrame(rows).to_csv("results/gamma_zt9_test.csv", index=False)
    print("saved results/gamma_zt9_test.csv")


if __name__ == "__main__":
    main()
