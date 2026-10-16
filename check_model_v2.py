"""Regression checks for the Model v2 candidate simulators.

Verifies, for a small grid of parameter sets, that:
  1. sim_gamma.simulate with gamma = 0 reproduces sim.simulate exactly;
  2. sim_clock_gamma.simulate with a = 0, gamma = 0 reproduces
     sim.simulate exactly;
  3. sim_clock_gamma.batch_simulate agrees with simulate under static
     light;
  4. with the ZT9-1 published clock parameters, sim_clock_gamma
     reproduces the validated RMSE of 3.204 deg (and 6.759 deg for the
     official static fit), anchoring the refactor to the numbers in
     clock_validation.py.

Run: python check_model_v2.py   (exit code 0 = all checks pass)
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import sim
import sim_clock_gamma
import sim_gamma
from data_real import curves, load_long
from sim import static_light

CASES = [(10.0, 2.0, 30.0, 50.0), (25.0, 0.5, 8.0, 35.0), (5.0, 6.0, 100.0, 70.0),
         (40.77, 4.37, 7.39, 45.3)]  # last: ZT3-8, the cap-binding fit


def main():
    ok = True
    for k, tau, mr, phi in CASES:
        _, ref = sim.simulate(k, tau, mr, static_light(phi), 24.0)
        _, g0 = sim_gamma.simulate(k, tau, mr, static_light(phi), 24.0, gamma=0.0)
        _, c0 = sim_clock_gamma.simulate(k, tau, mr, static_light(phi), 24.0,
                                         gamma=0.0, a=0.0)
        d1 = float(np.abs(ref - g0).max())
        d2 = float(np.abs(ref - c0).max())
        print("k=%g tau=%g: gamma0 diff %.2e | clock0 diff %.2e" % (k, tau, d1, d2))
        ok &= (d1 == 0.0 and d2 == 0.0)

    # batch vs single, clock active
    ks = np.array([c[0] for c in CASES])
    taus = np.array([c[1] for c in CASES])
    mrs = np.array([c[2] for c in CASES])
    phis = np.array([c[3] for c in CASES])
    times, batch = sim_clock_gamma.batch_simulate_static(
        ks, taus, mrs, phis, 24.0, record_dt=0.5,
        gamma=0.3, a=0.5, phase=8.0, ZT_start=9.0)
    _, single = sim_clock_gamma.simulate(
        ks[0], taus[0], mrs[0], static_light(phis[0]), 24.0,
        gamma=0.3, a=0.5, phase=8.0, ZT_start=9.0)
    single_rec = single[::30]  # dt=1/60 -> every 0.5 h
    d3 = float(np.abs(batch[:, 0] - single_rec).max())
    print("batch vs single (clock on): max diff %.2e" % d3)
    ok &= d3 < 1e-9

    # anchor to validated ZT9-1 numbers
    df = load_long(sheet="Fig1C")
    t, y = curves(df)["Fig1C/ZT9-1"]
    te = t - t[0]

    def rmse(k, tau, mr, phi, a, phase):
        ts, th = sim_clock_gamma.simulate(k, tau, mr, static_light(phi),
                                          float(te[-1]), gamma=0.0, a=a,
                                          phase=phase, ZT_start=9.0)
        return float(np.sqrt(np.mean((np.interp(te, ts, th) - y) ** 2)))

    r_clock = rmse(4.74, 0.10, 75.3, 56.4, 0.90, 7.6)
    r_static = rmse(4.53, 0.05, 46.57, 55.06, 0.0, 0.0)
    print("ZT9-1 anchors: clock %.3f (expect 3.204) | static %.3f (expect 6.759)"
          % (r_clock, r_static))
    ok &= abs(r_clock - 3.204) < 0.01 and abs(r_static - 6.759) < 0.01

    print("ALL CHECKS PASSED" if ok else "CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
