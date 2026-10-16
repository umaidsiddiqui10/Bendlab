#!/usr/bin/env python3
"""Inverse model v3: one any-hour network.

A single ensemble is trained on curves truncated at a random observation
window (4 to 24 h). Inputs are the 13 angle slots (unobserved slots set
to zero), the 12 successive differences, a 13-wide observation mask and
the observed fraction of the day, so one model answers "what can be
known by hour X" for every X instead of needing one model per window.

Population, split and noise protocol match v2 (100,000 plants, seed
20261016; each training plant appears once exact and once with noise up
to 1 deg) so results are comparable with inverse_model_v2_eval.csv.

Outputs:
  results/inverse_model_v3_weights.json
  results/inverse_model_v3_eval_by_hour.csv   (median errors per window)
  results/inverse_model_v3_eval.csv           (headline rows)
"""
from __future__ import annotations

import json
import re
import time

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from generate_plants import uniform_population
from sim import batch_simulate_static

TARGETS = ["k", "tau", "max_rate", "phi0"]
N_PLANTS = 100_000
SEED_POP = 20261016
CUTS = list(range(3, 14))  # 3..13 readings = 4..24 h


def simulate_pop(pop: np.ndarray) -> np.ndarray:
    _, th = batch_simulate_static(pop[:, 0], pop[:, 1], pop[:, 2], pop[:, 3],
                                  24.0, record_dt=2.0)
    return th.T


def add_noise(curves: np.ndarray, rng: np.random.Generator,
              sigma_max: float = 1.0) -> np.ndarray:
    sigma = rng.uniform(0.0, sigma_max, size=(len(curves), 1))
    noisy = curves + rng.normal(0.0, 1.0, size=curves.shape) * sigma
    noisy[:, 0] = 0.0
    return noisy


def feats_any(curves: np.ndarray, npts: np.ndarray) -> np.ndarray:
    n = len(curves)
    mask = (np.arange(13)[None, :] < npts[:, None]).astype(float)
    angles = curves * mask
    dmask = (np.arange(12)[None, :] < (npts[:, None] - 1)).astype(float)
    diffs = np.diff(curves, axis=1) * dmask
    hours = ((npts - 1) * 2.0 / 24.0).reshape(n, 1)
    return np.hstack([angles, diffs, mask, hours])


def y_transform(y: np.ndarray) -> np.ndarray:
    return np.column_stack([np.log(y[:, 0]), np.log(y[:, 1]),
                            np.log(y[:, 2]), y[:, 3] / 90.0])


def y_inverse(yt: np.ndarray) -> np.ndarray:
    return np.column_stack([np.exp(yt[:, 0]), np.exp(yt[:, 1]),
                            np.exp(yt[:, 2]), yt[:, 3] * 90.0])


def main() -> None:
    t0 = time.time()
    pop = uniform_population(N_PLANTS, seed=SEED_POP)
    print(f"simulating {N_PLANTS} plants...", flush=True)
    curves = simulate_pop(pop)
    rng = np.random.default_rng(7)
    idx = rng.permutation(N_PLANTS)
    te, tr = idx[:12000], idx[12000:]
    curves_train = np.vstack([curves[tr],
                              add_noise(curves[tr], np.random.default_rng(11))])
    y_train = np.vstack([pop[tr], pop[tr]])
    npts_train = np.random.default_rng(31).integers(3, 14, size=len(curves_train))
    print(f"simulation done ({time.time()-t0:.0f}s); training...", flush=True)

    x_scaler = StandardScaler().fit(feats_any(curves_train, npts_train))
    y_scaler = StandardScaler().fit(y_transform(y_train))
    X = x_scaler.transform(feats_any(curves_train, npts_train))
    Y = y_scaler.transform(y_transform(y_train))

    models = []
    for s in (0, 1, 2):
        m = MLPRegressor(hidden_layer_sizes=(128, 128), activation="relu",
                         solver="adam", alpha=1e-4, batch_size=512,
                         learning_rate_init=1e-3, max_iter=300,
                         early_stopping=True, validation_fraction=0.08,
                         n_iter_no_change=25, random_state=s)
        m.fit(X, Y)
        models.append(m)
        print(f"  seed={s}: {m.n_iter_} epochs", flush=True)

    def predict(curves_q: np.ndarray, n_pts: int) -> np.ndarray:
        npts = np.full(len(curves_q), n_pts, dtype=int)
        Xq = x_scaler.transform(feats_any(curves_q, npts))
        pr = np.stack([y_inverse(y_scaler.inverse_transform(m.predict(Xq)))
                       for m in models])
        return pr.mean(axis=0)

    hour_rows = []
    for n_pts in CUTS:
        pred = predict(curves[te], n_pts)
        rel = np.abs(pred - pop[te]) / pop[te]
        row = {"hours_observed": int((n_pts - 1) * 2), "n_readings": n_pts,
               "n": len(te)}
        for j, t in enumerate(TARGETS):
            row[f"{t}_median_rel_err"] = round(float(np.median(rel[:, j])), 4)
        row["phi0_median_abs_err_deg"] = round(
            float(np.median(np.abs(pred[:, 3] - pop[te][:, 3]))), 3)
        hour_rows.append(row)
        print(row, flush=True)
    pd.DataFrame(hour_rows).to_csv("results/inverse_model_v3_eval_by_hour.csv",
                                   index=False)

    rows = []
    def ev(name, curves_q, truth, n_pts=13):
        pred = predict(curves_q, n_pts)
        rel = np.abs(pred - truth) / truth
        row = {"split": name, "n": len(truth)}
        for j, t in enumerate(TARGETS):
            row[f"{t}_median_rel_err"] = round(float(np.median(rel[:, j])), 4)
        row["phi0_median_abs_err_deg"] = round(
            float(np.median(np.abs(pred[:, 3] - truth[:, 3]))), 3)
        rows.append(row)
        print(name, row, flush=True)

    ev("full_heldout_clean", curves[te], pop[te], 13)
    noisy1 = curves[te] + np.random.default_rng(21).normal(0, 1.0, curves[te].shape)
    noisy1[:, 0] = 0.0
    ev("full_heldout_noise1deg", noisy1, pop[te], 13)
    ev("at12h_heldout_clean", curves[te], pop[te], 7)

    prompts = pd.read_csv("results/adaption_test_prompts.csv")
    truth10 = pd.read_csv("results/adaption_test_truth.csv")
    curves10 = np.array([[float(v) for v in re.search(r"\[([^\]]+)\]", p).group(1).split(",")]
                         for p in prompts.prompt])
    ev("full_adaption_test10", curves10, truth10[TARGETS].to_numpy(float), 13)

    pop9 = uniform_population(2000, seed=999)
    ev("full_fresh_seed999", simulate_pop(pop9), pop9, 13)
    pd.DataFrame(rows).to_csv("results/inverse_model_v3_eval.csv", index=False)

    export = {"x_mean": x_scaler.mean_.tolist(), "x_scale": x_scaler.scale_.tolist(),
              "y_mean": y_scaler.mean_.tolist(), "y_scale": y_scaler.scale_.tolist(),
              "models": [{"coefs": [c.tolist() for c in m.coefs_],
                          "intercepts": [b.tolist() for b in m.intercepts_]}
                         for m in models]}
    with open("results/inverse_model_v3_weights.json", "w") as fh:
        json.dump(export, fh)
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
