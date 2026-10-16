#!/usr/bin/env python3
"""Inverse model v2: bigger, noise-robust, with an early-readout model.

Changes from v1 (train_inverse_model.py):
  * 100,000 freshly simulated plants (seed 20261016) instead of 17,000.
  * Training curves carry per-plant measurement noise (sigma uniform in
    0 to 1.5 deg, first reading kept at 0) so the network sees data that
    resembles measurements, not only perfect simulator output.
  * Two model families are exported:
      - "full":  13 readings over 24 h, ensemble of 3 MLPs (96, 96)
      - "early": first 7 readings over 12 h, ensemble of 2 MLPs (64, 64)
    The early model quantifies how much can be known halfway through
    the experiment, a question BendLab's design analysis also asks.

Outputs:
  results/inverse_model_v2_weights.json
  results/inverse_model_v2_eval.csv
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


def simulate_pop(pop: np.ndarray) -> np.ndarray:
    _, th = batch_simulate_static(pop[:, 0], pop[:, 1], pop[:, 2], pop[:, 3],
                                  24.0, record_dt=2.0)
    return th.T  # (n, 13)


def add_noise(curves: np.ndarray, rng: np.random.Generator,
              sigma_max: float = 1.5) -> np.ndarray:
    sigma = rng.uniform(0.0, sigma_max, size=(len(curves), 1))
    noisy = curves + rng.normal(0.0, 1.0, size=curves.shape) * sigma
    noisy[:, 0] = 0.0
    return noisy


def feats(curves: np.ndarray, n_pts: int) -> np.ndarray:
    c = curves[:, :n_pts]
    return np.hstack([c, np.diff(c, axis=1)])


def y_transform(y: np.ndarray) -> np.ndarray:
    return np.column_stack([np.log(y[:, 0]), np.log(y[:, 1]),
                            np.log(y[:, 2]), y[:, 3] / 90.0])


def y_inverse(yt: np.ndarray) -> np.ndarray:
    return np.column_stack([np.exp(yt[:, 0]), np.exp(yt[:, 1]),
                            np.exp(yt[:, 2]), yt[:, 3] * 90.0])


def train_family(curves_tr, y_tr, n_pts, hidden, seeds):
    xs = StandardScaler().fit(feats(curves_tr, n_pts))
    ys = StandardScaler().fit(y_transform(y_tr))
    X = xs.transform(feats(curves_tr, n_pts))
    Y = ys.transform(y_transform(y_tr))
    models = []
    for s in seeds:
        m = MLPRegressor(hidden_layer_sizes=hidden, activation="relu",
                         solver="adam", alpha=1e-4, batch_size=512,
                         learning_rate_init=1e-3, max_iter=300,
                         early_stopping=True, validation_fraction=0.08,
                         n_iter_no_change=25, random_state=s)
        m.fit(X, Y)
        models.append(m)
        print(f"  n_pts={n_pts} seed={s}: {m.n_iter_} epochs", flush=True)

    def predict(curves):
        Xq = xs.transform(feats(curves, n_pts))
        pr = np.stack([y_inverse(ys.inverse_transform(m.predict(Xq)))
                       for m in models])
        return pr.mean(axis=0)

    export = {"x_mean": xs.mean_.tolist(), "x_scale": xs.scale_.tolist(),
              "y_mean": ys.mean_.tolist(), "y_scale": ys.scale_.tolist(),
              "models": [{"coefs": [c.tolist() for c in m.coefs_],
                          "intercepts": [b.tolist() for b in m.intercepts_]}
                         for m in models]}
    return predict, export


def main() -> None:
    t0 = time.time()
    pop = uniform_population(N_PLANTS, seed=SEED_POP)
    print(f"simulating {N_PLANTS} plants...", flush=True)
    curves = simulate_pop(pop)
    rng = np.random.default_rng(7)
    idx = rng.permutation(N_PLANTS)
    te, tr = idx[:12000], idx[12000:]
    curves_tr_clean = curves[tr]
    curves_tr_noisy = add_noise(curves[tr], np.random.default_rng(11), sigma_max=1.0)
    curves_train = np.vstack([curves_tr_clean, curves_tr_noisy])
    y_train = np.vstack([pop[tr], pop[tr]])
    print(f"simulation done ({time.time()-t0:.0f}s); training...", flush=True)

    predict_full, exp_full = train_family(curves_train, y_train, 13,
                                          (96, 96), (0, 1, 2))
    predict_early, exp_early = train_family(curves_train, y_train, 7,
                                            (64, 64), (0, 1))

    rows = []

    def ev(name, predict, curves_q, truth):
        pred = predict(curves_q)
        rel = np.abs(pred - truth) / truth
        row = {"split": name, "n": len(truth)}
        for j, t in enumerate(TARGETS):
            row[f"{t}_median_rel_err"] = round(float(np.median(rel[:, j])), 4)
        row["phi0_median_abs_err_deg"] = round(
            float(np.median(np.abs(pred[:, 3] - truth[:, 3]))), 3)
        rows.append(row)
        print(name, row, flush=True)

    ev("full_heldout_clean", predict_full, curves[te], pop[te])
    rng_ev = np.random.default_rng(21)
    noisy1 = curves[te] + rng_ev.normal(0, 1.0, curves[te].shape)
    noisy1[:, 0] = 0.0
    noisy_te = curves[te] + rng_ev.normal(0, 2.0, curves[te].shape)
    noisy_te[:, 0] = 0.0
    ev("full_heldout_noise1deg", predict_full, noisy1, pop[te])
    ev("full_heldout_noise2deg", predict_full, noisy_te, pop[te])
    ev("early12_heldout_clean", predict_early, curves[te], pop[te])
    ev("early12_heldout_noise2deg", predict_early, noisy_te, pop[te])

    prompts = pd.read_csv("results/adaption_test_prompts.csv")
    truth10 = pd.read_csv("results/adaption_test_truth.csv")
    curves10 = np.array([[float(v) for v in re.search(r"\[([^\]]+)\]", p).group(1).split(",")]
                         for p in prompts.prompt])
    ev("full_adaption_test10", predict_full, curves10, truth10[TARGETS].to_numpy(float))

    pop9 = uniform_population(2000, seed=999)
    ev("full_fresh_seed999", predict_full, simulate_pop(pop9), pop9)

    pd.DataFrame(rows).to_csv("results/inverse_model_v2_eval.csv", index=False)
    with open("results/inverse_model_v2_weights.json", "w") as fh:
        json.dump({"full": exp_full, "early": exp_early}, fh)
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
