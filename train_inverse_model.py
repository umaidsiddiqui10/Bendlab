#!/usr/bin/env python3
"""Train a neural inverse model for BendLab: bending curve -> parameters.

Trains an ensemble of small MLPs on simulated trajectories
(results/adaption_inverse_dataset.csv, uniform population, seed 777).
Targets are k, tau, max_rate, phi0. Ensemble spread is exported with the
weights so downstream tools can report model disagreement as a simple
uncertainty signal. Evaluation uses a held-out split, the 10-plant
Adaption test set (seed 31337) and a fresh simulated population
(seed 999) that no training touched.

Outputs:
  results/inverse_model_weights.json  (scalers + weights, JS-ready)
  results/inverse_model_eval.csv      (per-split error summary)
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

ANGLE_COLS = [f"angle_t{h:02d}h" for h in range(0, 26, 2)]
TARGETS = ["k", "tau", "max_rate", "phi0"]


def features(curves: np.ndarray) -> np.ndarray:
    """13 angles + 12 successive differences."""
    return np.hstack([curves, np.diff(curves, axis=1)])


def y_transform(y: np.ndarray) -> np.ndarray:
    return np.column_stack([np.log(y[:, 0]), np.log(y[:, 1]),
                            np.log(y[:, 2]), y[:, 3] / 90.0])


def y_inverse(yt: np.ndarray) -> np.ndarray:
    return np.column_stack([np.exp(yt[:, 0]), np.exp(yt[:, 1]),
                            np.exp(yt[:, 2]), yt[:, 3] * 90.0])


def make_model(seed: int) -> MLPRegressor:
    return MLPRegressor(hidden_layer_sizes=(80, 80), activation="relu",
                        solver="adam", alpha=1e-4, batch_size=256,
                        learning_rate_init=1e-3, max_iter=300,
                        early_stopping=True, validation_fraction=0.1,
                        n_iter_no_change=25, random_state=seed)


def main() -> None:
    t0 = time.time()
    df = pd.read_csv("results/adaption_inverse_dataset.csv")
    X_raw = df[ANGLE_COLS].to_numpy(float)
    y = df[TARGETS].to_numpy(float)
    rng = np.random.default_rng(123)
    idx = rng.permutation(len(df))
    n_test = int(0.15 * len(df))
    te, tr = idx[:n_test], idx[n_test:]

    x_scaler = StandardScaler().fit(features(X_raw[tr]))
    y_scaler = StandardScaler().fit(y_transform(y[tr]))

    Xtr = x_scaler.transform(features(X_raw[tr]))
    ytr = y_scaler.transform(y_transform(y[tr]))

    models = []
    for seed in (0, 1, 2):
        m = make_model(seed)
        m.fit(Xtr, ytr)
        models.append(m)
        print(f"model seed={seed}: {m.n_iter_} epochs, "
              f"train score={m.score(Xtr, ytr):.4f}", flush=True)

    def predict(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = x_scaler.transform(features(curves))
        preds = np.stack([y_inverse(y_scaler.inverse_transform(m.predict(X)))
                          for m in models])
        return preds.mean(axis=0), preds.std(axis=0)

    def evaluate(name: str, curves: np.ndarray, truth: np.ndarray) -> dict:
        pred, spread = predict(curves)
        rel = np.abs(pred - truth) / truth
        row = {"split": name, "n": len(truth)}
        for j, t in enumerate(TARGETS):
            row[f"{t}_median_rel_err"] = float(np.median(rel[:, j]))
        row["phi0_median_abs_err_deg"] = float(np.median(np.abs(pred[:, 3] - truth[:, 3])))
        row["max_rate_median_spread_rel"] = float(
            np.median(spread[:, 2] / np.maximum(pred[:, 2], 1e-9)))
        print(name, {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in row.items()}, flush=True)
        return row

    rows = [evaluate("heldout_split", X_raw[te], y[te])]

    prompts = pd.read_csv("results/adaption_test_prompts.csv")
    truth10 = pd.read_csv("results/adaption_test_truth.csv")
    curves10 = np.array([[float(v) for v in re.search(r"\[([^\]]+)\]", p).group(1).split(",")]
                         for p in prompts.prompt])
    rows.append(evaluate("adaption_test10", curves10, truth10[TARGETS].to_numpy(float)))

    pop = uniform_population(2000, seed=999)
    _, th = batch_simulate_static(pop[:, 0], pop[:, 1], pop[:, 2], pop[:, 3],
                                  24.0, record_dt=2.0)
    rows.append(evaluate("fresh_seed999", th.T, pop))

    pd.DataFrame(rows).to_csv("results/inverse_model_eval.csv", index=False)

    export = {
        "architecture": {"input": int(Xtr.shape[1]), "hidden": [80, 80], "output": 4},
        "x_mean": x_scaler.mean_.tolist(), "x_scale": x_scaler.scale_.tolist(),
        "y_mean": y_scaler.mean_.tolist(), "y_scale": y_scaler.scale_.tolist(),
        "models": [{"coefs": [c.tolist() for c in m.coefs_],
                    "intercepts": [b.tolist() for b in m.intercepts_]}
                   for m in models],
    }
    with open("results/inverse_model_weights.json", "w") as fh:
        json.dump(export, fh)
    print(f"exported weights ({sum(len(json.dumps(mdl)) for mdl in export['models'])//1024} KB of matrices)")
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
