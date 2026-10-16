"""Phototropic bending simulator.

Model: first-order tropic drive with a sensing lag and a saturating rate.

    tau * dr/dt = k * sin(phi(t) - theta) - r        (rate relaxes to stimulus)
    r_clipped   = clip(r, -max_rate, +max_rate)      (saturation)
    dtheta/dt   = r_clipped

Two states: theta (stem angle, degrees) and r (bend rate, degrees/hour).
k is in degrees/hour (the stimulus is expressed as a target bend rate,
so k carries the rate's units), tau and t in hours, max_rate in degrees/hour.

Why two states: with r alone proportional to the stimulus the plant bends
fastest at t=0 and only ever decelerates. Measured curves start flat, speed
up, then slow (sigmoid onset) — that is what tau produces, and it makes tau a
real parameter instead of a rescaling of k.

`simulate` is a pure function of (params, phi_fn, dt, t_end): no randomness
anywhere, so a rerun is bit-identical.
"""

from __future__ import annotations

import numpy as np

DEFAULT_DT = 1.0 / 60.0  # one minute


def simulate(k: float, tau: float, max_rate: float, phi_fn, t_end: float,
             dt: float = DEFAULT_DT) -> tuple[np.ndarray, np.ndarray]:
    """Integrate one plant under a time-varying light angle.

    phi_fn(t) returns the light angle in degrees at time t, so regimes
    (static, flip, sweep) are plain callables.

    Returns (times, thetas), both length int(t_end/dt) + 1.
    """
    n = int(round(t_end / dt))
    times = np.linspace(0.0, n * dt, n + 1)
    thetas = np.empty(n + 1)
    theta, r = 0.0, 0.0
    thetas[0] = theta
    for i in range(n):
        stimulus = k * np.sin(np.deg2rad(phi_fn(times[i]) - theta))
        r += (stimulus - r) * (dt / tau)
        if r > max_rate:
            r = max_rate
        elif r < -max_rate:
            r = -max_rate
        theta += r * dt
        thetas[i + 1] = theta
    return times, thetas


def batch_simulate(k, tau, max_rate, phi0, light_fn=None, t_end: float = 24.0,
                   dt: float = DEFAULT_DT, record_dt: float | None = None,
                   theta0=None, r0=None):
    """Vectorised simulation for a whole population, stepped in lockstep.

    k, tau, max_rate, phi0 are 1-D arrays of equal length. light_fn(t, phi0)
    returns the per-plant light angle at time t; the default is static light
    at each plant's own phi0. Initial states theta0/r0 default to zero.

    Integration is the same scheme as `simulate`, and `simulate` and this
    function agree to machine precision under static light.

    Returns (times, thetas) with thetas shape (n_records, n_plants).
    """
    k = np.asarray(k, float)
    tau = np.asarray(tau, float)
    max_rate = np.asarray(max_rate, float)
    phi0 = np.asarray(phi0, float)
    if light_fn is None:
        light_fn = lambda t, p0: p0
    n = int(round(t_end / dt))
    rec = max(1, int(round(record_dt / dt))) if record_dt else 1
    thetas = np.empty((n // rec + 1, len(k)))
    theta = np.zeros(len(k)) if theta0 is None else np.array(theta0, float).copy()
    r = np.zeros(len(k)) if r0 is None else np.array(r0, float).copy()
    thetas[0] = theta
    j = 1
    for i in range(n):
        phi_rad = np.deg2rad(light_fn(i * dt, phi0))
        theta_rad = np.deg2rad(theta)
        stimulus = k * (np.sin(phi_rad) * np.cos(theta_rad)
                        - np.cos(phi_rad) * np.sin(theta_rad))
        r += (stimulus - r) * (dt / tau)
        np.clip(r, -max_rate, max_rate, out=r)
        theta += r * dt
        if (i + 1) % rec == 0:
            thetas[j] = theta
            j += 1
    times = np.arange(thetas.shape[0]) * (dt * rec)
    return times, thetas


def batch_simulate_static(k, tau, max_rate, phi0, t_end: float,
                          dt: float = DEFAULT_DT, record_dt: float | None = None):
    """Static-light convenience wrapper around `batch_simulate`."""
    return batch_simulate(k, tau, max_rate, phi0, None, t_end, dt, record_dt)


def static_light(phi0: float):
    """Constant light angle."""
    def phi(t):
        return phi0
    return phi


def flip_light(phi0: float, phi1: float, t_flip: float):
    """Constant light that steps to a new angle at t_flip."""
    def phi(t):
        return phi0 if t < t_flip else phi1
    return phi


def sweep_light(phi_start: float, phi_end: float, day_len: float):
    """Light sweeping linearly across the arc each day, resetting overnight."""
    def phi(t):
        frac = (t % day_len) / day_len
        return phi_start + (phi_end - phi_start) * frac
    return phi


def sweep_with_night_reset(phi_start: float, phi_end: float, day_len: float,
                           night_len: float, return_rate: float):
    """Daytime sweep plus an overnight return sweep.

    Young sunflowers reorient eastward overnight (Atamian et al. 2016), so the
    reset is not instantaneous. return_rate = 1.0 returns over the full night.
    """
    period = day_len + night_len

    def phi(t):
        tc = t % period
        if tc < day_len:
            return phi_start + (phi_end - phi_start) * (tc / day_len)
        frac = min((tc - day_len) / night_len * return_rate, 1.0)
        return phi_end + (phi_start - phi_end) * frac
    return phi


def subsample(times: np.ndarray, thetas: np.ndarray, out_dt: float
              ) -> tuple[np.ndarray, np.ndarray]:
    """Report at a coarser cadence, e.g. to match half-hourly measurements."""
    idx = np.arange(0, len(times), max(1, int(round(out_dt / (times[1] - times[0])))))
    if idx[-1] != len(times) - 1:
        idx = np.append(idx, len(times) - 1)
    return times[idx], thetas[idx]
