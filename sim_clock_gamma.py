"""BendLab simulator with circadian gating and an optional proprioceptive term.

Model v2 candidate B. The light sensitivity is modulated by a 24 h clock,
following the circadian gating reported by Atamian et al. (2016), and the
proprioceptive term of Bastien et al. (2012) can be added on the same
stimulus:

    k_eff(t)  = k0 * (1 + a * cos(2 * pi * (ZT - phase) / 24))
    tau * dr/dt = k_eff * sin(phi - theta) - gamma * theta - r
    r is clipped to [-max_rate, max_rate];  dtheta/dt = r.

ZT = ZT_start + t is absolute circadian time, where ZT_start is the
circadian time at stimulus onset for the plant being simulated.

With a = 0 and gamma = 0 (defaults) this reproduces sim.py exactly; the
regression check in check_model_v2.py verifies this, and
clock_validation.py evaluates the candidate against held-out data.

Parameters
----------
k0 : base light sensitivity (degrees/hour; the stimulus is a target bend
    rate, so k carries rate units as in sim.py).
tau : sensing lag (hours).
max_rate : bend-rate limit (degrees/hour).
phi0 : light angle (degrees).
gamma : proprioceptive sensitivity (per hour), as in sim_gamma.py.
a : clock amplitude, 0 <= a <= 1 (fractional modulation of k0).
phase : circadian time (hours) at which k_eff is maximal.
ZT_start : circadian time at t = 0 for each plant.

API mirrors sim.py: simulate() for one plant (phi_fn(t) convention),
batch_simulate() for a population (light_fn(t, phi0) convention).
k_eff is floored at 0.01 degrees/hour to keep the stimulus non-negative
under strong modulation.
"""

from __future__ import annotations

import numpy as np

DEFAULT_DT = 1.0 / 60.0
K_EFF_FLOOR = 0.01


def _check(tau, dt):
    if tau <= 0:
        raise ValueError("tau must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")


def _k_eff(k0, a, phase, zt):
    return np.maximum(k0 * (1.0 + a * np.cos(2.0 * np.pi * (zt - phase) / 24.0)),
                      K_EFF_FLOOR)


def simulate(k0: float, tau: float, max_rate: float, phi_fn, t_end: float,
             dt: float = DEFAULT_DT, gamma: float = 0.0,
             a: float = 0.0, phase: float = 0.0, ZT_start: float = 0.0):
    """Simulate one plant. phi_fn(t) returns the light angle in degrees."""
    _check(tau, dt)
    n = int(round(t_end / dt))
    times = np.linspace(0.0, n * dt, n + 1)
    thetas = np.empty(n + 1)
    theta, r = 0.0, 0.0
    thetas[0] = theta
    for i in range(n):
        t = i * dt
        k_eff = _k_eff(k0, a, phase, ZT_start + t)
        stimulus = k_eff * np.sin(np.deg2rad(phi_fn(t) - theta)) - gamma * theta
        r += (stimulus - r) * (dt / tau)
        if r > max_rate:
            r = max_rate
        elif r < -max_rate:
            r = -max_rate
        theta += r * dt
        thetas[i + 1] = theta
    return times, thetas


def _broadcast(x, n, name):
    x = np.asarray(x, float)
    if x.size == 1:
        return np.full(n, float(x))
    if x.size != n:
        raise ValueError("%s must be scalar or length %d" % (name, n))
    return x


def batch_simulate(k0, tau, max_rate, phi0, light_fn=None, t_end: float = 24.0,
                   dt: float = DEFAULT_DT, record_dt: float | None = None,
                   theta0=None, r0=None, gamma=0.0, a=0.0, phase=0.0,
                   ZT_start=0.0):
    """Vectorised population simulation; mirrors sim.batch_simulate.

    gamma, a, phase and ZT_start may be scalars or per-plant arrays.
    """
    k0 = np.asarray(k0, float)
    tau = np.asarray(tau, float)
    max_rate = np.asarray(max_rate, float)
    phi0 = np.asarray(phi0, float)
    n_plants = len(k0)
    gamma = _broadcast(gamma, n_plants, "gamma")
    a = _broadcast(a, n_plants, "a")
    phase = _broadcast(phase, n_plants, "phase")
    ZT_start = _broadcast(ZT_start, n_plants, "ZT_start")
    if np.any(tau <= 0):
        raise ValueError("tau must be positive for all plants")
    if light_fn is None:
        light_fn = lambda t, p0: p0
    n = int(round(t_end / dt))
    rec = max(1, int(round(record_dt / dt))) if record_dt else 1
    thetas = np.empty((n // rec + 1, n_plants))
    theta = np.zeros(n_plants) if theta0 is None else np.array(theta0, float).copy()
    r = np.zeros(n_plants) if r0 is None else np.array(r0, float).copy()
    thetas[0] = theta
    j = 1
    for i in range(n):
        t = i * dt
        k_eff = _k_eff(k0, a, phase, ZT_start + t)
        phi_rad = np.deg2rad(light_fn(t, phi0))
        theta_rad = np.deg2rad(theta)
        stimulus = (k_eff * (np.sin(phi_rad) * np.cos(theta_rad)
                             - np.cos(phi_rad) * np.sin(theta_rad))
                    - gamma * theta)
        r += (stimulus - r) * (dt / tau)
        np.clip(r, -max_rate, max_rate, out=r)
        theta += r * dt
        if (i + 1) % rec == 0:
            thetas[j] = theta
            j += 1
    times = np.arange(thetas.shape[0]) * (dt * rec)
    return times, thetas


def batch_simulate_static(k0, tau, max_rate, phi0, t_end, dt=DEFAULT_DT,
                          record_dt=None, gamma=0.0, a=0.0, phase=0.0,
                          ZT_start=0.0):
    """Static-light convenience wrapper around batch_simulate."""
    return batch_simulate(k0, tau, max_rate, phi0, None, t_end, dt, record_dt,
                          gamma=gamma, a=a, phase=phase, ZT_start=ZT_start)


def static_light(phi0: float):
    """Constant light angle (as in sim.py)."""
    return lambda t: phi0
