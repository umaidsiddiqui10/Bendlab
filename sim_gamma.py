"""BendLab simulator with a proprioceptive (autostraightening) term.

Model v2 candidate A, following the proprioception term of Bastien et al.
(2012), reduced to a single tip angle:

    tau * dr/dt = k * sin(phi - theta) - gamma * theta - r
    r is clipped to [-max_rate, max_rate];  dtheta/dt = r.

gamma (per hour) is the proprioceptive sensitivity: the larger theta, the
stronger the straightening stimulus. gamma = 0 (default) reproduces sim.py
exactly; the regression check in check_model_v2.py verifies this to
machine precision.

Theta is in degrees and gamma * theta is in degrees/hour, so gamma carries
the same units as k. The dimensionless bending number of Bastien et al. is
B = k * L / gamma for a stem of length L; this model has no length scale,
so only the ratio k / gamma is meaningful here.

API mirrors sim.py: simulate() for one plant, batch_simulate() for a
population stepped in lockstep (light_fn(t, phi0) convention as in sim.py).
"""

from __future__ import annotations

import numpy as np

DEFAULT_DT = 1.0 / 60.0


def _check(tau, dt):
    if tau <= 0:
        raise ValueError("tau must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")


def simulate(k: float, tau: float, max_rate: float, phi_fn, t_end: float,
             dt: float = DEFAULT_DT, gamma: float = 0.0):
    """Simulate one plant. phi_fn(t) returns the light angle in degrees."""
    _check(tau, dt)
    n = int(round(t_end / dt))
    times = np.linspace(0.0, n * dt, n + 1)
    thetas = np.empty(n + 1)
    theta, r = 0.0, 0.0
    thetas[0] = theta
    for i in range(n):
        stimulus = (k * np.sin(np.deg2rad(phi_fn(times[i]) - theta))
                    - gamma * theta)
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
                   theta0=None, r0=None, gamma=0.0):
    """Vectorised population simulation; mirrors sim.batch_simulate.

    gamma may be a scalar or a per-plant array.
    """
    k = np.asarray(k, float)
    tau = np.asarray(tau, float)
    max_rate = np.asarray(max_rate, float)
    phi0 = np.asarray(phi0, float)
    gamma = np.asarray(gamma, float)
    if gamma.size == 1:
        gamma = np.full_like(k, float(gamma))
    if np.any(tau <= 0):
        raise ValueError("tau must be positive for all plants")
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
        stimulus = (k * (np.sin(phi_rad) * np.cos(theta_rad)
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


def batch_simulate_static(k, tau, max_rate, phi0, t_end, dt=DEFAULT_DT,
                          record_dt=None, gamma=0.0):
    """Static-light convenience wrapper around batch_simulate."""
    return batch_simulate(k, tau, max_rate, phi0, None, t_end, dt, record_dt,
                          gamma=gamma)


def static_light(phi0: float):
    """Constant light angle (as in sim.py)."""
    return lambda t: phi0
