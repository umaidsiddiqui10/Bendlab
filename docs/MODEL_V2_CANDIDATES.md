# Model v2 candidates: proprioception and circadian gating

Status: **candidates under evaluation; not adopted.** The reference model
remains `sim.py`. Both candidates reduce to the reference model exactly at
their default settings.

## Candidate A — proprioceptive term (`sim_gamma.py`)

```
tau * dr/dt = k * sin(phi - theta) - gamma * theta - r
```

Following Bastien et al. (2012), reduced to a single tip angle.
Evaluation on ZT9-1 (`gamma_zt9_test.py`): no improvement; the optimiser
returns gamma to ~0. See `docs/CHANGES_2026-10-15.md`, Section 1.

## Candidate B — circadian gating (`sim_clock_gamma.py`)

```
k_eff(t) = k0 * (1 + a * cos(2 * pi * (ZT - phase) / 24))
tau * dr/dt = k_eff * sin(phi - theta) - gamma * theta - r
```

Following the circadian gating reported by Atamian et al. (2016). The
gamma term can be combined with the clock in the same simulator.

Evaluation (`clock_validation.py`): a per-plant free clock improves all
six ZT9 fits in-sample (median RMSE 2.42 → 1.11 deg); a single shared
clock fitted without ZT9 (a = 0.47, phase = 8.14 h) improves all six
held-out ZT9 fits (median 2.42 → 1.44 deg), while slightly worsening
median fit on the training blocks (1.78 → 2.00 deg). See
`docs/CHANGES_2026-10-15.md`, Section 2, for per-plant values and the
outstanding tests required before adoption.

## Usage

```python
from sim_clock_gamma import simulate, batch_simulate, static_light

# Reference model (defaults reproduce sim.py exactly):
times, thetas = simulate(k0=5, tau=1, max_rate=30, phi_fn=static_light(45),
                         t_end=24)

# ZT9-1 with the published clock parameters (RMSE 3.204 deg):
times, thetas = simulate(k0=4.74, tau=0.10, max_rate=75.3,
                         phi_fn=static_light(56.4), t_end=32.5,
                         gamma=0.0, a=0.90, phase=7.6, ZT_start=9.0)

# Population under a shared clock:
times, thetas = batch_simulate(k0, tau, max_rate, phi0, None, 24.0,
                               record_dt=0.5, a=0.47, phase=8.14,
                               ZT_start=9.0)
```

Parameter units follow `sim.py`: k0 and max_rate in degrees/hour, tau in
hours, angles in degrees, a dimensionless, phase and ZT_start in hours.
