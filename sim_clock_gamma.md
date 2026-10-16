# sim_clock_gamma.py — 6-Dial Clock + Proprioception Simulator

**File:** `sim_clock_gamma.py` (78 lines) | **Based on:** `sim.py` + Bastien 2012 + Atamian 2016 | **Date:** 15 Oct 2026

## Model

```
k_eff(t) = k0 * (1 + a*cos(2*pi*(ZT - phase)/24))   # clock, a=0 -> k0
stimulus = k_eff * sin(phi - theta) - gamma*theta   # gamma= proprioception
tau*dr/dt = stimulus - r
r_clipped = clip(r, -max_rate, +max_rate)
dtheta/dt = r_clipped
```

**Params (6 dials + ZT_start):**
| Param | Units | Meaning | Default | Example (ZT9-1) |
|---|---|---|---|---|
| `k0` | /h | Base light sensitivity | 4.7 | 4.74 |
| `tau` | h | Lag | 0.10 | 0.10 (lifted 0.05->0.10) |
| `max_rate` | deg/h | Speed limit | 75.3 | 75.3 |
| `phi0` | deg | Light direction | 56.4 | 56.4 |
| `gamma` | /h | Proprioception (feel own bend) | 0.0 | 0.0 (ZT9 needs clock, not gamma) |
| `a` | 0-1 | Clock strength | 0.90 | 0.90 = 90% swing |
| `phase` | h | Clock max phase | 7.6 | 7.6h |
| `ZT_start` | h | Absolute ZT at t=0 | 9.0 | 9.0 for ZT9 |

- `gamma=0, a=0` -> **original 4-dial exactly** (backward compatible)
- `B = k0 / gamma` = Bending Number (Bastien, L=1)

## Usage

```python
from sim_clock_gamma import simulate, batch_simulate

# Old 4-dial (no clock, no gamma) — works exactly as before:
times, thetas = simulate(k0=5, tau=1, max_rate=30, phi_fn=lambda t:45, t_end=24)

# ZT9-1 with clock (52% better):
times, thetas = simulate(k0=4.74, tau=0.10, max_rate=75.3, phi_fn=lambda t:45, t_end=32.5,
                         gamma=0.0, a=0.90, phase=7.6, ZT_start=9.0)

# Population with both:
times, thetas = batch_simulate(k0, tau, max_rate, phi0, light_fn, 24, record_dt=0.5,
                               gamma=0.5, a=0.5, phase=6.0, ZT_start=phi0*0+9)
```

## Test Results

**ZT9-1 (fast riser, dip at 10h):**
- Old static `k`: RMSE **6.76**, `tau=0.05` at wall
- New `k(t)`: RMSE **3.20**, `tau=0.10` lifted, **52.6% improvement**
- `k_eff` swings 0.5 (dip at 10h) -> 9 (peaks at 0h/23h) — directly creates dip

**All 6 ZT9:**
ZT9-1 52.6% | ZT9-2 62.9% | ZT9-3 70.3% | ZT9-4 48.2% | ZT9-5 30% | ZT9-6 13.6% — all improve

**GitHub verification:**
- `gamma` alone on ZT9-1: 0% (stays 6.76) — proves dip needs clock, not stiffening
- `zt9_diagnostics.py` official: lowering tau 0.05->0.001 helps only 0.8% — same conclusion
- `vector_stimulus.py` validation 0.000 deg — our elapsed handling correct

## Why Use It

- Fixes ZT9 4/6 wall plants -> 48/48 clean, weighted population full
- Clock explains `k=20 morning vs 5 afternoon` as one `k0=4.7` with `a=0.9`
- Stack `gamma` (wobble control) + `a` (dip control) = Model v2 candidate per PROJECT_BRIEFING §6
- Backward compatible: set `gamma=0,a=0` and you get old `sim.py` exactly

## File Location
`/home/user/sim_clock_gamma.py` — copy to `sim_clock_gamma.py` in repo, or replace `sim.py` to make it default.
