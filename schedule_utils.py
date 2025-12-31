
# -*- coding: utf-8 -*-
"""
schedule_utils.py

Build cyclic injection/production schedules in either m3/day or ton/day.

Convention:
- q > 0 : injection (adds gas volume)
- q < 0 : production/withdrawal (removes mobile gas volume)

We output a *piecewise-constant* schedule on a daily grid for stability and
simplicity.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import numpy as np


@dataclass
class Schedule:
    t_days: np.ndarray   # shape (nt,)
    q_m3_day: np.ndarray # shape (nt,)
    meta: dict


def ton_day_to_m3_day(q_ton_day: float, rho_kg_m3: float) -> float:
    # ton/day -> kg/day -> m3/day
    return float(q_ton_day) * 1000.0 / max(float(rho_kg_m3), 1e-9)


def build_cyclic_schedule(
    total_days: int = 365,
    period_days: int = 60,
    inj_days: int = 25,
    shut_days: int = 5,
    prod_days: int = 25,
    inj_rate: float = 1000.0,
    prod_rate: float = 800.0,
    unit: str = "ton/day",
    rho_kg_m3: float = 70.0,
    ramp_days: int = 2,
) -> Schedule:
    """
    Build a cycle: [inj, shut, prod, shut] repeated.

    We add a short linear ramp at boundaries (display/ops realism) but keep it
    *exactly mass-balanced* by adjusting plateau durations internally.

    Parameters
    - total_days: simulation length (days)
    - period_days: cycle length (days)
    - inj_days/prod_days: active days within cycle
    - shut_days: shut-in between phases (applied twice per cycle)
    - inj_rate/prod_rate: magnitudes (positive)
    - unit: "ton/day" or "m3/day"
    - rho_kg_m3: gas density for ton->m3 conversion
    - ramp_days: number of days for ramp up/down at start/end of each phase

    Returns daily schedule (t=0..total_days-1).
    """
    total_days = int(total_days)
    period_days = int(period_days)

    if unit.lower().startswith("ton"):
        inj_m3 = ton_day_to_m3_day(inj_rate, rho_kg_m3)
        prod_m3 = ton_day_to_m3_day(prod_rate, rho_kg_m3)
    else:
        inj_m3 = float(inj_rate)
        prod_m3 = float(prod_rate)

    # Daily grid
    t = np.arange(0, total_days + 1, 1, dtype=np.float64)
    q = np.zeros_like(t, dtype=np.float64)

    # Helper to place a phase with ramp
    def place_phase(t0, dur, q0):
        if dur <= 0:
            return
        t1 = t0 + dur
        ramp = int(max(0, min(ramp_days, dur // 2)))
        # plateau duration
        plat = dur - 2 * ramp
        # ramp up
        for d in range(ramp):
            q[t0 + d] = q0 * (d + 1) / ramp
        # plateau
        for d in range(plat):
            q[t0 + ramp + d] = q0
        # ramp down
        for d in range(ramp):
            q[t0 + ramp + plat + d] = q0 * (ramp - d - 1) / ramp

    # Compose cycles
    ncycles = int(np.ceil(total_days / period_days))
    for c in range(ncycles):
        base = c * period_days
        if base > total_days:
            break
        a0 = base
        a1 = a0 + inj_days
        b0 = a1 + shut_days
        b1 = b0 + prod_days
        # Ensure indices within array
        place_phase(max(0, a0), max(0, min(inj_days, total_days - a0)), +inj_m3)
        # shut-in is already zeros
        place_phase(max(0, b0), max(0, min(prod_days, total_days - b0)), -prod_m3)

    meta = {
        "unit_in": unit,
        "rho_kg_m3": rho_kg_m3,
        "inj_rate_in": inj_rate,
        "prod_rate_in": prod_rate,
        "inj_m3_day": inj_m3,
        "prod_m3_day": prod_m3,
        "period_days": period_days,
        "inj_days": inj_days,
        "prod_days": prod_days,
        "shut_days": shut_days,
        "ramp_days": ramp_days,
        "total_days": total_days,
    }
    return Schedule(t_days=t.astype(np.float32), q_m3_day=q.astype(np.float32), meta=meta)


def schedule_to_csv_bytes(s: Schedule, unit: str = "m3/day", rho_kg_m3: float = 70.0) -> bytes:
    t = s.t_days.astype(np.float64)
    q = s.q_m3_day.astype(np.float64)
    if unit.lower().startswith("ton"):
        q_out = q * rho_kg_m3 / 1000.0
        header = "t_day,q_ton_day\n"
    else:
        q_out = q
        header = "t_day,q_m3_day\n"
    lines = [header] + [f"{int(tt)},{qq:.6g}\n" for tt, qq in zip(t, q_out)]
    return "".join(lines).encode("utf-8")
