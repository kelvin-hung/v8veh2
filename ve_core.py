
# -*- coding: utf-8 -*-
"""
ve_core.py

Height-based VE finite-volume solver (2D map) with:
- Explicit FV with CFL substepping
- Land trapping (simple hysteresis)
- Optional 9-point diffusion (reduces grid-aligned artifacts)
- Flux limiting (Van Leer) on face gradients (helps with checkerboard-ish oscillations)
- Mass-conserving core update; any smoothing is display-only

This is a pragmatic VE proxy meant for interactive screening, not a replacement
for full 3D compositional simulation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np


DEFAULT_PARAMS = {
    # grid
    "dx_m": 25.0,
    "dy_m": 25.0,

    # physics-ish knobs (dimensionless / scaled)
    "D0": 0.20,          # base diffusion-like spreading coefficient [m^2/day] (effective)
    "mob_exp": 0.5,      # k scaling exponent in D = D0*(k/kref)^mob_exp
    "anisD": 1.0,        # y-direction diffusivity multiplier

    # saturation / trapping
    "Sgr_max": 0.25,     # max trapped gas saturation
    "Land_C": 1.0,       # Land hysteresis constant (higher -> less trapping)
    "Sg_min": 0.0,
    "Sg_max": 1.0,

    # numerics
    "cfl": 0.45,
    "use_9pt": True,     # include diagonal diffusion
    "use_limiter": True,
}


@dataclass
class VEResult:
    t_days: np.ndarray                 # (nt,)
    q_m3_day: np.ndarray               # (nt,)
    h_list: Optional[np.ndarray]       # (nt, nx, ny)
    sg_list: Optional[np.ndarray]      # (nt, nx, ny)
    p_list: Optional[np.ndarray]       # (nt, nx, ny)
    plume_area_m2: np.ndarray          # (nt,)
    eq_radius_m: np.ndarray            # (nt,)
    mass_m3: np.ndarray                # (nt,)
    meta: dict


def _harmonic(a: np.ndarray, b: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    return 2.0 * a * b / (a + b + eps)


def _vanleer(r: np.ndarray) -> np.ndarray:
    # Van Leer limiter
    return (r + np.abs(r)) / (1.0 + np.abs(r) + 1e-30)



def _face_flux_limited(h: np.ndarray, D: np.ndarray, dx: float, axis: int) -> np.ndarray:
    """
    Compute a *conservative* diffusive flux divergence using face fluxes (TPFA),
    with an optional Van-Leer style limiter applied to the face gradient to
    suppress odd-even/checkerboard artifacts.

    Returns:
        div: array same shape as h, where div = -∇·( -D ∇h ) = ∇·(D ∇h)
    """
    eps = 1e-12

    if axis == 0:  # i-direction (rows)
        # Face gradients g at i+1/2
        g = (h[1:, :] - h[:-1, :]) / dx                  # (nx-1, ny)
        Df = _harmonic(D[:-1, :], D[1:, :])              # (nx-1, ny)

        # Build previous-face gradient with padding (same shape as g)
        g_prev = np.vstack([g[:1, :], g[:-1, :]])        # (nx-1, ny)

        # Van Leer limiter on gradient ratio
        r = g_prev / (g + eps)
        phi = (r + np.abs(r)) / (1.0 + np.abs(r) + eps)  # in [0,2]
        g_lim = phi * g

        F = -Df * g_lim                                  # diffusive face flux (nx-1, ny)

        div = np.zeros_like(h)
        div[:-1, :] += F / dx
        div[1:, :]  -= F / dx
        return div

    elif axis == 1:  # j-direction (cols)
        g = (h[:, 1:] - h[:, :-1]) / dx                  # (nx, ny-1)
        Df = _harmonic(D[:, :-1], D[:, 1:])              # (nx, ny-1)

        g_prev = np.hstack([g[:, :1], g[:, :-1]])        # (nx, ny-1)

        r = g_prev / (g + eps)
        phi = (r + np.abs(r)) / (1.0 + np.abs(r) + eps)
        g_lim = phi * g

        F = -Df * g_lim                                  # (nx, ny-1)

        div = np.zeros_like(h)
        div[:, :-1] += F / dx
        div[:, 1:]  -= F / dx
        return div

    else:
        raise ValueError("axis must be 0 or 1")



def _diag_diffusion_9pt(h: np.ndarray, D: np.ndarray, dx: float) -> np.ndarray:
    """
    Conservative diagonal diffusion using 4 diagonal neighbors.
    Uses a reduced conductance to keep isotropy-ish.
    """
    # conductance for diagonals ~ 1/sqrt(2) distance -> factor
    dd = dx * np.sqrt(2.0)
    out = np.zeros_like(h)

    # NE-SW
    Dne = _harmonic(D[:-1, :-1], D[1:, 1:])
    grad = (h[1:, 1:] - h[:-1, :-1]) / dd
    F = -Dne * grad
    out[:-1, :-1] += F / dd
    out[1:, 1:]   -= F / dd

    # NW-SE
    Dnw = _harmonic(D[1:, :-1], D[:-1, 1:])
    grad = (h[:-1, 1:] - h[1:, :-1]) / dd
    F = -Dnw * grad
    out[1:, :-1] += F / dd
    out[:-1, 1:] -= F / dd

    # Slightly reduce diagonal strength to avoid oversmoothing
    return 0.5 * out


def _safe_box_smooth_2d(x: np.ndarray, k: int = 3, weight: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Box smoothing via integral image; safe for odd/even k.
    Display-only helper (NOT used in solver).
    """
    if x is None:
        return None
    if k <= 1:
        return x
    k = int(k)
    pad = k // 2
    # pad with edge values
    xp = np.pad(x, ((pad, pad), (pad, pad)), mode="edge").astype(np.float64)
    if weight is None:
        wp = np.ones_like(xp)
    else:
        wp = np.pad(weight, ((pad, pad), (pad, pad)), mode="edge").astype(np.float64)

    # integral images (prefix sums). IMPORTANT:
    # We add a leading zero row/col so window sums are always well-defined.
    c = np.cumsum(np.cumsum(xp * wp, axis=0), axis=1)
    w = np.cumsum(np.cumsum(wp, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)), mode="constant", constant_values=0.0)
    w = np.pad(w, ((1, 0), (1, 0)), mode="constant", constant_values=0.0)

    # Window sum for each output cell (same shape as x).
    # With pad=k//2 and xp being edge-padded, the kxk window corresponding to
    # x[i,j] is xp[i:i+k, j:j+k]. Using integral image, sum = S(i+k,j+k)-S(i,j+k)-S(i+k,j)+S(i,j).
    nx, ny = x.shape
    num = (
        c[k : k + nx, k : k + ny]
        - c[0:nx, k : k + ny]
        - c[k : k + nx, 0:ny]
        + c[0:nx, 0:ny]
    )
    den = (
        w[k : k + nx, k : k + ny]
        - w[0:nx, k : k + ny]
        - w[k : k + nx, 0:ny]
        + w[0:nx, 0:ny]
    )

    return (num / np.maximum(den, 1e-30)).astype(x.dtype)


def choose_well_ij(mask: np.ndarray) -> Tuple[int, int]:
    """Choose a default well location: center of active cells."""
    idx = np.argwhere(mask > 0)
    if idx.size == 0:
        return mask.shape[0] // 2, mask.shape[1] // 2
    ci = int(np.round(np.mean(idx[:, 0])))
    cj = int(np.round(np.mean(idx[:, 1])))
    return ci, cj


def run_ve_height_fv(
    phi: np.ndarray,
    k: np.ndarray,
    H: np.ndarray,
    mask: np.ndarray,
    t_days: np.ndarray,
    q_m3_day: np.ndarray,
    well_ij: Tuple[int, int],
    params: Optional[Dict] = None,
    return_fields: bool = True,
) -> VEResult:
    """
    Main forward simulator.

    Inputs
    - phi, k, H: (nx, ny) float arrays
    - mask: (nx, ny) bool/uint8 active map
    - t_days: (nt,) days (monotone)
    - q_m3_day: (nt,) piecewise-constant on same grid
    - well_ij: injector/producer cell (i,j)
    - params: dict (overrides DEFAULT_PARAMS)
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)

    nx, ny = phi.shape
    dx = float(p["dx_m"])
    dy = float(p["dy_m"])

    phi = phi.astype(np.float64)
    k = k.astype(np.float64)
    H = H.astype(np.float64)
    m = mask.astype(bool)

    # Normalize k for diffusion scaling
    kref = np.nanmedian(k[m]) if np.any(m) else 1.0
    kref = max(float(kref), 1e-12)
    D = float(p["D0"]) * np.power(np.maximum(k / kref, 1e-12), float(p["mob_exp"]))
    Dy = float(p["anisD"]) * D

    # State: gas thickness h (m) and hysteresis tracking
    h = np.zeros((nx, ny), dtype=np.float64)
    sg_max_hist = np.zeros_like(h)
    sgr = np.zeros_like(h)

    # Storage
    nt = int(len(t_days))
    if return_fields:
        h_out = np.zeros((nt, nx, ny), dtype=np.float32)
        sg_out = np.zeros((nt, nx, ny), dtype=np.float32)
        p_out = np.zeros((nt, nx, ny), dtype=np.float32)
    else:
        h_out = sg_out = p_out = None

    area = dx * dy
    plume_area = np.zeros(nt, dtype=np.float64)
    eq_r = np.zeros(nt, dtype=np.float64)
    mass = np.zeros(nt, dtype=np.float64)

    wi, wj = int(well_ij[0]), int(well_ij[1])
    wi = max(0, min(nx - 1, wi))
    wj = max(0, min(ny - 1, wj))

    # Helper: pressure surrogate for visualization
    def pressure_field(qval):
        # simple smooth kernel around well, scaled by q
        rr2 = (np.arange(nx)[:, None] - wi) ** 2 + (np.arange(ny)[None, :] - wj) ** 2
        sig2 = (0.08 * max(nx, ny)) ** 2
        amp = float(qval) / max(area, 1e-9)
        return amp * np.exp(-rr2 / (2.0 * sig2))

    # time stepping
    for it in range(nt):
        t0 = float(t_days[it])
        t1 = float(t_days[it + 1]) if it < nt - 1 else t0 + 1.0
        dt_big = max(t1 - t0, 1e-9)
        qval = float(q_m3_day[it])

        # CFL substepping based on max diffusivity
        Dmax = float(np.nanmax(D[m])) if np.any(m) else 1.0
        Dmaxy = float(np.nanmax(Dy[m])) if np.any(m) else 1.0
        dt_cfl = float(p["cfl"]) * min(dx * dx / (4.0 * max(Dmax, 1e-12)),
                                       dy * dy / (4.0 * max(Dmaxy, 1e-12)))
        nsub = int(np.ceil(dt_big / max(dt_cfl, 1e-9)))
        dt = dt_big / nsub

        for _ in range(nsub):
            # Source term -> thickness change at well
            if m[wi, wj]:
                dh_src = (qval * dt) / max(phi[wi, wj] * area, 1e-12)  # m of gas volume per pore area
            else:
                dh_src = 0.0

            # Diffusion update (conservative FV)
            if bool(p["use_limiter"]):
                divx = _face_flux_limited(h, D, dx, axis=0)
                divy = _face_flux_limited(h, Dy, dy, axis=1)
            else:
                divx = _face_flux_plain(h, D, dx, axis=0)
                divy = _face_flux_plain(h, Dy, dy, axis=1)

            div = divx + divy

            if bool(p["use_9pt"]):
                div += _diag_diffusion_9pt(h, 0.5 * (D + Dy), min(dx, dy))

            h_new = h + dt * div
            # Apply source at well
            h_new[wi, wj] += dh_src

            # Enforce boundaries / mask: no-flow outside active
            h_new[~m] = 0.0

            # Convert to saturation
            sg = np.clip(h_new / np.maximum(H, 1e-9), float(p["Sg_min"]), float(p["Sg_max"]))

            # Hysteresis / Land trapping (simple)
            sg_max_hist = np.maximum(sg_max_hist, sg)
            # Land-style trapped saturation target
            # Sgr = Sgr_max * Sg_max / (Sg_max + C*(1-Sg_max))
            C = float(p["Land_C"])
            Sgr_max = float(p["Sgr_max"])
            sgr_target = Sgr_max * sg_max_hist / np.maximum(sg_max_hist + C * (1.0 - sg_max_hist), 1e-9)
            # During withdrawal, prevent sg from dropping below sgr_target
            if qval < 0:
                sg = np.maximum(sg, sgr_target)
                h_new = sg * H
            sgr = sgr_target

            # Commit
            h = np.clip(h_new, 0.0, np.nanmax(H[m]) if np.any(m) else np.inf)

        # Diagnostics
        sg = np.clip(h / np.maximum(H, 1e-9), 0.0, 1.0)
        mobile = np.maximum(sg - sgr, 0.0)
        # Gas volume (mobile + trapped) in reservoir units (m^3) per cell: phi*area*h
        mass[it] = float(np.nansum(phi[m] * area * h[m]))
        plume_area[it] = float(np.sum((sg[m] > 0.02).astype(np.float64)) * area)
        eq_r[it] = float(np.sqrt(plume_area[it] / np.pi)) if plume_area[it] > 0 else 0.0

        if return_fields:
            h_out[it] = h.astype(np.float32)
            sg_out[it] = sg.astype(np.float32)
            p_out[it] = pressure_field(qval).astype(np.float32)

    meta = {
        "params": p,
        "well_ij": (wi, wj),
        "dx_m": dx,
        "dy_m": dy,
    }
    return VEResult(
        t_days=t_days.astype(np.float32),
        q_m3_day=q_m3_day.astype(np.float32),
        h_list=h_out,
        sg_list=sg_out,
        p_list=p_out,
        plume_area_m2=plume_area.astype(np.float32),
        eq_radius_m=eq_r.astype(np.float32),
        mass_m3=mass.astype(np.float32),
        meta=meta,
    )


# Export display-only smoother so app can call it safely
box_smooth_2d = _safe_box_smooth_2d
