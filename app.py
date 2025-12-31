
# -*- coding: utf-8 -*-
import io
import zipfile
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from eclipse_io import build_2d_fields_from_eclipse_uploads
from schedule_utils import build_cyclic_schedule, schedule_to_csv_bytes
from ve_core import run_ve_height_fv, choose_well_ij, box_smooth_2d

st.set_page_config(page_title="H2Plume VE (FullFix Height-FV)", layout="wide")

st.title("VE height-based FV simulator (FullFix) + Eclipse loader")
st.caption("Upload Eclipse deck/GRDECL/INC files, build a cyclic schedule (ton/day or m³/day), and run a height-based VE FV model with Land trapping + CFL.")

with st.sidebar:
    st.header("1) Upload Eclipse files")
    uploads = st.file_uploader(
        "Upload .DATA/.GRDECL/.INC files together (multi-select).",
        type=["DATA", "data", "GRDECL", "grdecl", "INC", "inc"],
        accept_multiple_files=True,
    )

    st.header("2) VE upscaling options")
    layer_mode = st.selectbox("Layer selection", ["all", "single", "range"], index=0)
    k0 = st.number_input("k0 (0-based)", min_value=0, value=0, step=1)
    k1 = st.number_input("k1 (exclusive, for range)", min_value=1, value=5, step=1)
    upscale_mode = st.selectbox("Upscaling", ["trans_weighted", "layer_avg"], index=0)
    default_dz = st.number_input("Default thickness dz (m) if ZCORN/DZ missing", min_value=0.01, value=1.0, step=0.1)

    st.header("3) Schedule (cyclic)")
    unit = st.selectbox("Schedule unit", ["ton/day", "m3/day"], index=0)
    rho = st.number_input("Gas density (kg/m³) for ton↔m³ conversion", min_value=0.1, value=70.0, step=1.0)

    total_days = st.number_input("Total days", min_value=30, value=365, step=10)
    period_days = st.number_input("Cycle period (days)", min_value=10, value=60, step=5)
    inj_days = st.number_input("Injection days per cycle", min_value=1, value=25, step=1)
    shut_days = st.number_input("Shut-in days between phases", min_value=0, value=5, step=1)
    prod_days = st.number_input("Withdrawal days per cycle", min_value=1, value=25, step=1)

    inj_rate = st.number_input(f"Injection rate ({unit})", min_value=0.0, value=1000.0, step=50.0)
    prod_rate = st.number_input(f"Withdrawal rate magnitude ({unit})", min_value=0.0, value=800.0, step=50.0)
    ramp_days = st.number_input("Ramp days (per phase)", min_value=0, value=2, step=1)

    st.header("4) Numerics / physics knobs")
    dx = st.number_input("dx (m)", min_value=1.0, value=25.0, step=1.0)
    dy = st.number_input("dy (m)", min_value=1.0, value=25.0, step=1.0)

    D0 = st.number_input("D0 (effective spreading, m²/day)", min_value=0.001, value=0.20, step=0.01)
    mob_exp = st.number_input("mob_exp (k scaling exponent)", min_value=0.0, value=0.5, step=0.1)
    anisD = st.number_input("anisD (y multiplier)", min_value=0.1, value=1.0, step=0.1)

    Sgr_max = st.number_input("Sgr_max", min_value=0.0, max_value=0.9, value=0.25, step=0.01)
    Land_C = st.number_input("Land_C", min_value=0.01, value=1.0, step=0.1)

    use_9pt = st.checkbox("9-point diffusion (reduces grid artifacts)", value=True)
    use_limiter = st.checkbox("Flux limiter (Van Leer)", value=True)
    cfl = st.number_input("CFL", min_value=0.05, max_value=0.9, value=0.45, step=0.05)

    st.header("Display")
    smooth_display = st.checkbox("Light post-filter (display only)", value=True)
    smooth_k = st.slider("Smoothing kernel k", min_value=1, max_value=11, value=5, step=2)

# Build schedule
sched = build_cyclic_schedule(
    total_days=int(total_days),
    period_days=int(period_days),
    inj_days=int(inj_days),
    shut_days=int(shut_days),
    prod_days=int(prod_days),
    inj_rate=float(inj_rate),
    prod_rate=float(prod_rate),
    unit=unit,
    rho_kg_m3=float(rho),
    ramp_days=int(ramp_days),
)

st.subheader("Schedule")
colA, colB = st.columns([2, 1])
with colA:
    fig = plt.figure()
    plt.plot(sched.t_days, sched.q_m3_day)
    plt.xlabel("t (days)")
    plt.ylabel("q (m³/day)  (+: inject, -: withdraw)")
    plt.title("q(t) used by simulator")
    st.pyplot(fig, clear_figure=True)

with colB:
    st.write("Download schedule")
    st.download_button(
        label="Download schedule (m³/day) CSV",
        data=schedule_to_csv_bytes(sched, unit="m3/day", rho_kg_m3=float(rho)),
        file_name="schedule_m3_day.csv",
        mime="text/csv",
    )
    st.download_button(
        label="Download schedule (ton/day) CSV",
        data=schedule_to_csv_bytes(sched, unit="ton/day", rho_kg_m3=float(rho)),
        file_name="schedule_ton_day.csv",
        mime="text/csv",
    )

st.divider()
st.subheader("Grid inputs (Eclipse)")

if not uploads:
    st.info("Upload your Eclipse files in the sidebar to run the VE simulator.")
    st.stop()

try:
    phi2d, k2d, mask2d, meta = build_2d_fields_from_eclipse_uploads(
        uploads=uploads,
        layer_mode=layer_mode,
        k0=int(k0),
        k1=int(k1),
        upscale_mode=upscale_mode,
        default_dz_m=float(default_dz),
    )
except Exception as e:
    st.error(f"Failed to read Eclipse files: {e}")
    st.stop()

H2d = meta["H_m"]

st.write(f"Parsed grid: nx={meta['nx']} ny={meta['ny']} nz={meta['nz']} | layers [{meta['k0']},{meta['k1']}) | upscaling={meta['upscale_mode']}")
st.write("H (thickness) statistics (m):", float(np.nanmin(H2d[mask2d > 0])), float(np.nanmedian(H2d[mask2d > 0])), float(np.nanmax(H2d[mask2d > 0])))

# Default well
wi, wj = choose_well_ij(mask2d)
st.write(f"Default well cell: (i={wi}, j={wj}) (center of active region)")

# Run model
st.subheader("Run VE height FV model")
run_btn = st.button("Run VE height FV model", type="primary")

if run_btn:
    params = dict(
        dx_m=float(dx),
        dy_m=float(dy),
        D0=float(D0),
        mob_exp=float(mob_exp),
        anisD=float(anisD),
        Sgr_max=float(Sgr_max),
        Land_C=float(Land_C),
        use_9pt=bool(use_9pt),
        use_limiter=bool(use_limiter),
        cfl=float(cfl),
    )

    with st.spinner("Running VE height FV model..."):
        res = run_ve_height_fv(
            phi=phi2d,
            k=k2d,
            H=H2d,
            mask=mask2d,
            t_days=sched.t_days,
            q_m3_day=sched.q_m3_day,
            well_ij=(wi, wj),
            params=params,
            return_fields=True,
        )

    st.success("Done.")

    # Slider for time index
    tidx = st.slider("t index", min_value=0, max_value=len(res.t_days) - 1, value=min(24, len(res.t_days) - 1), step=1)

    sg = res.sg_list[tidx]
    h = res.h_list[tidx]
    p = res.p_list[tidx]

    if smooth_display:
        # weight smoothing by mask to avoid bleeding outside active region
        w = (mask2d > 0).astype(np.float32)
        sg_show = box_smooth_2d(sg, k=int(smooth_k), weight=w)
        h_show = box_smooth_2d(h, k=int(smooth_k), weight=w)
        p_show = box_smooth_2d(p, k=int(smooth_k), weight=w)
    else:
        sg_show, h_show, p_show = sg, h, p

    c1, c2 = st.columns(2)
    with c1:
        fig1 = plt.figure()
        plt.imshow(sg_show.T, origin="lower", vmin=0, vmax=1)
        plt.title(f"Sg | tidx={tidx}")
        plt.xlabel("i")
        plt.ylabel("j")
        plt.colorbar()
        st.pyplot(fig1, clear_figure=True)
    with c2:
        fig2 = plt.figure()
        plt.imshow(h_show.T, origin="lower")
        plt.title(f"h (m) | tidx={tidx}")
        plt.xlabel("i")
        plt.ylabel("j")
        plt.colorbar()
        st.pyplot(fig2, clear_figure=True)

    fig3 = plt.figure()
    plt.imshow(p_show.T, origin="lower")
    plt.title(f"Pressure surrogate | tidx={tidx}")
    plt.xlabel("i")
    plt.ylabel("j")
    plt.colorbar()
    st.pyplot(fig3, clear_figure=True)

    # Time series
    c3, c4, c5 = st.columns(3)
    with c3:
        fig = plt.figure()
        plt.plot(res.t_days, res.plume_area_m2)
        plt.xlabel("t (days)")
        plt.ylabel("Plume area (m²)")
        st.pyplot(fig, clear_figure=True)
    with c4:
        fig = plt.figure()
        plt.plot(res.t_days, res.eq_radius_m)
        plt.xlabel("t (days)")
        plt.ylabel("Equivalent radius (m)")
        st.pyplot(fig, clear_figure=True)
    with c5:
        fig = plt.figure()
        plt.plot(res.t_days, res.mass_m3)
        plt.xlabel("t (days)")
        plt.ylabel("Gas volume in pores (m³)")
        st.pyplot(fig, clear_figure=True)

    # Export results NPZ
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        t_days=res.t_days,
        q_m3_day=res.q_m3_day,
        h=res.h_list,
        sg=res.sg_list,
        p=res.p_list,
        plume_area_m2=res.plume_area_m2,
        eq_radius_m=res.eq_radius_m,
        mass_m3=res.mass_m3,
        meta=str(res.meta),
    )
    st.download_button("Download results (.npz)", data=buf.getvalue(), file_name="ve_height_results.npz", mime="application/octet-stream")
