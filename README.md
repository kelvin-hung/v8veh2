# H2Plume FullFix VE Height-FV (Streamlit)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Uploading Eclipse files
Upload the `.DATA` file *together with* all referenced `.INC` / `.GRDECL` files.
The loader resolves `INCLUDE` by basename.

## Features
- Robust INCLUDE parsing (filename on same line or next line)
- ZCORN/DZ thickness support for transmissibility-weighted VE upscaling
- VE upscaling choices: layer-average vs thickness-weighted
- Height-based VE FV solver + CFL substepping + Land trapping
- Optional 9-point diffusion + flux limiter to reduce grid artifacts
- Optional display-only smoothing (does not affect conservation)

