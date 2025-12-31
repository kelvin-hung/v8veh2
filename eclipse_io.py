
# -*- coding: utf-8 -*-
"""
eclipse_io.py

Robust, pure-Python helpers to extract PORO/PERM and (optionally) thickness from
Eclipse-style decks.

Key features
- Works with Streamlit Cloud uploads (no filesystem paths).
- Resolves INCLUDE statements using uploaded companion files by basename.
- Handles INCLUDE where the filename is on the next line.
- Handles quoted INCLUDE paths and Windows-style paths.
- Supports GRDECL & INCLUDE arrays (PORO, PERMX, PERMY, PERMZ, TOPS, ZCORN).

Limitations
- This parser is intentionally lightweight: it reads numeric keyword arrays and
  basic grid geometry (SPECGRID, COORD, ZCORN). It does not interpret full
  scheduling, PVT, etc.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Iterable
import io
import os
import re
import numpy as np


# -----------------------------
# Utilities
# -----------------------------
_FLOAT_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eEdD][+-]?\d+)?$")


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and ((s[0] == "'" and s[-1] == "'") or (s[0] == '"' and s[-1] == '"')):
        return s[1:-1].strip()
    return s


def _basename_any(path_like: str) -> str:
    p = _strip_quotes(path_like.strip())
    p = p.replace("\\", "/")
    return os.path.basename(p)


def _clean_line(line: str) -> str:
    # Eclipse comments: '--' or '*' or '#' often used; keep it conservative.
    # Remove everything after '--' and after '#'
    if "--" in line:
        line = line.split("--", 1)[0]
    if "#" in line:
        line = line.split("#", 1)[0]
    return line.rstrip("\n")


def _tokenize_lines(text: str) -> List[str]:
    # Keep lines, but strip comments and whitespace; preserve blank lines.
    return [_clean_line(l) for l in text.splitlines()]


def _read_text_bytes(b: bytes) -> str:
    # Try utf-8 first; fall back to latin-1.
    try:
        return b.decode("utf-8")
    except Exception:
        return b.decode("latin-1", errors="ignore")


def _is_end_slash(line: str) -> bool:
    return line.strip() == "/"


def _split_words(line: str) -> List[str]:
    # Split on whitespace, but keep quoted tokens.
    # Example: INCLUDE 'SPE10MODEL2_TOPS.INC'
    out = []
    cur = ""
    inq = None
    for ch in line.strip():
        if inq is None:
            if ch in ("'", '"'):
                inq = ch
                cur += ch
            elif ch.isspace():
                if cur:
                    out.append(cur)
                    cur = ""
            else:
                cur += ch
        else:
            cur += ch
            if ch == inq:
                inq = None
    if cur:
        out.append(cur)
    return out


def _expand_repetition_token(tok: str) -> Optional[Tuple[int, float]]:
    """
    Eclipse repetition format: '10*1.23' or '100*0' or '3*1e-3'
    Returns (count, value) or None if not repetition.
    """
    if "*" not in tok:
        return None
    parts = tok.split("*", 1)
    if len(parts) != 2:
        return None
    left, right = parts[0].strip(), parts[1].strip()
    if not left.isdigit():
        return None
    try:
        v = float(right.replace("D", "E").replace("d", "e"))
    except Exception:
        return None
    return int(left), v


def _parse_floats_from_tokens(tokens: List[str]) -> List[float]:
    vals: List[float] = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        rep = _expand_repetition_token(tok)
        if rep is not None:
            n, v = rep
            vals.extend([v] * n)
            continue
        # Some decks use commas
        tok = tok.strip(",")
        # Ignore stray slashes in same line
        if tok == "/":
            continue
        # Some files include 'F' flags etc; treat as non-numeric
        if not _FLOAT_RE.match(tok.replace("D", "E").replace("d", "e")):
            continue
        vals.append(float(tok.replace("D", "E").replace("d", "e")))
    return vals


# -----------------------------
# Core parsing (keywords)
# -----------------------------
_KEYWORDS_OF_INTEREST = {
    "SPECGRID",
    "DIMENS",
    "COORD",
    "ZCORN",
    "PORO",
    "PERMX",
    "PERMY",
    "PERMZ",
    "TOPS",
    "DZ",
}


@dataclass
class EclipseGridArrays:
    nx: int = 0
    ny: int = 0
    nz: int = 0
    coord: Optional[np.ndarray] = None  # flat
    zcorn: Optional[np.ndarray] = None  # flat
    poro: Optional[np.ndarray] = None   # flat (nx*ny*nz)
    permx: Optional[np.ndarray] = None
    permy: Optional[np.ndarray] = None
    permz: Optional[np.ndarray] = None
    tops: Optional[np.ndarray] = None   # flat (nx*ny) or (nx*ny*nz) depending
    dz: Optional[np.ndarray] = None     # flat (nx*ny*nz)


def _parse_keyword_block(lines: List[str], start_idx: int) -> Tuple[str, List[float], int]:
    """
    Given lines where lines[start_idx] begins with a keyword,
    read subsequent numeric values until a '/' line.
    Returns (keyword, values, next_idx_after_block).
    """
    first = lines[start_idx].strip()
    key = _split_words(first)[0].upper()
    vals: List[float] = []

    i = start_idx + 1
    # Sometimes keyword line also contains numbers
    toks = _split_words(first)[1:]
    vals.extend(_parse_floats_from_tokens(toks))

    while i < len(lines):
        line = lines[i]
        if _is_end_slash(line):
            i += 1
            break
        toks = _split_words(line)
        vals.extend(_parse_floats_from_tokens(toks))
        i += 1
    return key, vals, i


def _resolve_include_target(lines: List[str], i: int) -> Tuple[str, int]:
    """
    Resolve INCLUDE statement at line i.
    Returns (basename, next_line_index_after_include_statement).

    Handles:
    - INCLUDE file.inc
    - INCLUDE 'file.inc'
    - INCLUDE   (no path) then next non-empty line is filename
    """
    line = lines[i].strip()
    words = _split_words(line)
    if len(words) >= 2:
        tgt = _basename_any(words[1])
        return tgt, i + 1

    # No filename on same line -> look ahead
    j = i + 1
    while j < len(lines):
        cand = lines[j].strip()
        if not cand:
            j += 1
            continue
        cand = _strip_quotes(cand)
        # If line is '/', that's not a filename
        if cand == "/":
            j += 1
            continue
        tgt = _basename_any(cand)
        return tgt, j + 1

    raise ValueError(f"INCLUDE without path near line {i+1}")


def _collect_files_from_streamlit_uploads(uploads: List[object]) -> Dict[str, bytes]:
    """
    uploads: list of Streamlit UploadedFile objects (or anything with .name and .read()).
    Returns dict basename->bytes
    """
    fs: Dict[str, bytes] = {}
    for up in uploads:
        name = getattr(up, "name", None) or "uploaded"
        b = up.read()
        fs[_basename_any(name)] = b
    return fs


def _read_virtual_file(fs: Dict[str, bytes], name: str) -> str:
    b = fs.get(_basename_any(name))
    if b is None:
        raise FileNotFoundError(f"Missing included file: {name}")
    return _read_text_bytes(b)


def parse_eclipse_deck_from_uploads(uploads: List[object]) -> EclipseGridArrays:
    """
    Parse Eclipse deck (DATA/GRDECL/INC files) from Streamlit uploads.
    At minimum, provide either:
      - a .GRDECL file containing PORO & PERMX, or
      - a .DATA deck with INCLUDE references and the referenced files uploaded too.

    Returns EclipseGridArrays with flattened arrays.
    """
    if not uploads:
        raise ValueError("No files uploaded.")

    fs = _collect_files_from_streamlit_uploads(uploads)

    # Find a primary entry file: prefer .DATA, else .GRDECL, else any.
    names = list(fs.keys())
    data_candidates = [n for n in names if n.lower().endswith(".data")]
    grdecl_candidates = [n for n in names if n.lower().endswith(".grdecl")]

    if data_candidates:
        entry = sorted(data_candidates)[0]
    elif grdecl_candidates:
        entry = sorted(grdecl_candidates)[0]
    else:
        entry = names[0]

    # BFS through includes, parsing keyword blocks in every visited file.
    visited = set()
    queue = [entry]

    grid = EclipseGridArrays()

    def ingest_file(text: str):
        nonlocal grid
        lines = _tokenize_lines(text)
        i = 0
        while i < len(lines):
            raw = lines[i].strip()
            if not raw:
                i += 1
                continue

            w0 = _split_words(raw)[0].upper() if _split_words(raw) else ""
            if w0 == "INCLUDE":
                tgt, i2 = _resolve_include_target(lines, i)
                if tgt not in visited and tgt not in queue:
                    queue.append(tgt)
                i = i2
                continue

            if w0 in _KEYWORDS_OF_INTEREST:
                key, vals, i2 = _parse_keyword_block(lines, i)

                if key in {"SPECGRID", "DIMENS"}:
                    # SPECGRID: NX NY NZ 1 F /
                    # DIMENS:  NX NY NZ /
                    if len(vals) >= 3:
                        grid.nx, grid.ny, grid.nz = int(vals[0]), int(vals[1]), int(vals[2])
                elif key == "COORD":
                    grid.coord = np.asarray(vals, dtype=np.float64)
                elif key == "ZCORN":
                    grid.zcorn = np.asarray(vals, dtype=np.float64)
                elif key == "PORO":
                    grid.poro = np.asarray(vals, dtype=np.float64)
                elif key == "PERMX":
                    grid.permx = np.asarray(vals, dtype=np.float64)
                elif key == "PERMY":
                    grid.permy = np.asarray(vals, dtype=np.float64)
                elif key == "PERMZ":
                    grid.permz = np.asarray(vals, dtype=np.float64)
                elif key == "TOPS":
                    grid.tops = np.asarray(vals, dtype=np.float64)
                elif key == "DZ":
                    grid.dz = np.asarray(vals, dtype=np.float64)

                i = i2
                continue

            i += 1

    while queue:
        name = queue.pop(0)
        if name in visited:
            continue
        visited.add(name)
        text = _read_virtual_file(fs, name)
        ingest_file(text)

    if grid.nx <= 0 or grid.ny <= 0:
        # Some GRDECL omit SPECGRID; infer from PORO length if possible
        if grid.poro is not None:
            # Try assume nz=1
            n = int(grid.poro.size)
            # Common SPE10: 60x220x?? ; if uploaded only 2D, allow
            grid.nx, grid.ny, grid.nz = n, 1, 1

    # Basic validations
    if grid.poro is None or grid.permx is None:
        raise ValueError("Could not find PORO and PERMX in uploaded deck/GRDECL/INC files.")

    return grid


# -----------------------------
# Geometry: thickness from ZCORN
# -----------------------------
def zcorn_to_dz(nx: int, ny: int, nz: int, zcorn_flat: np.ndarray) -> np.ndarray:
    """
    Convert Eclipse ZCORN (flat) into per-cell thickness dz (nx, ny, nz).
    Eclipse corner-point layout can be reshaped to (2*nz, 2*ny, 2*nx).
    Thickness computed as mean(bottom 4 corners) - mean(top 4 corners).
    """
    z = np.asarray(zcorn_flat, dtype=np.float64)
    expected = 8 * nx * ny * nz
    if z.size < expected:
        raise ValueError(f"ZCORN length {z.size} < expected {expected} for nx,ny,nz={nx},{ny},{nz}")
    z = z[:expected]
    # Reshape per Eclipse convention
    Z = z.reshape((2 * nz, 2 * ny, 2 * nx))
    dz = np.zeros((nz, ny, nx), dtype=np.float64)
    for k in range(nz):
        top = Z[2 * k, :, :]
        bot = Z[2 * k + 1, :, :]
        # For each cell (i,j), its 4 top corners are at indices (2i,2j),(2i,2j+1),(2i+1,2j),(2i+1,2j+1)
        # Here axes are (2*ny,2*nx) = (y,x)
        for j in range(ny):
            y0 = 2 * j
            y1 = y0 + 2
            for i in range(nx):
                x0 = 2 * i
                x1 = x0 + 2
                zt = float(np.mean(top[y0:y1, x0:x1]))
                zb = float(np.mean(bot[y0:y1, x0:x1]))
                # Depth sign conventions differ across decks (positive-down vs negative-down).
                # Use absolute thickness to be robust.
                dz[k, j, i] = abs(zb - zt)
    # Return in (nx,ny,nz) ordering used elsewhere
    return np.transpose(dz, (2, 1, 0))


# -----------------------------
# Public API used by Streamlit app
# -----------------------------
def build_2d_fields_from_eclipse_uploads(
    uploads: List[object],
    layer_mode: str = "all",
    k0: int = 0,
    k1: Optional[int] = None,
    upscale_mode: str = "trans_weighted",
    min_active_phi: float = 1e-8,
    default_dz_m: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Returns (phi2d, k2d, mask2d, meta)

    layer_mode:
      - 'all' : use all layers
      - 'range' : use layers [k0, k1) (0-based)
      - 'single' : use layer k0 only

    upscale_mode:
      - 'layer_avg' : arithmetic mean over chosen layers
      - 'trans_weighted' : thickness-weighted mean using dz (if available)
    """
    grid = parse_eclipse_deck_from_uploads(uploads)
    nx, ny, nz = int(grid.nx), int(grid.ny), int(grid.nz)

    poro = np.asarray(grid.poro, dtype=np.float64)
    permx = np.asarray(grid.permx, dtype=np.float64)

    # Reshape to (nx, ny, nz) if possible
    ncell = nx * ny * nz
    if poro.size >= ncell:
        poro = poro[:ncell].reshape((nz, ny, nx))  # (k,j,i)
    else:
        raise ValueError(f"PORO length {poro.size} does not match nx*ny*nz={ncell}")

    if permx.size >= ncell:
        permx = permx[:ncell].reshape((nz, ny, nx))
    else:
        raise ValueError(f"PERMX length {permx.size} does not match nx*ny*nz={ncell}")

    # dz
    dz = None
    if grid.dz is not None and grid.dz.size >= ncell:
        dz = np.asarray(grid.dz[:ncell], dtype=np.float64).reshape((nz, ny, nx))
    elif grid.zcorn is not None and nx > 0 and ny > 0 and nz > 0:
        try:
            dz_xyz = zcorn_to_dz(nx, ny, nz, grid.zcorn)  # (nx,ny,nz)
            dz = np.transpose(dz_xyz, (2, 1, 0))  # to (nz,ny,nx)
        except Exception:
            dz = None

    if dz is None:
        dz = np.full((nz, ny, nx), float(default_dz_m), dtype=np.float64)

    # Select layers
    if layer_mode == "single":
        kk0 = int(k0)
        kk1 = kk0 + 1
    elif layer_mode == "range":
        kk0 = int(k0)
        kk1 = int(k1) if k1 is not None else nz
    else:
        kk0, kk1 = 0, nz

    kk0 = max(0, min(nz - 1, kk0))
    kk1 = max(kk0 + 1, min(nz, kk1))

    poro_sel = poro[kk0:kk1]
    permx_sel = permx[kk0:kk1]
    dz_sel = dz[kk0:kk1]

    # Active mask per layer
    active = (poro_sel > min_active_phi) & np.isfinite(permx_sel) & (permx_sel > 0)
    # Replace inactive with nan for averaging
    poro_sel = np.where(active, poro_sel, np.nan)
    permx_sel = np.where(active, permx_sel, np.nan)
    dz_sel = np.where(active, dz_sel, np.nan)

    if upscale_mode == "layer_avg":
        phi2d = np.nanmean(poro_sel, axis=0)
        k2d = np.nanmean(permx_sel, axis=0)
        H = np.nanmean(dz_sel, axis=0) * (kk1 - kk0)  # approximate total thickness
    else:
        w = dz_sel
        wsum = np.nansum(w, axis=0)
        phi2d = np.nansum(poro_sel * w, axis=0) / np.maximum(wsum, 1e-30)
        k2d = np.nansum(permx_sel * w, axis=0) / np.maximum(wsum, 1e-30)
        H = wsum

    # 2D in (nx,ny) expected by solver -> transpose from (ny,nx)
    phi2d = np.transpose(phi2d, (1, 0)).astype(np.float32)  # (nx,ny)
    k2d = np.transpose(k2d, (1, 0)).astype(np.float32)
    H2d = np.transpose(H, (1, 0)).astype(np.float32)

    mask2d = np.isfinite(phi2d) & np.isfinite(k2d) & (phi2d > min_active_phi) & (k2d > 0)

    meta = {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "layer_mode": layer_mode,
        "k0": kk0,
        "k1": kk1,
        "upscale_mode": upscale_mode,
        "H_m": H2d,
    }
    return phi2d, k2d, mask2d.astype(np.uint8), meta
