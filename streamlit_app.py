"""
## Streamlit demo: local JWST MIRI catalog + 3-panel preview

- **Data:** reads only `fits_images/<target_name>/*.fits` (no network).
- **Science image:** `jwst_fits_preview.read_sci_2d` / `percentile_stretch_u8` — same
  module the notebook can import for parity with `Working_draft.ipynb`.
- **Filter column:** parsed from filenames (`_miri_f770w`, etc.); MAST table columns are
  not used in this app.
- **Target description:** loads `{target}_target_description.json` next to the false-color
  PNG; if missing and `OPENAI_API_KEY` is set, generates once and saves the same file.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from astropy.io import fits

import numpy as np
from jwst_fits_preview import percentile_stretch_u8, read_sci_2d
from target_llm import (
    fetch_strict_target_blurb,
    load_target_description,
    save_target_description,
    target_description_path,
)

st.set_page_config(page_title="Local JWST FITS", layout="wide")

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
FITS_ROOT = PROJECT_ROOT / "fits_images"

# -----------------------------------------------------------------------------
# Triptych bands (same three as `target_filters` in Working_draft.ipynb)
# -----------------------------------------------------------------------------

BANDS: tuple[str, ...] = ("F770W", "F1500W", "F2550W")


def _filter_guess(filename: str) -> str:
    # e.g. ..._miri_f770w-sub128_... or ..._miri_f1500w_i2d.fits
    m = re.search(r"_miri_f(\d+w)(?:_|-|\.)", filename.lower())
    return f"F{m.group(1).upper()}" if m else ""


def _row_for_file(path: Path, target: str) -> dict:
    h0 = fits.getheader(str(path), 0)
    try:
        hsci = fits.getheader(str(path), "SCI")
        xposure = hsci.get("XPOSURE")
    except Exception:
        xposure = None
    size_mb = path.stat().st_size / (1024 * 1024)
    return {
        "target": target,
        "file": path.name,
        "filter": _filter_guess(path.name),
        "telescope": h0.get("TELESCOP", ""),
        "file_date": h0.get("DATE", ""),
        "xposure_s": xposure,
        "size_mb": round(size_mb, 2),
        "rel_path": str(path.relative_to(FITS_ROOT.parent)).replace("\\", "/"),
    }


@st.cache_data(show_spinner=False)
def load_catalog() -> pd.DataFrame:
    rows: list[dict] = []
    if not FITS_ROOT.is_dir():
        return pd.DataFrame()
    for target_dir in sorted(p for p in FITS_ROOT.iterdir() if p.is_dir()):
        for fpath in sorted(target_dir.glob("*.fits")):
            try:
                rows.append(_row_for_file(fpath, target_dir.name))
            except Exception:
                rows.append(
                    {
                        "target": target_dir.name,
                        "file": fpath.name,
                        "filter": _filter_guess(fpath.name),
                        "telescope": "",
                        "file_date": "",
                        "xposure_s": None,
                        "size_mb": round(fpath.stat().st_size / (1024 * 1024), 2),
                        "rel_path": str(fpath.relative_to(FITS_ROOT.parent)).replace(
                            "\\", "/"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _path_for_band(catalog: pd.DataFrame, target: str, band: str) -> Path | None:
    """Pick one FITS path per band (alphabetically first filename if duplicates)."""
    sub = catalog[(catalog["target"] == target) & (catalog["filter"] == band)]
    if sub.empty:
        return None
    fname = sorted(sub["file"].tolist())[0]
    p = FITS_ROOT / target / fname
    return p if p.is_file() else None


def _file_date_for_band(catalog: pd.DataFrame, target: str, band: str) -> str | None:
    sub = catalog[(catalog["target"] == target) & (catalog["filter"] == band)]
    if sub.empty:
        return None
    v = sub.iloc[0]["file_date"]
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    return s or None


def _fits_context_for_target(catalog: pd.DataFrame, target: str) -> str:
    lines: list[str] = []
    for band in BANDS:
        fd = _file_date_for_band(catalog, target, band)
        if fd:
            lines.append(f"{band}: primary FITS DATE={fd}")
    return (
        "\n".join(lines)
        if lines
        else "No FITS DATE in catalog rows for this target."
    )


@st.cache_data(show_spinner=False)
def _cached_u8_preview(path_str: str) -> np.ndarray | None:
    """Uint8 preview for ``path_str``; cached by absolute path."""
    arr = read_sci_2d(path_str)
    if arr is None:
        return None
    return percentile_stretch_u8(arr)


# =============================================================================
# Streamlit UI
# =============================================================================

st.title("Local JWST FITS catalog")
st.caption("Local data only: `fits_images/<target_name>/*.fits`.")

if st.button("🔄 Refresh catalog"):
    load_catalog.clear()
    st.rerun()

catalog = load_catalog()

if catalog.empty:
    st.warning(f"No `.fits` under `{FITS_ROOT}`. Add files there and reload.")
    st.stop()

targets = sorted(catalog["target"].unique())
choice = st.selectbox("Target", targets)

view = catalog[catalog["target"] == choice].copy()
paths = view.pop("rel_path") if "rel_path" in view.columns else None

st.dataframe(view, use_container_width=True, hide_index=True)

if paths is not None:
    with st.expander("Relative paths"):
        for p in paths:
            st.code(p)

st.markdown("---")

st.markdown(
    """
### MIRI quicklook (three bands)

**F770W** · **F1500W** · **F2550W** — one column per band when a matching file exists.

Pixels: `read_sci_2d()` in `jwst_fits_preview.py` (same SCI / squeeze / cube-sum steps as
`Working_draft.ipynb`). Display: linear **uint8** stretch (`percentile_stretch_u8`), not
the notebook’s LogNorm. Missing or bad files show a status line in that column only.
"""
)

cols = st.columns(3)
for col, band in zip(cols, BANDS):
    with col:
        st.markdown(f"#### {band}")
        p = _path_for_band(catalog, choice, band)
        if p is None:
            st.info(f"**{band}:** no file (expect `_miri_<band>` in the filename).")
            continue
        try:
            u8 = _cached_u8_preview(str(p.resolve()))
            if u8 is None:
                st.warning(f"**{band}:** could not build 2D science array.")
            else:
                st.image(u8, caption=p.name, use_container_width=True)
        except Exception as exc:  # noqa: BLE001 — surface read errors in this column only
            st.warning(f"**{band}:** `{exc}`")

# -------------------------------------------------------------------------
# False-color composite
# -------------------------------------------------------------------------

st.markdown("---")
st.markdown("### False-Color Composite")

fc_path = PROJECT_ROOT / f"{choice}_false_color.png"

if fc_path.is_file():
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.image(str(fc_path), caption=f"{choice} — JWST MIRI False Color", use_container_width=True)
else:
    st.info(
        f"No false-color image found for **{choice}**. "
        f"Run `False_color.ipynb` with this target to generate `{choice}_false_color.png`."
    )

# -------------------------------------------------------------------------
# Target description:    next to false-color PNG, or auto-generate with API
# -------------------------------------------------------------------------

desc_path = target_description_path(PROJECT_ROOT, choice)
fail_key = f"desc_api_fail_{choice}"
blurb = load_target_description(desc_path)
llm_err: str | None = None

if blurb is None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        if st.session_state.get(fail_key):
            llm_err = st.session_state.get(f"{fail_key}_msg", "Description request failed.")
        else:
            ctx = _fits_context_for_target(catalog, choice)
            with st.spinner("Generating target description…"):
                data, err = fetch_strict_target_blurb(choice, ctx)
            if err:
                st.session_state[fail_key] = True
                st.session_state[f"{fail_key}_msg"] = err
                llm_err = err
            elif data:
                try:
                    save_target_description(desc_path, data)
                except OSError as exc:
                    msg = f"Could not save description file: {exc}"
                    st.session_state[fail_key] = True
                    st.session_state[f"{fail_key}_msg"] = msg
                    llm_err = msg
                else:
                    blurb = data
    else:
        llm_err = "no_key"

if blurb:
    st.markdown("---")
    st.markdown("### Target description")
    st.caption(
        "What · distance · when the image was taken — from "
        f"`{desc_path.name}` (same folder as the false-color PNG)."
    )
    st.markdown(f"**What it is:** {blurb['what']}")
    st.markdown(f"**Distance:** {blurb['distance']}")
    st.markdown(f"**When the image was taken:** {blurb['when_image_taken']}")
elif llm_err == "no_key":
    st.markdown("---")
    st.info(
        f"No `{desc_path.name}` yet. Add `OPENAI_API_KEY` to `.env` to generate it here, "
        "or run the last cell of `False_color.ipynb` after saving the false-color image."
    )
elif llm_err:
    st.markdown("---")
    st.warning(f"Could not load or create target description: {llm_err}")
    if llm_err != "no_key" and st.button("Retry description", key=f"retry_desc_{choice}"):
        st.session_state.pop(fail_key, None)
        st.session_state.pop(f"{fail_key}_msg", None)
        st.rerun()
