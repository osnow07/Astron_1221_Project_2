"""
## Streamlit demo: local JWST MIRI catalog + 3-panel preview

- **Data:** reads only `fits_images/<target_name>/*.fits` (no network).
- **Science image:** `jwst_fits_preview.read_sci_2d` / `percentile_stretch_u8` — same
  module the notebook can import for parity with `Working_draft.ipynb`.
- **Filter column:** parsed from filenames (`_miri_f770w`, etc.); MAST table columns are
  not used in this app.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import streamlit as st
from astropy.io import fits

import numpy as np
from jwst_fits_preview import percentile_stretch_u8, read_sci_2d

st.set_page_config(page_title="Local JWST FITS", layout="wide")

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

FITS_ROOT = Path(__file__).resolve().parent / "fits_images"

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
