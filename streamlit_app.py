r"""
# Streamlit app: local JWST MIRI catalog and viewer

## Overview

Single script: **scan FITS on disk** → **pick a target** → **show metadata**, **three-band
quicklook**, optional **false-color PNG**, **LLM target description**, and **chat**. No MAST
downloads inside the app; data must already live under `fits_images/<target>/`.

---

## Sections (how the script runs top to bottom)

| # | Section | What it does |
|---|---------|----------------|
| 1 | **Setup** | Imports, `reload_env()`, page config |
| 2 | **Paths & bands** | `PROJECT_ROOT`, `FITS_ROOT`, MIRI band list `BANDS` |
| 3 | **Catalog helpers** | Walk FITS tree, parse filters from filenames, build `DataFrame` |
| 4 | **UI — catalog** | Refresh button, target selectbox, table, path expander |
| 5 | **UI — MIRI triptych** | Three columns: F770W / F1500W / F2550W cached uint8 previews |
| 6 | **UI — false color** | `{target}_false_color.png` if present (from `False_color.ipynb`) |
| 7 | **UI — target description** | Load or generate `{target}_target_description.json` via `target_llm` |
| 8 | **UI — chat** | Per-target `st.session_state` history; calls `chat_about_jwst_target` |

**Shared per target:** `view` (catalog rows), `api_key`, `fits_ctx` — computed once after
you pick a target so FITS context is not rebuilt repeatedly.

---

## Related files

- `jwst_fits_preview.py` — `read_sci_2d`, `percentile_stretch_u8` (aligned with notebooks)
- `target_llm.py` — `.env`, LiteLLM, OSU proxy defaults, blurb + chat
- `False_color.ipynb` — writes `{target}_false_color.png` and optional description cell
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import streamlit as st
from astropy.io import fits

import numpy as np
from jwst_fits_preview import percentile_stretch_u8, read_sci_2d
from target_llm import (
    chat_about_jwst_target,
    fetch_strict_target_blurb,
    get_llm_api_key,
    load_target_description,
    reload_env,
    save_target_description,
    target_description_path,
)

# =============================================================================
# 1. Setup — refresh `.env` before any `st.*` call
# =============================================================================
reload_env()

st.set_page_config(page_title="Local JWST FITS", layout="wide")

# =============================================================================
# 2. Paths & MIRI bands
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
FITS_ROOT = PROJECT_ROOT / "fits_images"

# Same three filters as `target_filters` in `Working_draft.ipynb`
BANDS: tuple[str, ...] = ("F770W", "F1500W", "F2550W")


# =============================================================================
# 3. Catalog helpers — scan `fits_images/` and summarize FITS headers
# =============================================================================


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


def _path_for_band(target_rows: pd.DataFrame, target: str, band: str) -> Path | None:
    """Pick one FITS path per band (alphabetically first filename if duplicates)."""
    sub = target_rows[target_rows["filter"] == band]
    if sub.empty:
        return None
    fname = sorted(sub["file"].tolist())[0]
    p = FITS_ROOT / target / fname
    return p if p.is_file() else None


def _file_date_for_band(target_rows: pd.DataFrame, band: str) -> str | None:
    sub = target_rows[target_rows["filter"] == band]
    if sub.empty:
        return None
    v = sub.iloc[0]["file_date"]
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    return s or None


def _fits_context_for_target_rows(target_rows: pd.DataFrame) -> str:
    lines: list[str] = []
    for band in BANDS:
        fd = _file_date_for_band(target_rows, band)
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
# 4. UI — Catalog: refresh, pick target, show table
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

api_key = get_llm_api_key()
fits_ctx = _fits_context_for_target_rows(view)

st.dataframe(view, use_container_width=True, hide_index=True)

if paths is not None:
    with st.expander("Relative paths"):
        for p in paths:
            st.code(p)

# =============================================================================
# 5. UI — MIRI triptych (one column per band)
# =============================================================================

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
        p = _path_for_band(view, choice, band)
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

# =============================================================================
# 6. UI — False-color PNG (from notebook)
# =============================================================================

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

# =============================================================================
# 7. UI — Target description (JSON + optional LLM)
# =============================================================================

desc_path = target_description_path(PROJECT_ROOT, choice)
fail_key = f"desc_api_fail_{choice}"
blurb = load_target_description(desc_path)
llm_err: str | None = None

if blurb is None:
    if api_key:
        if st.session_state.get(fail_key):
            llm_err = st.session_state.get(f"{fail_key}_msg", "Description request failed.")
        else:
            with st.spinner("Generating target description…"):
                data, err = fetch_strict_target_blurb(choice, fits_ctx)
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
        f"No `{desc_path.name}` yet. Add `ASTRO1221_API_KEY` or `OPENAI_API_KEY` to `.env` "
        "to generate it here, or run the last cell of `False_color.ipynb` after saving the false-color image."
    )
elif llm_err:
    st.markdown("---")
    st.warning(f"Could not load or create target description: {llm_err}")
    if llm_err != "no_key" and st.button("Retry description", key=f"retry_desc_{choice}"):
        st.session_state.pop(fail_key, None)
        st.session_state.pop(f"{fail_key}_msg", None)
        st.rerun()

# =============================================================================
# 8. UI — Chat (same LiteLLM stack as `target_llm`)
# =============================================================================

st.markdown("---")
st.markdown("### Chat about this target")

chat_state_key = f"chat_messages_{choice}"
if chat_state_key not in st.session_state:
    st.session_state[chat_state_key] = []

if not api_key:
    st.info(
        "Add `ASTRO1221_API_KEY` or `OPENAI_API_KEY` to `.env` (project root) to use the chat."
    )
else:
    for msg in st.session_state[chat_state_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input(f"Ask about {choice}…"):
        st.session_state[chat_state_key].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                reply, err = chat_about_jwst_target(
                    choice,
                    st.session_state[chat_state_key],
                    target_blurb=blurb,
                    fits_context=fits_ctx,
                )
            if err:
                st.error(err)
                st.session_state[chat_state_key].append(
                    {"role": "assistant", "content": f"(Error) {err}"}
                )
            else:
                st.markdown(reply)
                st.session_state[chat_state_key].append(
                    {"role": "assistant", "content": reply}
                )

    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("Clear chat", key=f"clear_chat_{choice}"):
            st.session_state[chat_state_key] = []
            st.rerun()

