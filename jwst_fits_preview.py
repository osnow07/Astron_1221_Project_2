"""
## JWST `_i2d.fits` preview helpers

Single implementation used by the Streamlit demo and importable from
`Working_draft.ipynb` (e.g. `from jwst_fits_preview import read_sci_2d`).

**Pipeline (same as the notebook plotting loop)**

1. `fits.open(path)` — `memmap=False` for small local files.
2. Science array: extension **`SCI`**, or index **`1`** if `SCI` is missing.
3. `np.squeeze`; if rank-**3**, collapse with `np.nansum(..., axis=0)`; rank-**2** passes through.

**Display**

- Notebook: `matplotlib` **LogNorm** on percentiles (typical JWST quicklook).
- Streamlit: `percentile_stretch_u8()` uses the **same percentile cutoffs** but maps
  linearly to **uint8** for `st.image` (no matplotlib dependency). Brightness structure
  is comparable; scaling is not identical to LogNorm.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits


def read_sci_2d(path: Path | str) -> np.ndarray | None:
    """Return 2D float science data, or ``None`` on I/O failure or unsupported rank."""
    path = Path(path)
    try:
        hdul = fits.open(path, memmap=False)
    except OSError:
        return None
    try:
        try:
            cube = hdul["SCI"].data
        except KeyError:
            cube = hdul[1].data
        cube = np.squeeze(cube)
        if cube.ndim == 3:
            return np.nansum(cube, axis=0).astype(np.float64, copy=False)
        if cube.ndim == 2:
            return cube.astype(np.float64, copy=False)
        return None
    finally:
        hdul.close()


def percentile_stretch_u8(
    image: np.ndarray,
    lo_pct: float = 5.0,
    hi_pct: float = 99.0,
) -> np.ndarray | None:
    """Linear percentile stretch to 2D ``uint8`` for ``st.image``."""
    work = np.where(np.isfinite(image) & (image > 0), image, np.nan)
    vmin = np.nanpercentile(work, lo_pct)
    vmax = np.nanpercentile(work, hi_pct)
    vmin = max(float(vmin), 1e-10)
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin * 10.0
    scaled = (work - vmin) / (vmax - vmin)
    scaled = np.clip(np.nan_to_num(scaled, nan=0.0), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)
