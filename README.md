# Astron 1221 — JWST MIRI catalog, false-color, and LLM target notes

## Project goals

- Build a **local catalog** of JWST **MIRI** `_i2d.fits` files: per-target metadata (filter, telescope, dates, exposure, file paths) in a **Pandas** `DataFrame`.
- Produce **science-ready false-color** (or single-band) images from three MIRI bands (**F770W**, **F1500W**, **F2550W**), with **north-up alignment** via **reproject**, display in **Matplotlib**, and save high-DPI PNGs.
- Provide a **Streamlit** app to browse the catalog, preview each band, show the false-color composite when present, and optionally **generate or load** a short **JSON “target description”** (what / distance / when observed) plus a **chat** interface — both using the same **LiteLLM** stack and **OSU Astronomy 1221** proxy settings as in class.
- Support **MAST-based download** (when files are missing) from **`Working_draft.ipynb`**, while allowing **fully offline** use once FITS live under `fits_images/<target>/`.

---

## Methodology

1. **Data acquisition** (`Working_draft.ipynb`): Query **MAST** via **astroquery** for JWST MIRI imaging, filter by instrument and product type, download `_i2d.fits` products into `fits_images/<target_name>/`. If matching files already exist locally, the notebook **skips** the query to save time and bandwidth.
2. **Configuration**: `target_config.json` stores the active **`target_name`** and **`target_filters`** for the false-color pipeline (`False_color.ipynb` reads this file).
3. **False-color pipeline** (`False_color.ipynb`): Discover FITS per filter, build a wavelength-sorted table, load 2D science data (SCI extension or extension 1), **fill NaNs**, **reproject** all bands to a common **north-up WCS**, apply **background subtraction**, **arcsinh** stretch, optional **Gaussian** smoothing and **saturation** boost, **crop** to valid footprint, save `{target_name}_false_color.png`.
4. **Catalog & previews** (`streamlit_app.py` + `jwst_fits_preview.py`): Scan `fits_images/`, parse **filter** from filenames, read **FITS headers** for table columns, and build **uint8** previews with the same **read/science/squeeze/cube-sum** logic as the notebooks where applicable.
5. **LLM layer** (`target_llm.py`): Load `.env` (keys and optional `OPENAI_BASE_URL`); default **OSU** course setup uses `https://litellmproxy.osu-ai.org` with `ASTRO1221_API_KEY` when no explicit base URL is set. **Strict JSON** blurbs for descriptions; **plain text** for chat.

---

## Data sources

| Source | Role |
|--------|------|
| **MAST** (via **astroquery.mast**) | Discovery and download of JWST MIRI `_i2d.fits` products when not already on disk |
| **Local `fits_images/<target>/`** | Authoritative input for `False_color.ipynb` and `streamlit_app.py` (no network required in those steps) |
| **`target_config.json`** | Written by `Working_draft` (or by hand): current target and filter list for the false-color notebook |

JWST MIRI science context: MIRI is the **mid-infrared** imager on JWST; `_i2d.fits` files are **calibrated, rectified** 2D (or 3D collapsed) science products suitable for analysis and visualization.

---

## Preprocessing steps (summary)

- **Header/metadata**: Primary header + SCI extension for exposure and quality-related keywords; primary `DATE` (and similar) propagated into the catalog and LLM context strings.
- **Array handling**: SCI data **squeezed**; 3D cubes **summed** along the spectral axis with `nansum` where needed (same pattern as `jwst_fits_preview.read_sci_2d`).
- **NaN filling** (`False_color.ipynb`): Nearest-valid-pixel fill via **scipy** `distance_transform_edt` so gaps from dithering do not dominate the stretch.
- **Astrometry**: **reproject** onto a shared **north-up, east-left** tangent-plane WCS centered on the field.
- **Display stretch**: Median background subtraction, **percentile** clipping, **arcsinh** mapping for false-color; Streamlit uses a **linear percentile → uint8** path for web display (`percentile_stretch_u8`).

---

## Installation (Anaconda, class environment)

Use the **Anaconda** (or Miniconda) distribution your section uses. From the **project root** (`Astron_1221_Project_2`):

```bash
conda create -n astro1221 python=3.11 -y
conda activate astro1221
pip install -r requirements.txt
```

If you prefer conda for heavy scientific stacks:

```bash
conda install -c conda-forge astropy numpy pandas matplotlib scipy reproject jupyter -y
pip install streamlit astroquery litellm python-dotenv
```

Place a **`.env`** file next to `target_llm.py` (see **`.env.example`**) with `ASTRO1221_API_KEY` and, if needed, `OPENAI_BASE_URL` / `OPENAI_MODEL` — the code defaults to the **OSU LiteLLM proxy** when the course key is active.

---

## Usage

### Jupyter

1. **`Working_draft.ipynb`** — Set `target_name`, run all cells: downloads (if needed) to `fits_images/<target>/`, writes **`target_config.json`**.
2. **`False_color.ipynb`** — Run **sequentially from the top**: reads `target_config.json`, builds the false-color (or mono) figure, saves **`{target}_false_color.png`**, optional last cell writes **`{target}_target_description.json`** via the API (requires `.env` and network).

Kernel: use the same conda env you installed into (**Python 3**).

### Streamlit

From the project root:

```bash
conda activate astro1221
streamlit run streamlit_app.py
```

Browser opens to the local URL (default **http://localhost:8501**). Use **Refresh catalog** after adding FITS files. Pick a target to see the table, MIRI triptych, false-color PNG if present, description JSON (generate or load), and chat.

### Python modules (imported by notebooks / Streamlit)

| Module | Purpose |
|--------|---------|
| `jwst_fits_preview.py` | `read_sci_2d`, `percentile_stretch_u8` for FITS → 2D / uint8 preview |
| `target_llm.py` | `.env`, LiteLLM calls, blurb + chat helpers |

---

## Repository layout (high level)

```
fits_images/          # Local JWST data (not in git — large)
target_config.json    # Active target + filters for False_color
*_false_color.png     # Notebook outputs (often gitignored)
*_target_description.json  # Optional LLM JSON blurbs
streamlit_app.py      # Interactive catalog + LLM UI
target_llm.py         # LLM + environment configuration
jwst_fits_preview.py  # Shared FITS read / stretch helpers
```

---

## Course deliverable checklist

| Requirement | Where |
|-------------|--------|
| README (goals, methods, data, preprocessing, install, usage) | This file |
| `requirements.txt` | Core dependencies listed with brief comments |
| Notebook flow in markdown | `False_color.ipynb`, `Working_draft.ipynb` (overview cells) |
| Importable Python modules | `jwst_fits_preview.py`, `target_llm.py` |
| Docstrings | Module and public/private helpers documented in `.py` files |
