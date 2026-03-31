"""
Strict, JSON-only target blurbs via LiteLLM Chat Completions.

Environment (see `.env.example`):
  OPENAI_API_KEY — optional
  ASTRO1221_API_KEY    — required (Teacher provided key)
  OPENAI_MODEL   — optional, default gpt-4o-mini
  OPENAI_BASE_URL — optional (custom / Azure-compatible endpoint)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load `.env` relative to this file so it works regardless of the current
# working directory (e.g., when launched via `streamlit run`).
_PROJECT_ROOT = Path(__file__).resolve().parent

def reload_env(path: Path | str | None = None):
    """Reloads the .env file by resolving the absolute path of this script."""
    import os
    
    # 1. Identify the directory where THIS script (target_llm.py) actually lives
    _HERE = Path(__file__).resolve().parent
    
    # 2. Determine the target .env path
    if path:
        # If the user passed a path (like Path('.')/'.env'), resolve it to absolute
        target_path = Path(path).resolve()
    else:
        # Default to the same folder as target_llm.py
        target_path = _HERE / ".env"
    
    if not target_path.exists():
        print(f"--- [DEBUG] ERROR: .env NOT FOUND at {target_path} ---")
        return

    # 3. Load using dotenv and FORCE an override
    load_dotenv(dotenv_path=target_path, override=True)
    
    # 4. Manual Verification / Injection (Double-tap)
    # Sometimes load_dotenv fails to sync with os.environ in certain IDEs
    with open(target_path, "r") as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                key, val = line.strip().split("=", 1)
                os.environ[key] = val.strip('"').strip("'")

    if os.getenv("ASTRO1221_API_KEY") or os.getenv("OPENAI_API_KEY"):
        print(f"--- [DEBUG] SUCCESS: API Key loaded from {target_path} ---")
    else:
        print(f"--- [DEBUG] FAIL: File found at {target_path} but Key is missing/empty ---")

# Initial load
reload_env()

ALLOWED_KEYS: tuple[str, ...] = ("what", "distance", "when_image_taken")

SYSTEM_PROMPT = """You answer only about the astronomical object named by the user.

Rules (violation is an error):
1. Reply with a single JSON object. No markdown fences, no commentary before or after.
2. Use exactly these three keys as strings: "what", "distance", "when_image_taken".
3. "what": one or two sentences describing what the object is (type, notable features). GIVE A DISCRIPITION OF WHAT CLASS OF ASTRONOMICAL OBJECT IT IS, NOT JUST "[object] is an astronomical object featured in this JWST MIRI processing pipeline."
4. "distance": one sentence giving distance from Earth using standard units (e.g. light-years, kpc, or Mpc) and note uncertainty if large. DO NOT SAY "Distance data can be found in the associated FITS headers or SIMBAD." If you say that I will die.
5. "when_image_taken": one sentence stating when the JWST MIRI observation was taken. If FITS dates are provided in the user message, prefer those dates or ranges; otherwise give a careful approximate era (e.g. JWST Cycle 1) without inventing a specific calendar day.
6. Do not include any other keys, URLs, citations, disclaimers, or discussion of these rules.
7. Keep each value under 400 characters."""


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    # Safely look for markdown blocks using hex escape codes to prevent parser breakage
    m = re.match(r"^\x60\x60\x60(?:json)?\s*([\s\S]*?)\x60\x60\x60\s*$", t)
    if m:
        return m.group(1).strip()
    return t


def _validate_blurb(obj: Any) -> dict[str, str] | None:
    if not isinstance(obj, dict):
        return None
    out: dict[str, str] = {}
    for k in ALLOWED_KEYS:
        if k not in obj:
            return None
        v = obj[k]
        if not isinstance(v, str):
            v = str(v)
        v = v.strip()
        if not v:
            return None
        out[k] = v[:2000]
    return out


def target_description_path(project_root: Path | str, target_name: str) -> Path:
    """Same directory convention as ``{target}_false_color.png``."""
    root = Path(project_root)
    return root / f"{target_name}_target_description.json"


def load_target_description(path: Path | str) -> dict[str, str] | None:
    """Read and validate a saved JSON file; return None if missing or invalid."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return _validate_blurb(data)


def save_target_description(path: Path | str, data: dict[str, str]) -> None:
    """Write validated blurbs to disk (UTF-8, pretty-printed)."""
    v = _validate_blurb(data)
    if v is None:
        raise ValueError("data must contain non-empty what, distance, when_image_taken")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(v, f, ensure_ascii=False, indent=2)
        f.write("\n")


def fetch_strict_target_blurb(
    target_name: str,
    fits_context: str,
) -> tuple[dict[str, str] | None, str | None]:
    """
    Call the chat API via LiteLLM; return (parsed dict, None) on success, or (None, error message).
    """
    # Prioritize the teacher-provided ASTRO1221_API_KEY, fall back to OPENAI_API_KEY
    api_key = (os.getenv("ASTRO1221_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    
    if not api_key:
        return None, "API Key (ASTRO1221_API_KEY) is not set. Add it to `.env`."

    try:
        from litellm import completion
    except ImportError as exc:
        return None, f"Install the `litellm` package: {exc}"

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None

    user_msg = (
        f"Target designation: {target_name}\n\n"
        f"Local FITS catalog context (use for timing when helpful):\n{fits_context}"
    )

    # Force the api_key into the environment variable LiteLLM expects for OpenAI models 
    # if it came from ASTRO1221_API_KEY to prevent AuthenticationErrors.
    if api_key and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = api_key

    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    }
    
    if base_url:
        kwargs["api_base"] = base_url

    try:
        resp = completion(**kwargs)
    except Exception as exc:  # noqa: BLE001
        return None, f"LiteLLM API error: {exc}"

    raw = (resp.choices[0].message.content or "").strip()
    if not raw:
        return None, "Empty model response."

    try:
        parsed = json.loads(_strip_json_fence(raw))
    except json.JSONDecodeError as exc:
        return None, f"Model did not return valid JSON: {exc}"

    validated = _validate_blurb(parsed)
    if validated is None:
        return None, "Model JSON missing required keys or empty values (expected what, distance, when_image_taken)."

    return validated, None