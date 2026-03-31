r"""
# `target_llm` — LLM helpers for target blurbs and chat

## Overview

This module loads **`.env`** next to this file, calls **LiteLLM** with the right **API key**
and **base URL** (including Ohio State’s `litellmproxy.osu-ai.org` when you use
`ASTRO1221_API_KEY`), and exposes two main features:

1. **Structured target blurbs** — one JSON object with `what`, `distance`,
   `when_image_taken` (saved as `{target}_target_description.json`).
2. **Multi-turn chat** — free-form answers with the same credentials, using a
   system prompt that includes FITS context and optional saved blurb text.

---

## How the file is organized (sections)

| Section | What lives there |
|---------|------------------|
| **1. Imports & constants** | Module state, OSU proxy URL, lazy LiteLLM handle |
| **2. Environment** | `_normalize_secret`, `reload_env`, first `.env` load |
| **3. Blurb schema & prompts** | `ALLOWED_KEYS`, `SYSTEM_PROMPT`, JSON fence stripping, `_validate_blurb` |
| **4. Description files** | Paths, `load_target_description`, `save_target_description` |
| **5. Credentials & routing** | `get_llm_api_key`, `_resolve_api_base`, `_effective_model`, `_sync_openai_env` |
| **6. LiteLLM call** | `_get_litellm_completion`, `_litellm_complete` (single HTTP path) |
| **7. Public APIs** | `fetch_strict_target_blurb`, `chat_about_jwst_target` |

Execution flow: **blurb** → `fetch_strict_target_blurb` → `_litellm_complete` (with
`response_format=json_object`) → parse JSON → validate → save.
**Chat** → `chat_about_jwst_target` → build system + history → `_litellm_complete` (no JSON mode).

---

## Environment variables (`.env`)

| Variable | Role |
|----------|------|
| `ASTRO1221_API_KEY` | OSU course key; triggers default proxy if no base URL set |
| `OPENAI_API_KEY` | Optional personal OpenAI key (tried first unless `LLM_USE_COURSE_KEY_FIRST=1`) |
| `OPENAI_BASE_URL` / `OPENAI_API_BASE` / `ASTRO1221_BASE_URL` / `LITELLM_API_BASE` | Explicit OpenAI-compatible API root (first non-empty wins) |
| `OPENAI_MODEL` | Override model id (default on OSU proxy: `openai/GPT-4.1-mini`) |
| `LLM_USE_COURSE_KEY_FIRST` | `1` / `true` — prefer course key when both keys exist |
| `LLM_DISABLE_OSU_PROXY` | `1` / `true` — never auto-use `litellmproxy.osu-ai.org` |

---

## Dependencies

- **`python-dotenv`** — load `.env`
- **`litellm`** — only required when calling the API (lazy-imported)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# =============================================================================
# 1. Imports & module constants
# =============================================================================
# Lazy import so JSON helpers work without litellm installed; load once when needed.
_litellm_completion: Any = None  # None = unset, False = import failed

# Load `.env` relative to this file so it works regardless of the current
# working directory (e.g., when launched via `streamlit run`).
_PROJECT_ROOT = Path(__file__).resolve().parent

# Ohio State course LiteLLM gateway (same as Astronomy 1221 companion notebooks).
_OSU_LITELLM_PROXY = "https://litellmproxy.osu-ai.org"
_OSU_DEFAULT_MODEL = "openai/GPT-4.1-mini"


# =============================================================================
# 2. Environment (`.env`) loading
# =============================================================================


def _normalize_secret(raw: str | None) -> str:
    """Strip whitespace; if pasted multi-line, keep first line only."""
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    if "\n" in s or "\r" in s:
        s = s.splitlines()[0].strip()
    return s


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

    # 3. Load using dotenv and FORCE an override (utf-8-sig strips BOM from some editors)
    try:
        load_dotenv(dotenv_path=target_path, override=True, encoding="utf-8-sig")
    except TypeError:
        load_dotenv(dotenv_path=target_path, override=True)

    # 4. Manual sync — matches Jupyter / some hosts where dotenv alone misses updates
    with open(target_path, encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = _normalize_secret(val.strip().strip('"').strip("'"))
            if key:
                os.environ[key] = val

    if os.getenv("ASTRO1221_API_KEY") or os.getenv("OPENAI_API_KEY"):
        print(f"--- [DEBUG] SUCCESS: API Key loaded from {target_path} ---")
    else:
        print(f"--- [DEBUG] FAIL: File found at {target_path} but Key is missing/empty ---")

# Initial load when this module is imported
reload_env()

# =============================================================================
# 3. Blurb schema, system prompt, and JSON validation
# =============================================================================

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


# =============================================================================
# 4. Target description JSON on disk (next to false-color PNG convention)
# =============================================================================


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


# =============================================================================
# 5. API key, proxy base URL, and model id
# =============================================================================


def get_llm_api_key() -> str:
    """
    Resolve which secret to send to LiteLLM.

    Default: ``OPENAI_API_KEY`` first, then ``ASTRO1221_API_KEY``, so a course key
    that is invalid at api.openai.com does not override a personal key.
    Set ``LLM_USE_COURSE_KEY_FIRST=1`` to reverse that order.
    """
    astro = _normalize_secret(os.getenv("ASTRO1221_API_KEY"))
    openai = _normalize_secret(os.getenv("OPENAI_API_KEY"))
    prefer_course = os.getenv("LLM_USE_COURSE_KEY_FIRST", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if prefer_course:
        return astro or openai
    return openai or astro


def _resolve_api_base() -> str | None:
    """OpenAI-compatible root URL (LiteLLM ``api_base``). Env wins; else OSU proxy for course key."""
    for name in (
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "ASTRO1221_BASE_URL",
        "LITELLM_API_BASE",
    ):
        v = _normalize_secret(os.getenv(name))
        if v:
            return v.rstrip("/")
    if os.getenv("LLM_DISABLE_OSU_PROXY", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return None
    astro = _normalize_secret(os.getenv("ASTRO1221_API_KEY"))
    if not astro:
        return None
    if get_llm_api_key() != astro:
        return None
    return _OSU_LITELLM_PROXY.rstrip("/")


def _effective_model(base_url: str | None) -> str:
    """Model id for LiteLLM: env override, else OSU proxy default, else OpenAI direct default."""
    custom = os.getenv("OPENAI_MODEL", "").strip()
    if custom:
        return custom
    if base_url and "litellmproxy.osu-ai.org" in base_url:
        return _OSU_DEFAULT_MODEL
    return "gpt-4o-mini"


def _sync_openai_env(api_key: str) -> None:
    if api_key and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = api_key


# =============================================================================
# 6. LiteLLM completion (single code path for all HTTP calls)
# =============================================================================


def _get_litellm_completion():
    global _litellm_completion
    if _litellm_completion is False:
        return None
    if _litellm_completion is not None:
        return _litellm_completion
    try:
        from litellm import completion as completion_fn

        _litellm_completion = completion_fn
        return completion_fn
    except ImportError:
        _litellm_completion = False
        return None


def _litellm_complete(
    messages: list[dict[str, str]],
    *,
    temperature: float,
    response_format: dict[str, str] | None = None,
) -> tuple[str | None, str | None]:
    """Single completion path: one import, shared model/base_url handling."""
    completion_fn = _get_litellm_completion()
    if completion_fn is None:
        return None, "Install the `litellm` package."

    api_key = get_llm_api_key()
    if not api_key:
        return None, "API Key (ASTRO1221_API_KEY or OPENAI_API_KEY) is not set. Add it to `.env`."

    _sync_openai_env(api_key)
    base_url = _resolve_api_base()
    model = _effective_model(base_url)

    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "temperature": temperature,
        "messages": messages,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    if base_url:
        kwargs["api_base"] = base_url

    try:
        resp = completion_fn(**kwargs)
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).strip()
        low = msg.lower()
        if (
            "incorrect api key" in low
            or "invalid api key" in low
            or "authentication" in low
            or "401" in msg
        ):
            base = _resolve_api_base()
            if base and "litellmproxy.osu-ai.org" in base:
                return None, (
                    f"{msg}\n\n"
                    "The **Ohio State LiteLLM proxy** rejected this request. "
                    "Confirm `ASTRO1221_API_KEY` matches the key from your instructor and that your "
                    "account still has access."
                )
            hint = (
                "This key was sent to **api.openai.com**, which rejected it. That usually means the "
                "secret is for a **different host** (class proxy / vendor), not platform.openai.com.\n\n"
                "For **OSU Astronomy 1221**, only `ASTRO1221_API_KEY` is required; this project "
                f"defaults the proxy to `{_OSU_LITELLM_PROXY}` when that key is active. "
                "If you set `LLM_DISABLE_OSU_PROXY=1`, add `OPENAI_BASE_URL` yourself.\n\n"
                "For other hosts, set `OPENAI_BASE_URL` (or `OPENAI_API_BASE` / `ASTRO1221_BASE_URL`).\n\n"
                "If both `OPENAI_API_KEY` and `ASTRO1221_API_KEY` exist, `OPENAI_API_KEY` is used first "
                "unless `LLM_USE_COURSE_KEY_FIRST=1`.\n\n"
                "Personal OpenAI keys: https://platform.openai.com/account/api-keys"
            )
            if not base:
                hint += (
                    f"\n\n**No API base URL resolved.** With only `ASTRO1221_API_KEY`, the app should use "
                    f"`{_OSU_LITELLM_PROXY}` unless `LLM_DISABLE_OSU_PROXY=1` is set."
                )
            return None, f"{msg}\n\n{hint}"
        return None, f"LiteLLM API error: {exc}"

    raw = (resp.choices[0].message.content or "").strip()
    if not raw:
        return None, "Empty model response."
    return raw, None


# =============================================================================
# 7. Public APIs — blurb generation and chat
# =============================================================================


def fetch_strict_target_blurb(
    target_name: str,
    fits_context: str,
) -> tuple[dict[str, str] | None, str | None]:
    """
    Call the chat API via LiteLLM; return (parsed dict, None) on success, or (None, error message).
    """
    user_msg = (
        f"Target designation: {target_name}\n\n"
        f"Local FITS catalog context (use for timing when helpful):\n{fits_context}"
    )
    raw, err = _litellm_complete(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    if err or raw is None:
        return None, err or "Empty model response."

    try:
        parsed = json.loads(_strip_json_fence(raw))
    except json.JSONDecodeError as exc:
        return None, f"Model did not return valid JSON: {exc}"

    validated = _validate_blurb(parsed)
    if validated is None:
        return None, "Model JSON missing required keys or empty values (expected what, distance, when_image_taken)."

    return validated, None


# --- Chat: system template + history cap (used only by `chat_about_jwst_target`) ---

CHAT_SYSTEM_TEMPLATE = """You are a helpful astronomy teaching assistant for a JWST MIRI course project.

The student is viewing local FITS data and images for this target: **{target_name}**.

{facts_block}

Guidelines:
- Answer clearly and accurately. Prefer mainstream astronomy; note uncertainty when appropriate.
- Tie answers to JWST, MIRI, and this target when relevant.
- Be concise unless the student asks for more depth.
- The FITS date lines above come from local file headers only — do not invent specific observation dates beyond that unless the student or saved description states them."""

# Cap history so each request stays small (tokens + latency).
_MAX_CHAT_MESSAGES = 24


def chat_about_jwst_target(
    target_name: str,
    messages: list[dict[str, str]],
    *,
    target_blurb: dict[str, str] | None = None,
    fits_context: str = "",
) -> tuple[str | None, str | None]:
    """
    Multi-turn chat about the selected target. ``messages`` are OpenAI-style
    {role, content} dicts for user/assistant only (no system); system is built here.
    """
    if target_blurb:
        facts_block = (
            "Saved target summary (from this project):\n"
            f"- What it is: {target_blurb['what']}\n"
            f"- Distance: {target_blurb['distance']}\n"
            f"- When the image was taken: {target_blurb['when_image_taken']}\n"
        )
    else:
        facts_block = "No saved target description JSON yet for this object.\n"

    facts_block += f"\nLocal FITS catalog context:\n{fits_context or 'None.'}"

    system_content = CHAT_SYSTEM_TEMPLATE.format(
        target_name=target_name,
        facts_block=facts_block.strip(),
    )

    recent = messages[-_MAX_CHAT_MESSAGES:] if len(messages) > _MAX_CHAT_MESSAGES else messages

    api_messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
    for m in recent:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        api_messages.append({"role": role, "content": content})

    if len(api_messages) < 2:
        return None, "No user message to send."

    text, err = _litellm_complete(api_messages, temperature=0.4)
    if err:
        return None, err
    return text, None