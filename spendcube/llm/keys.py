"""Key selection. Gemini entries use GEMINI_API_KEY_DEV in dev mode, GEMINI_API_KEY in blind mode.
Never print key values."""
from __future__ import annotations

import os


def resolve_key_env(key_env: str | None, mode: str) -> str | None:
    if key_env == "GEMINI_API_KEY":
        return "GEMINI_API_KEY" if mode == "blind" else "GEMINI_API_KEY_DEV"
    return key_env


def key_present(key_env: str | None, mode: str) -> bool:
    k = resolve_key_env(key_env, mode)
    return bool(k and os.environ.get(k, "").strip())
