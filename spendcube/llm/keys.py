"""Key selection. Gemini entries use GEMINI_API_KEY_DEV in dev mode, GEMINI_API_KEY in blind mode.
Never print key values."""
from __future__ import annotations

import os


def resolve_key_env(key_env: str | None, mode: str) -> str | None:
    if key_env == "GEMINI_API_KEY":
        return "GEMINI_API_KEY" if mode == "blind" else "GEMINI_API_KEY_DEV"
    return key_env


def resolve_key_envs(key_env: str | None, mode: str) -> list[str]:
    if key_env == "GEMINI_API_KEY":
        if mode == "blind":
            keys = [k for k in ("GEMINI_API_KEY", "GEMINI_API_KEY_DEV") if os.environ.get(k, "").strip()]
            return keys or ["GEMINI_API_KEY"]
        return ["GEMINI_API_KEY_DEV"]
    return [key_env] if key_env else []


def key_present(key_env: str | None, mode: str) -> bool:
    return any(os.environ.get(k, "").strip() for k in resolve_key_envs(key_env, mode))
