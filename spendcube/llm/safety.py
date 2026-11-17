"""Safety checks for mock providers and blind runs."""
from __future__ import annotations

import os
import sys


def mock_env_vars() -> list[str]:
    return sorted(name for name in os.environ
                  if name.startswith("SPENDCUBE_FORCE_MOCK") or name.startswith("SPENDCUBE_MOCK_"))


def mock_active(llm_cfg: dict) -> bool:
    if mock_env_vars():
        return True
    return any(entry.get("provider") == "mock" for entry in (llm_cfg or {}).get("chain", []))


def reject_mock_in_blind(mode: str, *, doctor: bool = False) -> bool:
    """Print a safe, value-free refusal; return True when the requested run must stop."""
    if mode != "blind":
        return False
    names = mock_env_vars()
    if not names:
        return False
    action = "doctor --blind" if doctor else "blind mode"
    print(f"ERROR: refusing {action}: mock safety environment variable(s) are set: "
          f"{', '.join(names)}. Unset them before continuing.", file=sys.stderr)
    return True
