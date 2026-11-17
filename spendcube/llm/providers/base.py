"""Provider interface: every backend implements complete_json and never raises."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


@dataclass
class ProviderResult:
    parsed: Optional[dict] = None
    raw_text: str = ""
    usage: dict = field(default_factory=lambda: {"in": 0, "out": 0, "thinking": 0})
    rate_headers: dict = field(default_factory=dict)
    error: Optional[str] = None
    error_kind: Optional[str] = None      # rate_limit | quota | server | auth | bad_request | timeout | parse | other
    retry_after: Optional[float] = None   # seconds, when the provider told us
    latency_s: float = 0.0


class LLMProvider(Protocol):
    name: str

    def complete_json(self, system: str, user: str, schema: dict, model: str,
                      max_tokens: int = 4096, thinking: Optional[str] = None,
                      timeout: float = 120.0) -> ProviderResult: ...


def extract_json(text: str) -> Optional[Any]:
    """Tolerant JSON extraction: strips code fences, takes the first balanced object/array."""
    import json
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    try:
        return json.loads(t)
    except Exception:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        s = t.find(open_c)
        if s < 0:
            continue
        depth = 0
        for i in range(s, len(t)):
            if t[i] == open_c:
                depth += 1
            elif t[i] == close_c:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[s:i + 1])
                    except Exception:
                        break
    return None
