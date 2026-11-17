"""Gemini adapter (official google-genai SDK).

JSON output via response_json_schema; thinking via ThinkingConfig(thinking_level). Models that reject a
feature (system_instruction, JSON mode, thinking) are retried once without it and remembered.
"""
from __future__ import annotations

import os
import re
import threading
import time
from typing import Optional

from .base import ProviderResult, extract_json

_LEVELS = {"minimal": "MINIMAL", "low": "LOW", "medium": "MEDIUM", "high": "HIGH"}


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str):
        from google import genai
        self._genai = genai
        from google.genai import types
        self._types = types
        self._api_key = api_key
        self._clients: dict[int, object] = {}
        self._client_lock = threading.Lock()      # concurrent first calls must not build (and GC-close) duplicate clients
        self._no_feature: dict[str, set[str]] = {}   # model -> features it rejected

    def _client(self, timeout: float):
        ms = int(timeout * 1000)
        with self._client_lock:
            if ms not in self._clients:
                self._clients[ms] = self._genai.Client(
                    api_key=self._api_key, http_options=self._types.HttpOptions(timeout=ms))
            return self._clients[ms]

    @property
    def client(self):
        return self._client(60.0)

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _classify_error(e: Exception) -> tuple[str, Optional[float], str]:
        msg = str(e)
        code = getattr(e, "code", None) or getattr(e, "status_code", None)
        m = re.search(r"retry(?:Delay)?\W+(\d+(?:\.\d+)?)s", msg, re.I)
        retry = float(m.group(1)) if m else None
        low = msg.lower()
        if code == 429 or "resource_exhausted" in low or "quota" in low:
            kind = "quota" if ("per day" in low or "perday" in low or "daily" in low) else "rate_limit"
            return kind, retry, msg
        if code in (401, 403) or "api key" in low or "permission" in low:
            return "auth", None, msg
        if isinstance(code, int) and 500 <= code < 600:
            return "server", retry, msg
        if code == 400 or "invalid_argument" in low:
            return "bad_request", None, msg
        if "timeout" in low or "timed out" in low:
            return "timeout", None, msg
        return "other", retry, msg

    def _config(self, model: str, system: str, schema: Optional[dict], max_tokens: int,
                thinking: Optional[str]):
        from google.genai import types
        rej = self._no_feature.get(model, set())
        kw = dict(max_output_tokens=max_tokens, temperature=0.0)
        if system and "system" not in rej:
            kw["system_instruction"] = system
        if schema and "json" not in rej:
            kw["response_mime_type"] = "application/json"
            kw["response_json_schema"] = schema
        if thinking and "thinking" not in rej and thinking in _LEVELS:
            kw["thinking_config"] = types.ThinkingConfig(thinking_level=_LEVELS[thinking])
        return types.GenerateContentConfig(**kw)

    # -- main --------------------------------------------------------------
    def complete_json(self, system: str, user: str, schema: dict, model: str,
                      max_tokens: int = 4096, thinking: Optional[str] = None,
                      timeout: float = 120.0) -> ProviderResult:
        res = ProviderResult()
        t0 = time.time()
        prompt = user
        for attempt in range(4):
            try:
                rej = self._no_feature.get(model, set())
                contents = (system + "\n\n" + user) if (system and "system" in rej) else user
                if schema and "json" in rej:
                    contents += "\n\nReturn ONLY valid minified JSON, no prose."
                resp = self._client(timeout).models.generate_content(
                    model=model, contents=contents,
                    config=self._config(model, system, schema, max_tokens, thinking))
                res.raw_text = resp.text or ""
                um = getattr(resp, "usage_metadata", None)
                if um:
                    res.usage = {"in": um.prompt_token_count or 0, "out": um.candidates_token_count or 0,
                                 "thinking": getattr(um, "thoughts_token_count", 0) or 0}
                res.parsed = extract_json(res.raw_text)
                if res.parsed is None:
                    res.error, res.error_kind = "unparseable JSON", "parse"
                break
            except Exception as e:  # SDK raises APIError subclasses
                kind, retry, msg = self._classify_error(e)
                res.error, res.error_kind, res.retry_after = msg[:400], kind, retry
                low = msg.lower()
                if kind == "bad_request" and attempt < 3:
                    rej = self._no_feature.setdefault(model, set())
                    if "thinking" in low and "thinking" not in rej:
                        rej.add("thinking"); continue
                    if ("system" in low or "developer instruction" in low) and "system" not in rej:
                        rej.add("system"); continue
                    if ("json" in low or "mime" in low or "schema" in low) and "json" not in rej:
                        rej.add("json"); continue
                break
        res.latency_s = round(time.time() - t0, 2)
        return res


def make_gemini(key_env: str) -> Optional[GeminiProvider]:
    key = os.environ.get(key_env, "")
    return GeminiProvider(key) if key else None
