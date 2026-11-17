"""Groq adapter: OpenAI-style chat completions over httpx; JSON mode + client-side validation."""
from __future__ import annotations

import os
import time
from typing import Optional

import httpx

from .base import ProviderResult, extract_json

URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqProvider:
    name = "groq"

    def __init__(self, api_key: str):
        self.key = api_key

    def complete_json(self, system, user, schema, model, max_tokens=4096, thinking=None, timeout=120.0):
        res = ProviderResult()
        t0 = time.time()
        body = {"model": model, "temperature": 0, "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user + "\n\nReturn ONLY the JSON object."}]}
        if thinking and "gpt-oss" in model:
            body["reasoning_effort"] = {"minimal": "low", "low": "low", "medium": "medium", "high": "high"}.get(thinking, "low")
        try:
            r = httpx.post(URL, json=body, headers={"Authorization": f"Bearer {self.key}"}, timeout=timeout)
            res.rate_headers = {k: v for k, v in r.headers.items() if k.lower().startswith(("x-ratelimit", "retry-after"))}
            if r.status_code != 200:
                ra = r.headers.get("retry-after")
                res.retry_after = float(ra) if ra and ra.replace(".", "").isdigit() else None
                res.error = f"{r.status_code} {r.text[:200]}"
                res.error_kind = ("rate_limit" if r.status_code == 429 else "auth" if r.status_code in (401, 403)
                                  else "server" if r.status_code >= 500 else "bad_request")
                if r.status_code == 429 and "per day" in r.text.lower():
                    res.error_kind = "quota"
            else:
                j = r.json()
                res.raw_text = j["choices"][0]["message"].get("content") or ""
                u = j.get("usage", {})
                res.usage = {"in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0), "thinking": 0}
                res.parsed = extract_json(res.raw_text)
                if res.parsed is None:
                    res.error, res.error_kind = "unparseable JSON", "parse"
        except httpx.TimeoutException:
            res.error, res.error_kind = "timeout", "timeout"
        except Exception as e:
            res.error, res.error_kind = str(e)[:200], "other"
        res.latency_s = round(time.time() - t0, 2)
        return res
