"""Ollama adapter: local, private, unlimited. POST /api/chat with format=<json schema>."""
from __future__ import annotations

import time

import httpx

from .base import ProviderResult, extract_json


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base = base_url.rstrip("/")

    def complete_json(self, system, user, schema, model, max_tokens=4096, thinking=None, timeout=300.0):
        res = ProviderResult()
        t0 = time.time()
        body = {"model": model, "stream": False, "think": False, "format": schema or "json",
                "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": 8192},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        try:
            r = httpx.post(self.base + "/api/chat", json=body, timeout=timeout)
            if r.status_code != 200:
                res.error, res.error_kind = f"{r.status_code} {r.text[:200]}", "server" if r.status_code >= 500 else "bad_request"
            else:
                j = r.json()
                res.raw_text = (j.get("message") or {}).get("content", "")
                res.usage = {"in": j.get("prompt_eval_count", 0), "out": j.get("eval_count", 0), "thinking": 0}
                res.parsed = extract_json(res.raw_text)
                if res.parsed is None:
                    res.error, res.error_kind = "unparseable JSON", "parse"
        except httpx.TimeoutException:
            res.error, res.error_kind = "timeout", "timeout"
        except Exception as e:
            res.error, res.error_kind = str(e)[:200], "server"
        res.latency_s = round(time.time() - t0, 2)
        return res
