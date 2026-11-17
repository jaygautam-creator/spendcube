"""Deterministic mock provider (tests, --no-llm development). Chooses the leaf with max token overlap."""
from __future__ import annotations

import os
import re
import time

from .base import ProviderResult


class MockProvider:
    name = "mock"

    def __init__(self, fail_every: int = 0):
        self.calls = 0
        self.fail_every = fail_every

    def complete_json(self, system, user, schema, model, max_tokens=4096, thinking=None, timeout=120.0):
        self.calls += 1
        lat = float(os.environ.get("SPENDCUBE_MOCK_LATENCY", "0") or 0) * float(os.environ.get("SPENDCUBE_MOCK_SLOW", "1") or 1)
        if lat > 0:      # timing dress runs: sleep at the real per-call latency (rate limits are enforced by the router)
            time.sleep(lat)
        res = ProviderResult(usage={"in": len(system + user) // 4, "out": 10, "thinking": 0})
        if self.fail_every and self.calls % self.fail_every == 0:
            res.error, res.error_kind = "mock failure", "server"
            return res
        props = (schema or {}).get("properties", {})
        leaves = {}
        for line in (system + "\n" + user).splitlines():
            m = re.match(r"^(\d+)\|([^|]+)(?:\|(.*))?$", line)
            if m and ">" in m.group(2) or (m and "d" in props):
                leaves[int(m.group(1))] = (m.group(2) + " " + (m.group(3) or "")).upper()
        if "d" in props:
            res.parsed = {"d": [{"n": n, "def": "mock definition", "ex": "mock"} for n in leaves]}
            return res
        rows = []
        sys_leaves = {}
        for line in system.splitlines():
            m = re.match(r"^(\d+)\|([^|]+)", line)
            if m and ">" in line:
                sys_leaves[int(m.group(1))] = m.group(2).upper()
        for line in user.splitlines():
            parts = line.split("|")
            if len(parts) < 5 or not parts[0].isdigit():
                continue
            toks = set(re.findall(r"[A-Z0-9]{3,}", parts[4].upper()))
            best = max(sys_leaves or {1: ""}, key=lambda n: (len(toks & set(re.findall(r"[A-Z0-9]{3,}", sys_leaves.get(n, "")))), -n))
            rows.append([int(parts[0]), best, 50, 0])
        res.parsed = {"r": rows}
        res.raw_text = str(res.parsed)
        return res
