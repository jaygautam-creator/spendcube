"""Role-based LLM router: ordered chains, per-entry limiter + ledger RPD, cache, repair retry, fail-soft.

Nothing raises out of `call`; unplaceable requests return None so callers fall to lower tiers.
"""
from __future__ import annotations

import logging
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .cache import Cache, make_key
from .keys import key_present, resolve_key_env
from .ledger import Ledger
from .limiter import Limiter
from .prompts import PROMPT_VERSION

log = logging.getLogger("spendcube.llm")

Validator = Callable[[Any], tuple[bool, str]]


@dataclass
class Entry:
    id: str
    provider: str
    model: str
    key_env: Optional[str]
    rpm: Optional[int] = None
    tpm: Optional[int] = None
    rpd: Optional[int] = None
    optional: bool = False
    base_url: str = ""
    limiter: Limiter = field(default_factory=Limiter)
    profile: str = ""


@dataclass
class CallOutcome:
    parsed: Any
    entry_id: str
    model: str
    usage: dict
    cached: bool = False
    repaired: bool = False


class Router:
    def __init__(self, llm_cfg: dict, mode: str = "dev", enabled: bool = True, cache: Optional[Cache] = None,
                 ledger: Optional[Ledger] = None, job_budget: bool = False):
        self.cfg = llm_cfg or {}
        self.mode = mode
        self.enabled = enabled and bool(self.cfg.get("enabled", True))
        self.cache = cache or Cache(self.cfg.get("cache_file", ".cache/llm_cache.sqlite"))
        self.ledger = ledger or Ledger(self.cfg.get("ledger_file", ".cache/llm_ledger.sqlite"))
        self.timeout = float(self.cfg.get("timeout_s", 120))
        self.entries: dict[str, Entry] = {}
        self.roles: dict[str, list[str]] = self.cfg.get("roles", {})
        self.dead: dict[str, str] = {}
        self._providers: dict[tuple, Any] = {}
        self._lock = threading.Lock()
        self.stats: dict[str, Any] = {"calls": 0, "cache_hits": 0, "tok_in": 0, "tok_out": 0,
                                      "repairs": 0, "failures": 0, "by_entry": {}}
        for e in self.cfg.get("chain", []):
            ent = Entry(id=e["id"], provider=e["provider"], model=e["model"], key_env=e.get("key_env"),
                        rpm=e.get("rpm"), tpm=e.get("tpm"), rpd=e.get("rpd"), optional=bool(e.get("optional")),
                        base_url=e.get("base_url", ""))
            ent.limiter = Limiter(ent.rpm, ent.tpm)
            ent.profile = resolve_key_env(ent.key_env, mode) or ent.provider
            self.entries[ent.id] = ent
        self.budget: Optional[int] = None
        self.spent = 0
        self.budget_info: dict = {}
        if self.enabled and job_budget:
            self._init_budget()

    # ---- per-job request budget (D-42) -------------------------------------
    def pool_remaining(self, util_cap: Optional[float] = None) -> dict[str, int]:
        """Remaining requests today (up to util_cap x RPD) per budget-pool entry that has a key."""
        b = self.cfg.get("budget") or {}
        cap = float(util_cap if util_cap is not None else b.get("util_cap", 0.7))
        pool = b.get("pool") or [i for i, e in self.entries.items() if e.provider == "gemini" and e.rpd and e.rpd <= 1000]
        out = {}
        for i in pool:
            e = self.entries.get(i)
            if e is None or e.rpd is None or not self._usable(e):
                continue
            out[i] = max(0, int(cap * e.rpd) - self.ledger.requests_today(e.profile, e.id))
        return out

    def _init_budget(self) -> None:
        b = self.cfg.get("budget") or {}
        if not b or not b.get("enabled", True):
            return
        rem = self.pool_remaining()
        if b.get("fixed") is not None:       # evaluation: an exact request budget (curves), independent of the ledger
            rem = rem or {i: 0 for i in (b.get("pool") or [])}
        if not rem:      # no rate-limited pool (mock / local / Groq-only): nothing to budget
            return
        avail = sum(rem.values())
        share = float(b.get("share", 0.6))
        n = int(share * max(0, avail - int(b.get("reserve", 0))))
        if b.get("max_requests") is not None:
            n = min(n, int(b["max_requests"]))
        if b.get("fixed") is not None:
            n = int(b["fixed"])
        self.count_cached = bool(b.get("count_cached", False))
        self.budget = max(0, n)
        self.exempt_roles = set(b.get("exempt_roles", ["compile", "arbiter"]))
        self.pool_ids = set(rem) if rem else set()
        self.budget_info = {"share": share, "pool_remaining_at_start": rem, "budget": self.budget}
        log.info("LLM job budget: %d requests (share %.2f of %d remaining in pool %s)", self.budget, share, avail, sorted(rem))

    exempt_roles: set = {"compile", "arbiter"}
    pool_ids: set = set()
    count_cached: bool = False      # replay mode: a cache hit on a pool entry spends budget like a live request

    def budget_exhausted(self) -> bool:
        return self.budget is not None and self.spent >= self.budget

    # ---- availability -----------------------------------------------------
    def _usable(self, e: Entry) -> bool:
        if e.id in self.dead:
            return False
        if e.provider == "mock":
            return True
        if e.provider == "ollama":
            return self._ollama_up(e)
        return key_present(e.key_env, self.mode)

    def _ollama_up(self, e: Entry) -> bool:
        key = ("ollama_up", e.id)
        if key not in self._providers:
            try:
                import httpx
                ok = httpx.get((e.base_url or "http://localhost:11434") + "/api/tags", timeout=2).status_code == 200
                if ok:
                    names = [m["name"] for m in httpx.get((e.base_url or "http://localhost:11434") + "/api/tags", timeout=2).json().get("models", [])]
                    ok = any(n == e.model or n.startswith(e.model.split(":")[0]) for n in names) if names else False
            except Exception:
                ok = False
            self._providers[key] = ok
        return bool(self._providers[key])

    def chain(self, role: str) -> list[Entry]:
        if not self.enabled:
            return []
        if role not in self.exempt_roles and self.budget_exhausted():
            return []
        ids = list(self.roles.get(role) or list(self.entries))
        if self.cfg.get("allow_outage_models", False):
            ids += [i for i in self.cfg.get("outage_chain", []) if i not in ids]
        return [self.entries[i] for i in ids if i in self.entries and self._usable(self.entries[i])]

    def has_llm(self, role: str) -> bool:
        return bool(self.chain(role))

    def _provider(self, e: Entry):
        k = (e.provider, e.profile, e.base_url)
        with self._lock:
            if k not in self._providers:
                if e.provider == "gemini":
                    from .providers.gemini import make_gemini
                    self._providers[k] = make_gemini(resolve_key_env(e.key_env, self.mode))
                elif e.provider == "groq":
                    from .providers.groq import GroqProvider
                    self._providers[k] = GroqProvider(os.environ.get(e.key_env or "GROQ_API_KEY", ""))
                elif e.provider == "ollama":
                    from .providers.ollama import OllamaProvider
                    self._providers[k] = OllamaProvider(e.base_url or "http://localhost:11434")
                elif e.provider == "mock":
                    from .providers.mock import MockProvider
                    self._providers[k] = MockProvider()
                else:
                    self._providers[k] = None
            return self._providers[k]

    def _mark_dead(self, e: Entry, why: str) -> None:
        with self._lock:
            if e.id not in self.dead:
                self.dead[e.id] = why
                log.warning("LLM entry %s disabled for this run: %s", e.id, why)

    # ---- main call ---------------------------------------------------------
    def call(self, role: str, system: str, user: str, schema: dict, validate: Validator, *,
             thinking: Optional[str] = None, max_tokens: int = 4096, exclude_entries: tuple = (),
             max_wait: float = 120.0) -> Optional[CallOutcome]:
        est = (len(system) + len(user)) // 3 + max_tokens // 4
        for e in self.chain(role):
            if e.id in exclude_entries:
                continue
            ck = make_key(e.provider, e.model, PROMPT_VERSION, system, user, thinking or "")
            hit = self.cache.get(ck)
            if hit is not None:
                ok, _ = validate(hit["parsed"])
                if ok:
                    self.stats["cache_hits"] += 1
                    if self.count_cached and role not in self.exempt_roles and e.id in self.pool_ids:
                        with self._lock:
                            self.spent += 1
                    return CallOutcome(hit["parsed"], e.id, e.model, hit.get("usage", {}), cached=True)
            out = self._try_entry(e, role, system, user, schema, validate, thinking, max_tokens, est, max_wait, ck)
            if out is not None:
                return out
        self.stats["failures"] += 1
        return None

    def _try_entry(self, e, role, system, user, schema, validate, thinking, max_tokens, est, max_wait, ck):
        prov = self._provider(e)
        if prov is None:
            self._mark_dead(e, "provider unavailable")
            return None
        cur_user, repaired = user, False
        for attempt in range(4):
            if e.id in self.dead:
                return None
            if role not in self.exempt_roles and e.id in self.pool_ids and self.budget_exhausted():
                return None
            if e.rpd is not None and self.ledger.requests_today(e.profile, e.id) >= e.rpd:
                self._mark_dead(e, f"daily request cap {e.rpd} reached")
                return None
            if not e.limiter.acquire(est, max_wait):
                log.info("limiter wait too long for %s; trying next entry", e.id)
                return None
            r = prov.complete_json(system, cur_user, schema, e.model, max_tokens, thinking, self.timeout)
            ok_call = r.parsed is not None and r.error is None
            self.ledger.record(e.profile, e.id, r.usage.get("in", 0), r.usage.get("out", 0) + r.usage.get("thinking", 0),
                               ok_call, r.error_kind)
            with self._lock:
                if role not in self.exempt_roles and e.id in self.pool_ids:
                    self.spent += 1
                self.stats["calls"] += 1
                self.stats["tok_in"] += r.usage.get("in", 0)
                self.stats["tok_out"] += r.usage.get("out", 0) + r.usage.get("thinking", 0)
                be = self.stats["by_entry"].setdefault(e.id, {"calls": 0, "tok_in": 0, "tok_out": 0, "errors": 0})
                be["calls"] += 1
                be["tok_in"] += r.usage.get("in", 0)
                be["tok_out"] += r.usage.get("out", 0) + r.usage.get("thinking", 0)
                if not ok_call:
                    be["errors"] += 1
            if ok_call:
                valid, why = validate(r.parsed)
                if valid:
                    self.cache.put(ck, {"parsed": r.parsed, "usage": r.usage, "model": e.model})
                    return CallOutcome(r.parsed, e.id, e.model, r.usage, repaired=repaired)
                if not repaired:   # one repair retry with the reason
                    repaired = True
                    self.stats["repairs"] += 1
                    cur_user = user + f"\n\nYour previous answer was invalid ({why}). Return the corrected JSON only."
                    continue
                log.info("%s: invalid after repair (%s)", e.id, why)
                return None
            kind = r.error_kind
            if kind in ("rate_limit",):
                wait = r.retry_after if r.retry_after is not None else 10.0
                if wait <= 20 and attempt < 3:
                    e.limiter.penalise(wait)
                    time.sleep(min(wait + 1, 21))
                    continue
                self._mark_dead(e, f"rate limited (retry in {wait}s)")
                return None
            if kind == "quota":
                self._mark_dead(e, "quota exhausted")
                return None
            if kind == "auth":
                self._mark_dead(e, f"auth error: {(r.error or '')[:80]}")
                return None
            if kind in ("server", "timeout") and attempt < 1:   # back off once (jittered), then fail over (D-28c)
                time.sleep(random.uniform(2.0, 8.0))
                continue
            if kind == "parse" and not repaired:
                repaired = True
                self.stats["repairs"] += 1
                cur_user = user + "\n\nReturn valid minified JSON only."
                continue
            log.info("%s failed (%s): %s", e.id, kind, (r.error or "")[:120])
            return None
        return None

    def report(self) -> dict:
        return {**self.stats, "dead": dict(self.dead), "cache_hits_total": self.cache.hits,
                "budget": {**self.budget_info, "spent": self.spent, "exhausted": self.budget_exhausted()}}
