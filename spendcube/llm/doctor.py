"""`spendcube doctor [--blind]`: key presence (set/missing only), model resolution, one tiny JSON call per entry."""
from __future__ import annotations

import os
import sys

from ..config.loader import load_config
from .keys import key_present, resolve_key_env
from .prompts import RESPONSE_SCHEMA


def run_doctor(config_path: str, mode: str, list_models: bool = True, generate: bool = True) -> int:
    """generate=False (`doctor --list-only`): key presence, models.list (metadata) and local ledger reads only;
    NO generate_content request is made on any key. generate=True makes ONE tiny request per Gemini entry (the
    Thursday 09:00 blind check) and records each in the quota ledger."""
    cfg = load_config(config_path)
    llm = cfg.classification.get("llm", {})
    from .safety import mock_active
    if mock_active(llm):
        print("SPENDCUBE MOCK ACTIVE: mock_active: true", file=sys.stderr)
    chain = llm.get("chain", [])
    print(f"== spendcube doctor (mode={mode}) ==")
    for env in ("GEMINI_API_KEY_DEV", "GEMINI_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY_PAID"):
        print(f"  {env}: {'set' if os.environ.get(env, '').strip() else 'missing'}")
    if mode == "blind":
        print("  (blind mode: ONE tiny call per model only)" if generate else
              "  (blind mode, --list-only: metadata calls only, ZERO generate requests)")
    gem_env = resolve_key_env("GEMINI_API_KEY", mode)
    available: set[str] = set()
    if list_models and key_present("GEMINI_API_KEY", mode):
        from .providers.gemini import make_gemini
        prov = make_gemini(gem_env)
        try:
            available = {m.name.split("/", 1)[-1] for m in prov.client.models.list()}
            print(f"  gemini models visible: {len(available)}")
        except Exception as e:
            print(f"  model listing failed: {str(e)[:150]}")
    _print_quota(llm, mode)
    bad = 0
    for e in chain:
        if e.get("provider") != "gemini":
            ok = key_present(e.get("key_env"), mode) if e.get("provider") != "ollama" else _ollama_up(e)
            print(f"  [{e['id']}] provider={e['provider']} {'available' if ok else 'unavailable (optional)'}")
            continue
        if not key_present(e.get("key_env"), mode):
            print(f"  [{e['id']}] KEY MISSING"); bad += 1; continue
        if available and e["model"] not in available:
            print(f"  [{e['id']}] model id '{e['model']}' NOT in model list"); bad += 1; continue
        if not generate:
            print(f"  [{e['id']}] {e['model']}: key present, model listed (no generate request made)")
            continue
        from .providers.gemini import make_gemini
        prov = make_gemini(resolve_key_env(e["key_env"], mode))
        r = prov.complete_json("Reply with JSON only.", 'Return {"r":[[1,2,90,0]]}', RESPONSE_SCHEMA,
                               e["model"], max_tokens=512, thinking="low", timeout=45)
        try:      # doctor's own request counts against the day's quota like any other
            from .ledger import Ledger
            Ledger(llm.get("ledger_file", ".cache/llm_ledger.sqlite")).record(
                resolve_key_env(e["key_env"], mode) or "", e["id"], r.usage["in"], r.usage["out"], bool(r.parsed), r.error_kind)
        except Exception:
            pass
        status = "ok" if r.parsed else f"FAIL {r.error_kind}: {(r.error or '')[:120]}"
        if not r.parsed:
            bad += 1
        print(f"  [{e['id']}] {e['model']}: {status}  {r.latency_s}s tokens in/out={r.usage['in']}/{r.usage['out']}")
    return 1 if bad else 0


def _ollama_up(e) -> bool:
    try:
        import httpx
        return httpx.get(e.get("base_url", "http://localhost:11434") + "/api/tags", timeout=2).status_code == 200
    except Exception:
        return False


def _print_quota(llm: dict, mode: str) -> None:
    """Remaining RPD per model (from the local ledger, no API call) and the sequential per-job budgets (D-42)."""
    from .router import Router
    r = Router(dict(llm, enabled=True), mode=mode)
    print("  -- quota today (Pacific day), from the local ledger --")
    for e in r.entries.values():
        if e.rpd is None or not r._usable(e):
            continue
        used = r.ledger.requests_today(e.profile, e.id)
        print(f"  [{e.id}] used {used}/{e.rpd}  remaining {max(0, e.rpd - used)}  (profile {e.profile})")
    b = llm.get("budget") or {}
    plan = b.get("blind_plan") or {}
    rem = r.pool_remaining()
    avail = sum(rem.values())
    print(f"  -- job budgets: pool {sorted(rem)} remaining under {int(100 * float(b.get('util_cap', 0.7)))}% cap = {avail} requests --")
    left = avail
    for job in ("C", "B2", "A2"):
        pj = plan.get(job, {}) if isinstance(plan.get(job), dict) else {}
        n = int(float(pj.get("share", 1.0)) * max(0, left - int(pj.get("reserve", 0))))
        if pj.get("max") is not None:
            n = min(n, int(pj["max"]))
        print(f"  job {job}: share {pj.get('share', 1.0)} reserve {pj.get('reserve', 0)} max {pj.get('max', '-')} -> budget {n} requests")
        left -= n
