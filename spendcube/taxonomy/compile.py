"""Taxonomy compilation: per-leaf definitions + typical items + sibling disambiguation from the tree,
industry hint and a sample of the company's own descriptions. Cached through the router cache."""
from __future__ import annotations

import logging
from typing import Optional

from ..llm.prompts import COMPILE_SCHEMA, COMPILE_SYSTEM, leaf_lines

log = logging.getLogger("spendcube.compile")


def _chunks(taxonomy, max_leaves: int = 45) -> list[list[int]]:
    """Group leaf numbers (1-based) by L1 so siblings share a call; pack up to max_leaves."""
    groups: dict[str, list[int]] = {}
    for i, l in enumerate(taxonomy.leaves, 1):
        groups.setdefault(l.path[0], []).append(i)
    chunks, cur = [], []
    for g in groups.values():
        if cur and len(cur) + len(g) > max_leaves:
            chunks.append(cur); cur = []
        cur += g
        while len(cur) > max_leaves:
            chunks.append(cur[:max_leaves]); cur = cur[max_leaves:]
    if cur:
        chunks.append(cur)
    return chunks


def compile_taxonomy(router, taxonomy, industry_hint: str, sample_texts: list[str],
                     max_sample: int = 120) -> dict[str, str]:
    """Returns {leaf_id: definition}. Empty dict if no LLM is available (callers fall back)."""
    if not router.has_llm("compile"):
        return {}
    sample = "\n".join(f"- {t[:90]}" for t in sample_texts[:max_sample])
    full_tree = leaf_lines(taxonomy)
    defs: dict[str, str] = {}
    for chunk in _chunks(taxonomy):
        want = set(chunk)
        user = (f"Company context: {industry_hint or 'unknown'}\n\nFULL TREE (number|path):\n{full_tree}\n\n"
                f"SAMPLE OF THE COMPANY'S OWN LINE DESCRIPTIONS:\n{sample}\n\n"
                f"Write definitions for these leaf numbers only: {','.join(map(str, chunk))}")

        def validate(p):
            if not isinstance(p, dict) or not isinstance(p.get("d"), list):
                return False, "missing d"
            got = {int(x.get("n", -1)) for x in p["d"] if isinstance(x, dict) and str(x.get("n", "")).lstrip("-").isdigit()}
            miss = want - got
            return (len(miss) <= max(1, len(want) // 10), f"missing leaves {sorted(miss)[:10]}")

        out = router.call("compile", COMPILE_SYSTEM, user, COMPILE_SCHEMA, validate,
                          thinking="low", max_tokens=8192)
        if out is None:
            log.warning("taxonomy compilation failed for a chunk; continuing without those definitions")
            continue
        for x in out.parsed["d"]:
            try:
                n = int(x["n"])
            except Exception:
                continue
            if n in want:
                d = str(x.get("def", "")).strip()
                ex = str(x.get("ex", "")).strip()
                defs[taxonomy.leaves[n - 1].id] = (d + (f" e.g. {ex}" if ex else ""))[:260]
    log.info("compiled definitions for %d/%d leaves", len(defs), len(taxonomy))
    return defs
