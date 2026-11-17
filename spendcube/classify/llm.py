"""LLM batch classifier: builds prompts, validates answers, bisects bad batches, runs batches in parallel."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

from ..llm.prompts import RESPONSE_SCHEMA, SPEND_CLASSES, build_system, build_user, unit_line

log = logging.getLogger("spendcube.llmclf")


@dataclass
class UnitResult:
    uid: int
    leaf_id: str
    conf: float            # 0..1 as self-reported
    spend_class: int
    entry: str


class LLMClassifier:
    def __init__(self, router, taxonomy, industry_hint: str = "", definitions: Optional[dict] = None,
                 llm_cfg: Optional[dict] = None):
        self.router, self.tax = router, taxonomy
        self.cfg = llm_cfg or {}
        self.system = build_system(taxonomy, industry_hint, definitions)
        bc = self.cfg.get("batch", {})
        self.batch_max = int(bc.get("max", 60))
        self.arbiter_batch = int(bc.get("arbiter", 40))
        self.max_chars = int(self.cfg.get("max_desc_chars", 160))
        self.thinking = self.cfg.get("thinking", {})
        self.nleaf = len(taxonomy)

    def _validator(self, uids: set[int]):
        def v(p):
            if not isinstance(p, dict) or not isinstance(p.get("r"), list):
                return False, "no r list"
            seen = set()
            for row in p["r"]:
                if not (isinstance(row, list) and len(row) >= 3 and all(isinstance(x, (int, float)) for x in row[:3])):
                    return False, "bad row shape"
                u, n = int(row[0]), int(row[1])
                if u not in uids or u in seen:
                    return False, f"unknown/duplicate uid {u}"
                if not 1 <= n <= self.nleaf:
                    return False, f"leaf {n} out of range 1..{self.nleaf}"
                seen.add(u)
            if seen != uids:
                return False, f"missing uids {sorted(uids - seen)[:8]}"
            return True, ""
        return v

    def _run(self, units: list[dict], role: str, exclude: tuple, depth: int = 0, prefix: str = "") -> dict[int, UnitResult]:
        if not units:
            return {}
        local = {i + 1: u for i, u in enumerate(units)}
        lines = [unit_line(i, u.get("supplier", ""), u.get("gl", ""), u.get("ctx", ""), u["text"],
                           self.max_chars, u.get("cand")) for i, u in local.items()]
        out = self.router.call(role, self.system, prefix + build_user(lines), RESPONSE_SCHEMA,
                               self._validator(set(local)), thinking=self.thinking.get(role),
                               max_tokens=min(8192, 200 + 24 * len(units)), exclude_entries=exclude)
        if out is None:
            if len(units) == 1 or depth >= 3:
                return {}
            mid = len(units) // 2
            r = self._run(units[:mid], role, exclude, depth + 1, prefix)
            r.update(self._run(units[mid:], role, exclude, depth + 1, prefix))
            return r
        res = {}
        for row in out.parsed["r"]:
            u = local[int(row[0])]
            cls = int(row[3]) if len(row) > 3 and int(row[3]) in SPEND_CLASSES else 0
            res[u["uid"]] = UnitResult(u["uid"], self.tax.leaves[int(row[1]) - 1].id,
                                       max(0.0, min(1.0, float(row[2]) / 100.0)), cls, out.entry_id)
        return res

    def classify(self, units: list[dict], role: str = "bulk", exclude_entries: tuple = (),
                 workers: int = 4, user_prefix: str = "") -> dict[int, UnitResult]:
        """units: dicts with uid (global id), supplier, gl, ctx, text, optional cand (leaf numbers)."""
        size = self.arbiter_batch if role == "arbiter" else self.batch_max
        batches = [units[i:i + size] for i in range(0, len(units), size)]
        results: dict[int, UnitResult] = {}
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for r in ex.map(lambda b: self._run(b, role, exclude_entries, 0, user_prefix), batches):
                results.update(r)
        log.info("llm %s: %d/%d units classified", role, len(results), len(units))
        return results
