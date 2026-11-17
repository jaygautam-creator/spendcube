"""Token-frugal prompts: numbered leaves `n|L1 > L2 > L3 > L4|definition`; units `uid|supplier|gl|ctx|text`.

Output contract: {"r":[[uid, leaf_no, conf_0_100, spend_class_code], ...]} (all integers).
"""
from __future__ import annotations

PROMPT_VERSION = "v2"

SPEND_CLASSES = {0: "procurement", 1: "tax", 2: "intercompany", 3: "payroll_people",
                 4: "financing_interest", 5: "revenue_customer", 6: "royalty_partner", 7: "other_non_procurement"}

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"r": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}}},
    "required": ["r"],
}

COMPILE_SCHEMA = {
    "type": "object",
    "properties": {"d": {"type": "array", "items": {
        "type": "object",
        "properties": {"n": {"type": "integer"}, "def": {"type": "string"}, "ex": {"type": "string"}},
        "required": ["n", "def"]}}},
    "required": ["d"],
}

SYSTEM_TMPL = """You classify purchase-ledger lines into a category tree for procurement spend analysis.
{industry}Descriptions are short, abbreviated (SAP-style), may contain OCR noise and may be in any language.
Use every clue: description, supplier, GL text and the context key=value pairs (e.g. spend type, material type).
Pick the single best LEAF number for each line and always choose a leaf; give confidence 0-100.
Also give a spend class code: {classes}. Use 0 unless the line is clearly not a purchase from a third-party supplier.
Input lines: uid|supplier|gl|context|description, optionally ending with cand=a,b (two earlier suggestions: pick the better one unless both are clearly wrong).
Output ONLY JSON: {{"r":[[uid,leaf,confidence,class],...]}} with one entry per uid, all integers.

LEAVES (number|path|definition):
{leaves}"""

COMPILE_SYSTEM = """You are a procurement taxonomy expert. For each requested category leaf write a precise scope definition
(max 25 words), typical items/abbreviations/synonyms a ledger line would contain (max 15 words, may include abbreviations
and non-English terms), and, when sibling leaves are easily confused, what distinguishes this one ("not X: ..."), inside the definition.
Ground your answer in the company's own sample descriptions where relevant.
Output ONLY JSON: {"d":[{"n":leaf_number,"def":"...","ex":"..."}]} covering every requested number."""


def leaf_lines(taxonomy, definitions: dict[str, str] | None = None, only: list[int] | None = None) -> str:
    out = []
    for i, l in enumerate(taxonomy.leaves, 1):
        if only is not None and i not in only:
            continue
        d = (definitions or {}).get(l.id, "")
        out.append(f"{i}|{' > '.join(l.path)}" + (f"|{d}" if d else ""))
    return "\n".join(out)


def build_system(taxonomy, industry_hint: str = "", definitions: dict[str, str] | None = None) -> str:
    classes = ", ".join(f"{k}={v}" for k, v in SPEND_CLASSES.items())
    ind = f"Company context: {industry_hint}\n" if industry_hint else ""
    return SYSTEM_TMPL.format(industry=ind, classes=classes, leaves=leaf_lines(taxonomy, definitions))


def unit_line(uid: int, supplier: str, gl: str, ctx: str, text: str, max_chars: int = 160,
              cand: tuple[int, ...] | None = None) -> str:
    t = " ".join(str(text).split())[:max_chars].replace("|", "/")
    s = f"{uid}|{str(supplier)[:40]}|{str(gl)[:40]}|{str(ctx)[:140]}|{t}"
    if cand:
        s += "|cand=" + ",".join(str(c) for c in cand)
    return s


def build_user(lines: list[str]) -> str:
    return "\n".join(lines)
