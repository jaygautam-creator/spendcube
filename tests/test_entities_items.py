"""Item resolution: synthetic descriptions only."""
import pandas as pd

from spendcube.entities.items import build_item_alias, detect_generic, resolve_items, text_key
from spendcube.normalize.text import add_text_columns


def frame(rows):
    """rows: list of (material, description, source_role)."""
    df = pd.DataFrame({
        "material_id": [r[0] for r in rows],
        "desc_raw": [r[1] for r in rows],
        "source_role": [r[2] if len(r) > 2 else "primary" for r in rows],
        "gl_account": [""] * len(rows),
    })
    return add_text_columns(df, None)


def items(df, **kw):
    sk = pd.Series(["S:X"] * len(df), index=df.index)
    k, c, st = resolve_items(df, kw.pop("cfg", None), sk, primary_mask=(df["source_role"] == "primary"), **kw)
    return list(k), list(c), st


def test_text_key_unifies_decimal_comma_and_ocr():
    assert text_key("WD:2,2M") == text_key("WD:2.2M")
    assert text_key("DYNAM1TE") == text_key("DYNAMITE")
    assert text_key("") == ""


def test_material_key_and_generic_detection():
    rows = [("M1", "VALVE BALL 2IN")] * 3 + [("MGEN", f"FREE TEXT NUMBER {i} ITEM") for i in range(12)]
    df = frame(rows)
    assert detect_generic(df["material_id"], df["desc_compact"], {}) == {"MGEN"}
    k, _, st = items(df)
    assert k[0] == "M:M1" and k[1] == k[0]
    assert not any(x == "M:MGEN" for x in k)           # generic id is never item identity
    assert len(set(k[3:])) == 12                          # different texts stay different items


def test_free_text_rows_link_to_material_with_same_text_variants():
    rows = [("M1", "PUMP SEAL KIT 40MM"), ("M1", "PUMP SEAL KIT 40MM"),
            ("", "pump seal  kit 40mm", "secondary"),       # case + spacing
            ("", "PUMP SEAL K1T 40MM", "secondary"),         # OCR noise
            ("", "UNRELATED THING", "secondary")]
    k, _, _ = items(frame(rows))
    assert k[2] == k[3] == k[0] == "M:M1"
    assert k[4].startswith("D:")


def test_strict_fuzzy_never_merges_different_numbers():
    rows = [("", "BOLT ANCHR;DIA:20MM,1800MM,XX"), ("", "BOLT ANCHR;DIA:20MM,1500MM,XX")]
    k, _, _ = items(frame(rows))
    assert k[0] != k[1]


def test_strict_fuzzy_merges_dropped_letter_but_loose_flag_exists():
    rows = [("", "HYDRAULIC CYLINDER ASSEMBLY HEAVY"), ("", "HYDRAULIC CYLINDER ASSEMBLY HEAVY"),
            ("", "HYDRAULIC CYLINDER ASSEMBLY HEAVYX"), ("", "HYDRAULIC CYLINDER ASSEMBLY HEAVYX")]
    k, _, st = items(frame(rows))
    assert len(set(k)) == 1 and st["fuzzy_merges"] == 1
    k2, _, _ = items(frame(rows), cfg={"fuzzy": False})
    assert len(set(k2)) == 2


def test_blank_description_without_material_is_unknown_key():
    k, c, _ = items(frame([("", "   "), ("", "N/A")]))
    assert all(x.startswith("U:") for x in k) and max(c) < 0.5


def test_item_alias_maps_material_and_text():
    rows = pd.DataFrame({
        "material_id": ["M1", "M2", "M3", "M1"], "text_key": ["T", "REPAIRFAN", "REPAIRFAN", "T"],
        "gold_item": ["G1", "GR", "GR", "G1"]})
    mat, txt = build_item_alias(rows, 0.9)
    assert mat == {"M1": "G1", "M2": "GR", "M3": "GR"}
    assert txt == {"REPAIRFAN": "GR"}                      # one text over >=2 materials under one key
    df = frame([("M9", "REPAIR FAN"), ("M1", "ANYTHING")])
    k, _, _ = items(df, mat_alias=mat, txt_alias=txt)
    assert k == ["GR", "G1"]


def test_generic_material_with_alias_gets_golden_catch_all():
    df = frame([("MGEN", "RANDOM TEXT ONE"), ("MGEN", "RANDOM TEXT TWO")])
    k, _, _ = items(df, mat_alias={"MGEN": "CATCH"})
    assert k == ["CATCH", "CATCH"]


def test_merge_materials_by_text_is_optional():
    rows = [("M1", "SAME DESCRIPTION OF THING"), ("M2", "SAME DESCRIPTION OF THING")]
    assert len(set(items(frame(rows))[0])) == 2
    assert len(set(items(frame(rows), cfg={"merge_materials_by_text": True})[0])) == 1
