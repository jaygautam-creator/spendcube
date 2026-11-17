import pandas as pd
from spendcube.schema_cols import OUTPUT_COLUMNS
from spendcube.output.validator import validate_frame
from spendcube.taxonomy.tree import Taxonomy

TAX = Taxonomy([("A", "B", "C"), ("A", "D")])


def row(**kw):
    base = dict(record_id="r1", include="1", exclusion_reason="", supplier_key="s", item_key="i",
                spend=10.0, l1="A", l2="B", l3="C", l4="", confidence=0.5)
    base.update(kw)
    return base


def df(*rows):
    return pd.DataFrame(list(rows))[OUTPUT_COLUMNS]


def test_ok():
    assert validate_frame(df(row(), row(record_id="r2", include="0", exclusion_reason="DUPLICATE",
                                        spend=0, l1="", l2="", l3="", confidence=0.9)), TAX, ["r1", "r2"]) == []


def test_catches_problems():
    assert validate_frame(df(row(l3="Z")), TAX)                                   # path not in tree
    assert validate_frame(df(row(include="0", exclusion_reason="DUPLICATE")), TAX)  # spend/cats on excluded
    assert validate_frame(df(row(exclusion_reason="CANCELLED")), TAX)             # reason on included
    assert validate_frame(df(row(l1="A", l2="D", l3="C")), TAX)                   # wrong path
    assert validate_frame(df(row(confidence=1.5)), TAX)
    assert validate_frame(df(row(supplier_key="")), TAX)
    assert validate_frame(df(row()), TAX, ["r1", "r2"])                           # count mismatch
    assert validate_frame(df(row(l1="A", l2="D", l3="", l4="")), TAX) == []       # shallower leaf ok
