import pandas as pd
from spendcube.taxonomy.tree import Taxonomy, detect_format, _from_wide, _from_path, _from_adjacency


def test_wide_ragged_and_duplicate_names():
    df = pd.DataFrame({"category_l1": ["Fruit", "Fruit", "Veg", "Veg"],
                       "category_l2": ["Apples", "Pears", "Roots", "Roots"],
                       "category_l3": ["Red", "", "Carrot", "Carrot"],
                       "category_l4": ["", "", "", ""]})
    assert detect_format(df, " > ") == "wide"
    t = Taxonomy(_from_wide(df))
    assert len(t) == 3                    # duplicate path collapsed
    assert t.depth == 3
    paths = {l.path for l in t.leaves}
    assert ("Fruit", "Pears") in paths and ("Fruit", "Apples", "Red") in paths
    lid = t.leaf_by_path(("Fruit", "Pears")).id
    assert t.path_padded(lid) == ["Fruit", "Pears", "", ""]


def test_names_with_commas_and_same_name_different_parent():
    df = pd.DataFrame({"l1": ["A", "B"], "l2": ["Parts, Spares", "Parts, Spares"], "l3": ["X", "X"]})
    t = Taxonomy(_from_wide(df))
    assert len(t) == 2 and t.path_padded(t.leaves[0].id)[1] == "Parts, Spares"


def test_path_format_and_autodetect():
    df = pd.DataFrame({"path": ["Goods > Food > Dairy", "Goods > Food > Meat > Beef", "Services"]})
    assert detect_format(df, " > ") == "path"
    t = Taxonomy(_from_path(df, " > "))
    assert t.depth == 4 and len(t) == 3


def test_adjacency():
    df = pd.DataFrame({"id": ["1", "2", "3", "4"], "parent": ["", "1", "1", "2"],
                       "name": ["Root", "Left", "Right", "LL"]})
    assert detect_format(df, " > ") == "adjacency"
    t = Taxonomy(_from_adjacency(df))
    assert {l.path for l in t.leaves} == {("Root", "Right"), ("Root", "Left", "LL")}


def test_leaf_by_names_casefold():
    t = Taxonomy([("Alpha", "Beta")])
    assert t.leaf_by_names([" alpha", "BETA "]).path == ("Alpha", "Beta")


def test_parent_code_adjacency_and_code_columns():
    df = pd.DataFrame({"code": ["1", "2", "3"], "name": ["Root", "Child A", "Child B"], "parent_code": ["", "1", "1"]})
    assert detect_format(df, " > ") == "adjacency"
    t = Taxonomy(_from_adjacency(df))
    assert {l.path for l in t.leaves} == {("Root", "Child A"), ("Root", "Child B")}
    # code column followed by free-named level columns: the code is not an L1
    df2 = pd.DataFrame({"Code": ["10", "11", "12"], "Segment": ["Goods", "Goods", "Services"],
                        "Family": ["Food", "Tools", "Legal"], "Class": ["Dairy", "Hammers", "Advice"]})
    t2 = Taxonomy(_from_wide(df2))
    assert t2.leaves[0].path == ("Goods", "Food", "Dairy")


def test_swiss_apostrophe_numbers():
    from spendcube.normalize.numbers import parse_number
    assert parse_number("CHF 1'234.56") == 1234.56 and parse_number("1’234.50") == 1234.50
