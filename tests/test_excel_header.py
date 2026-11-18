"""Excel inputs with title rows above the header (synthetic data only)."""
import pandas as pd

from spendcube.ingest.readers import read_table


def _write(path, rows):
    pd.DataFrame(rows).to_excel(path, header=False, index=False)


def test_plain_excel_unchanged(tmp_path):
    p = tmp_path / "plain.xlsx"
    _write(p, [["line_id", "amount", "description"], ["L1", "10", "bolts"], ["L2", "20", "nuts"]])
    df = read_table(p)
    assert list(df.columns) == ["line_id", "amount", "description"]
    assert len(df) == 2


def test_title_rows_skipped(tmp_path):
    p = tmp_path / "title.xlsx"
    _write(p, [["Invoice line export", "", ""], ["", "", ""],
               ["line_id", "amount", "description"], ["L1", "10", "bolts"], ["L2", "20", "nuts"]])
    df = read_table(p)
    assert list(df.columns) == ["line_id", "amount", "description"]
    assert df["line_id"].tolist() == ["L1", "L2"]


def test_sparse_but_real_header_kept(tmp_path):
    # a real header whose data has blank cells must not be mistaken for a title row
    p = tmp_path / "sparse.xlsx"
    _write(p, [["line_id", "amount", "description", "note"], ["L1", "10", "", ""], ["L2", "", "nuts", ""]])
    df = read_table(p)
    assert list(df.columns) == ["line_id", "amount", "description", "note"]
    assert len(df) == 2
