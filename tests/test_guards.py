"""Anti-overfitting and data-safety guards."""
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "Cantill_DataML_Task_Kit"
sys.path.insert(0, str(ROOT / "scripts"))


def _kit_strings() -> set[str]:
    out: set[str] = set()
    for f in list(KIT.glob("company_*/taxonomy_*.csv")):
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        for c in df.columns:
            out.update(v.strip() for v in df[c] if len(v.strip()) >= 6)
    return out


def _string_literals(py: Path) -> list[str]:
    """String literals in code (docstrings and comments excluded)."""
    import ast
    tree = ast.parse(py.read_text(encoding="utf-8"))
    doc_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant):
                doc_nodes.add(id(node.body[0].value))
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in doc_nodes]


@pytest.mark.skipif(not KIT.exists(), reason="kit not present (CI)")
def test_no_kit_taxonomy_strings_in_code():
    """No kit category name may appear as a string literal in spendcube/ (comments/docstrings may use plain words)."""
    names = {s.casefold() for s in _kit_strings()}
    hits = []
    for py in (ROOT / "spendcube").rglob("*.py"):
        for lit in _string_literals(py):
            l = lit.casefold()
            for n in names:
                if l.strip() == n or (" " in n and n in l):
                    hits.append((py.name, n))
    assert not hits, f"kit taxonomy strings found in code: {hits}"


def test_data_guard_blocks_data_files():
    import data_guard
    bad = data_guard.violations(["spendcube/x.py", "data/a.csv", "outputs/jaygautam__A.csv",
                                 "tests/fixtures/s.csv", "labels_A.csv", ".env",
                                 "Cantill_DataML_Task_Kit/x.txt"])
    flagged = [b.split()[0] for b in bad]
    assert "data/a.csv" in flagged and "labels_A.csv" in flagged and ".env" in flagged
    assert "Cantill_DataML_Task_Kit/x.txt" in flagged
    assert "outputs/jaygautam__A.csv" not in flagged and "tests/fixtures/s.csv" not in flagged
    assert "spendcube/x.py" not in flagged


def test_mock_safety_rejects_blind_variables(monkeypatch, capsys):
    from spendcube.llm.safety import reject_mock_in_blind
    monkeypatch.setenv("SPENDCUBE_MOCK_LATENCY", "")
    assert reject_mock_in_blind("blind") is True
    assert "SPENDCUBE_MOCK_LATENCY" in capsys.readouterr().err
    monkeypatch.delenv("SPENDCUBE_MOCK_LATENCY")
    assert reject_mock_in_blind("dev") is False


def test_mock_detection_includes_configured_mock():
    from spendcube.llm.safety import mock_active
    assert mock_active({"chain": [{"provider": "mock"}]})
    assert not mock_active({"chain": [{"provider": "gemini"}]})
