"""D-43: the profiler suggests NON_PROCUREMENT rules for template-like lines only (synthetic data)."""
import pandas as pd

from spendcube.profile.profiler import suggest_exclusions


def _df():
    rows = []
    for i in range(30):
        rows.append({"gl": f"IC{i % 7:02d}", "desc": f"Withholding Tax Posting - 2024-{i % 12 + 1:02d}", "sup": "Acme Tools", "amt": 100.0})
    for i in range(12):
        rows.append({"gl": "Consulting", "desc": f"VAT filing support service {i}", "sup": "Taxwise Advisors", "amt": 500.0})
    rows.append({"gl": "Consulting", "desc": "Zenmark Tax Advisor - Q1", "sup": "Zenmark", "amt": 900.0})
    rows += [{"gl": "Tools", "desc": "Hex bolt M8", "sup": "Acme Tools", "amt": 5.0}] * 20
    return pd.DataFrame(rows)


def test_suggests_template_lines_and_skips_services():
    df = _df()
    roles = {"gl_description": "gl", "description": "desc", "supplier_name": "sup"}
    s = suggest_exclusions(df, roles, df["amt"])
    by = {(x["field"], x["category"]): x for x in s}
    assert ("gl_desc", "intercompany") in by and by[("gl_desc", "intercompany")]["rows"] == 30
    assert ("desc_raw", "tax") in by and by[("desc_raw", "tax")]["rows"] == 30
    assert by[("desc_raw", "tax")]["spend"] == 3000.0
    # service-like lines (consultant/support service) and ordinary goods are never suggested
    for x in s:
        assert "consult" not in x["regex"].lower() and "service" not in x["regex"].lower()
    assert all(x["rows"] >= 1 for x in s)
