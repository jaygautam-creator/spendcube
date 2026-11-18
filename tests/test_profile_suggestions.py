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


def test_suggests_partner_royalty_and_financing_templates_but_vetoes_services():
    phrases = ["Partner Billing", "Joint Interest Billing", "JIB Billing", "Cash Call", "Joint Venture",
               "JV Billing", "Royalty", "Finance Charge", "Lease Interest", "Revenue Share"]
    rows = [{"gl": "", "desc": f"{p} - 2025-Q1", "sup": "", "amt": 100.0} for p in phrases for _ in range(3)]
    rows += [{"gl": "", "desc": "Royalty advisory services", "sup": "", "amt": 500.0} for _ in range(5)]
    df = pd.DataFrame(rows)
    suggestions = suggest_exclusions(df, {"description": "desc"}, df["amt"])
    matched = {x["phrase"] for x in suggestions}
    assert {p.lower() for p in phrases} <= matched
    assert all("service" not in x["regex"].lower() for x in suggestions)


def test_jib_equipment_is_not_suggested_as_intercompany():
    rows = ([{"desc": "JIB billing - Q3", "amt": 10.0}] * 3 +
            [{"desc": "Jib crane hire - Q3", "amt": 20.0}] * 3 +
            [{"desc": "Jib crane", "amt": 30.0}] * 3)
    df = pd.DataFrame(rows)
    suggestions = suggest_exclusions(df, {"description": "desc"}, df["amt"])
    matched = {x["phrase"] for x in suggestions}
    assert "jib billing" in matched
    assert "jib crane" not in matched and "jib crane hire" not in matched
    assert not any(__import__("re").search(x["regex"], "Jib crane hire") for x in suggestions)
