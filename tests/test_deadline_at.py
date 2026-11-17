from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from test_synthC_e2e import _draft
from spendcube.config.loader import load_config
from spendcube.pipeline import run_pipeline
from spendcube.taxonomy.tree import Taxonomy


def test_deadline_in_past_skips_llm_and_validates(tmp_path, monkeypatch):
    monkeypatch.setenv("SPENDCUBE_FORCE_MOCK", "1")
    cfg_path = _draft(tmp_path)
    cfg = load_config(cfg_path)
    deadline = datetime.now(ZoneInfo("Asia/Kolkata")) - timedelta(minutes=1)
    out, errors = run_pipeline(cfg, out_dir=tmp_path / "out", llm_mode="dev", cfg_path=str(cfg_path),
                               deadline_at=deadline)
    assert errors == []
    assert len(pd.read_csv(out)) == 3000
    assert run_pipeline.__module__ == "spendcube.pipeline"
    from spendcube.pipeline import LAST_RUN_DIR
    report = (LAST_RUN_DIR / "report.md").read_text()
    assert "deadline_hit: absolute" in report and "llm_units_done: 0" in report


def test_no_deadline_keeps_normal_llm_behavior(tmp_path, monkeypatch):
    monkeypatch.setenv("SPENDCUBE_FORCE_MOCK", "1")
    cfg_path = _draft(tmp_path)
    cfg = load_config(cfg_path)
    out, errors = run_pipeline(cfg, out_dir=tmp_path / "out", limit=40, cfg_path=str(cfg_path))
    assert errors == [] and out.exists()
    from spendcube.pipeline import LAST_RUN_DIR
    report = (LAST_RUN_DIR / "report.md").read_text()
    assert "deadline_hit: None" in report


def test_cli_deadline_is_asia_kolkata_aware():
    from spendcube.cli import main
    import spendcube.config.loader as loader
    import spendcube.pipeline as pipeline
    from types import SimpleNamespace
    captured = {}
    monkeypatch = __import__("pytest").MonkeyPatch()
    monkeypatch.setattr(loader, "load_config", lambda *a, **k: SimpleNamespace(classification={"llm": {}}))
    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return Path("fake.csv"), []
    monkeypatch.setattr(pipeline, "run_pipeline", fake_run)
    assert main(["run", "--config", "fake.yaml", "--deadline-at", "2026-11-19T11:20"]) == 0
    assert captured["deadline_at"].tzinfo == ZoneInfo("Asia/Kolkata")
    monkeypatch.undo()
