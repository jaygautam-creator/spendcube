"""RunContext: everything stages share (config, taxonomy, run dir, timers, notes for the report)."""
from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .config.schema import Config


@dataclass
class RunContext:
    cfg: Config
    run_dir: Path
    taxonomy: Any = None            # Taxonomy
    no_llm: bool = False
    limit: Optional[int] = None
    out_dir: Path = Path("outputs")
    llm_mode: str = "dev"           # dev | blind (selects Gemini key profile)
    deadline_at: Optional[datetime] = None
    cfg_path: Optional[str] = None
    timings: dict[str, float] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)      # for report.md
    warnings: list[str] = field(default_factory=list)
    started: float = field(default_factory=time.time)
    state: dict[str, Any] = field(default_factory=dict)       # inter-stage objects (labels, models...)

    @property
    def log(self) -> logging.Logger:
        return logging.getLogger("spendcube")

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        self.log.warning(msg)

    def note(self, section: str, key: str, value: Any) -> None:
        self.notes.setdefault(section, {})[key] = value

    @contextmanager
    def timer(self, name: str):
        t0 = time.time()
        try:
            yield
        finally:
            self.timings[name] = round(self.timings.get(name, 0.0) + time.time() - t0, 2)

    @property
    def elapsed_min(self) -> float:
        return (time.time() - self.started) / 60.0


def make_run_dir(company: str, base: Path = Path("runs")) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = base / f"{ts}_{company}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def setup_logging(run_dir: Path, verbose: bool = False) -> None:
    root = logging.getLogger("spendcube")
    root.handlers.clear()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    root.propagate = False
