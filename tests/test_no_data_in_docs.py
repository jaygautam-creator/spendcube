"""No task data (supplier names, taxonomy names, descriptions, vendor ids) in any tracked text file.

outputs/ and submission/ legitimately contain keys; tests/fixtures are synthetic and checked separately by the
data guard (file types) rather than here.
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import leak_scan  # noqa: E402

pytestmark = pytest.mark.skipif(not leak_scan.KIT.exists(), reason="kit not present (CI)")


def _tracked_text_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    files = []
    for f in out:
        if not f or f.startswith(leak_scan.SKIP_PREFIX):
            continue
        p = ROOT / f
        if p.suffix.lower() in leak_scan.TEXT_EXT or p.name in (".gitignore", ".env.example"):
            if p.is_file():
                files.append(p)
    return files


def test_no_sensitive_strings_in_tracked_files():
    strings = leak_scan.light_strings() | leak_scan.heavy_strings()
    hits = leak_scan.scan_files(_tracked_text_files(), strings)
    assert not hits, "task data strings in tracked files:\n" + "\n".join(f"{f}: {s}" for f, s in hits[:40])
