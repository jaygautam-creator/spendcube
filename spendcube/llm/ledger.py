"""Quota ledger: requests/tokens per (key profile, entry id, Pacific quota day). Persisted in sqlite."""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def quota_day() -> str:
    return datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat()


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        self._db.execute("""CREATE TABLE IF NOT EXISTS u (
            profile TEXT, entry TEXT, qday TEXT, requests INT DEFAULT 0, ok INT DEFAULT 0,
            tok_in INT DEFAULT 0, tok_out INT DEFAULT 0, e429 INT DEFAULT 0, errors INT DEFAULT 0,
            PRIMARY KEY(profile, entry, qday))""")
        self._db.commit()

    def record(self, profile: str, entry: str, tok_in: int = 0, tok_out: int = 0, ok: bool = True,
               kind: str | None = None) -> None:
        d = quota_day()
        with self._lock:
            self._db.execute("INSERT OR IGNORE INTO u(profile,entry,qday) VALUES (?,?,?)", (profile, entry, d))
            self._db.execute(
                "UPDATE u SET requests=requests+1, ok=ok+?, tok_in=tok_in+?, tok_out=tok_out+?, "
                "e429=e429+?, errors=errors+? WHERE profile=? AND entry=? AND qday=?",
                (int(ok), tok_in, tok_out, int(kind in ("rate_limit", "quota")), int(not ok), profile, entry, d))
            self._db.commit()

    def requests_today(self, profile: str, entry: str) -> int:
        with self._lock:
            r = self._db.execute("SELECT requests FROM u WHERE profile=? AND entry=? AND qday=?",
                                 (profile, entry, quota_day())).fetchone()
        return r[0] if r else 0

    def summary(self, qday: str | None = None) -> list[tuple]:
        with self._lock:
            return self._db.execute(
                "SELECT profile, entry, requests, ok, tok_in, tok_out, e429, errors FROM u WHERE qday=? "
                "ORDER BY profile, entry", (qday or quota_day(),)).fetchall()

    def profile_requests(self, profile: str) -> int:
        with self._lock:
            r = self._db.execute("SELECT COALESCE(SUM(requests),0) FROM u WHERE profile=?", (profile,)).fetchone()
        return r[0]
