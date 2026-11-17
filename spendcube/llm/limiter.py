"""Per-entry RPM/TPM sliding-window limiter (thread-safe). RPD is enforced from the ledger."""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional


class Limiter:
    def __init__(self, rpm: Optional[int] = None, tpm: Optional[int] = None):
        self.rpm, self.tpm = rpm, tpm
        self._q: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def acquire(self, est_tokens: int = 1000, max_wait: float = 90.0) -> bool:
        deadline = time.time() + max_wait
        while True:
            with self._lock:
                now = time.time()
                while self._q and now - self._q[0][0] >= 60:
                    self._q.popleft()
                tok = sum(t for _, t in self._q)
                rpm_ok = self.rpm is None or len(self._q) < self.rpm
                tpm_ok = self.tpm is None or tok + est_tokens <= self.tpm or not self._q
                if rpm_ok and tpm_ok:
                    self._q.append((now, est_tokens))
                    return True
                wait = (60 - (now - self._q[0][0])) if self._q else 1.0
            if time.time() + wait > deadline:
                return False
            time.sleep(min(max(wait, 0.2), 5.0))

    def penalise(self, seconds: float) -> None:
        """After a 429, block the whole window for `seconds`."""
        with self._lock:
            now = time.time()
            for _ in range(self.rpm or 1):
                self._q.append((now + seconds - 60, 0))

    def available_in(self, est_tokens: int = 1000) -> float:
        """Estimated wait until a request fits, without reserving quota."""
        with self._lock:
            now = time.time()
            active = [(t, n) for t, n in self._q if now - t < 60]
            wait = 0.0
            if self.rpm and len(active) >= self.rpm:
                wait = max(wait, active[0][0] + 60 - now)
            if self.tpm and sum(n for _, n in active) + est_tokens > self.tpm and active:
                wait = max(wait, active[0][0] + 60 - now)
            return max(0.0, wait)
