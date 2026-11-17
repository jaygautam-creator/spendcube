"""Stage 6: classification cascade (see cascade.py). Degrades to a placeholder on failure."""
from __future__ import annotations

import pandas as pd

from .cascade import run_cascade


def classify_stage(ctx, df: pd.DataFrame) -> pd.DataFrame:
    return run_cascade(ctx, df)
