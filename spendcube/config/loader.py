"""Config loading: YAML/JSON -> env/placeholder expansion -> pydantic."""
from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

import yaml

from .schema import Config

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(Exception):
    pass


def _expand_env(s: str) -> str:
    def rep(m: re.Match) -> str:
        return os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else "")
    return _ENV_RE.sub(rep, s)


def _walk(obj: Any, fn) -> Any:
    if isinstance(obj, str):
        return fn(obj)
    if isinstance(obj, list):
        return [_walk(x, fn) for x in obj]
    if isinstance(obj, dict):
        return {k: _walk(v, fn) for k, v in obj.items()}
    return obj


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(
    path: str | Path,
    input_dir: Optional[str] = None,
    kit_dir: Optional[str] = None,
    suffix: Optional[str] = None,
    overrides: Optional[dict[str, Any]] = None,
) -> Config:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config not found: {p}")
    text = p.read_text(encoding="utf-8")
    raw = json.loads(text) if p.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")
    raw = _walk(raw, _expand_env)
    for inc in raw.pop("include", []) or []:
        ip = (p.parent / inc) if not Path(inc).is_absolute() else Path(inc)
        base = yaml.safe_load(ip.read_text(encoding="utf-8")) or {}
        raw = _deep_merge(_walk(base, _expand_env), raw)      # the company config wins
    if input_dir:
        raw["input_dir"] = input_dir
    if kit_dir:
        raw["kit_dir"] = kit_dir
    if suffix:
        raw["output_suffix"] = suffix
    if overrides:
        raw.update(overrides)
    in_dir = str(raw.get("input_dir", "."))
    k_dir = str(raw.get("kit_dir") or in_dir)

    def subst(s: str) -> str:
        return s.replace("{input_dir}", in_dir).replace("{kit_dir}", k_dir)
    raw = _walk(raw, subst)
    raw["kit_dir"] = k_dir
    try:
        return Config.model_validate(raw)
    except Exception as e:  # pydantic ValidationError -> readable
        raise ConfigError(f"invalid config {p}: {e}") from e


def resolve_files(pattern: str | list[str], required: bool = True) -> list[Path]:
    """Expand glob(s); sorted, de-duplicated. Raises ConfigError if required and empty."""
    pats = [pattern] if isinstance(pattern, str) else list(pattern)
    out: list[Path] = []
    for pat in pats:
        hits = sorted(glob.glob(os.path.expanduser(pat)))
        out.extend(Path(h) for h in hits if Path(h).is_file())
    seen, uniq = set(), []
    for h in out:
        if h not in seen:
            seen.add(h)
            uniq.append(h)
    if required and not uniq:
        raise ConfigError(f"no file matches: {pats}")
    return uniq
