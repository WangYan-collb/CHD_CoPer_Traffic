from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal bootstrap envs
    yaml = None


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        if yaml is not None:
            return yaml.safe_load(handle)
        return _load_simple_yaml(handle.read())


def apply_smoke_overrides(config: dict[str, Any], smoke: bool) -> dict[str, Any]:
    copied = dict(config)
    copied["smoke"] = bool(smoke or copied.get("smoke", False))
    if copied["smoke"]:
        copied.setdefault("training", {})["episodes"] = 1
        copied.setdefault("meta", {})["episodes"] = 1
    return copied


def _coerce_scalar(value: str) -> Any:
    value = value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_simple_yaml(text: str) -> dict[str, Any]:
    """Small fallback parser for this project's simple YAML configs."""

    root: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    current_key: str | None = None
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        if stripped.startswith("- ") and current_key is not None:
            item = stripped[2:].strip()
            if not isinstance(root[current_key], list):
                root[current_key] = []
            root[current_key].append(_coerce_scalar(item))
            continue
        key, _, value = stripped.partition(":")
        if indent == 0:
            if value.strip():
                root[key] = _coerce_scalar(value)
                current = None
                current_key = key
            else:
                root[key] = []
                current = None
                current_key = key
        elif current is not None:
            current[key] = _coerce_scalar(value)
        else:
            if not isinstance(root.get(current_key), dict):
                root[current_key or key] = {}
            current = root[current_key or key]
            current[key] = _coerce_scalar(value)
    return root
