"""Minimal YAML config + dotted CLI overrides (e.g. ``--train.learning_rate 1e-6``)."""

import json
from typing import Any

import yaml


def load_yaml_config(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _coerce(value: str) -> Any:
    """Best-effort scalar coercion from a CLI string using JSON, then raw string."""

    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value


def apply_dotted_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply ``["--a.b", "1", "--c", "x"]`` style overrides in place and return config."""

    index = 0
    while index < len(overrides):
        token = overrides[index]
        if not token.startswith("--"):
            raise ValueError(f"Expected an option starting with '--', got {token!r}.")
        key = token[2:]
        if "=" in key:
            key, raw_value = key.split("=", 1)
            index += 1
        else:
            if index + 1 >= len(overrides):
                raise ValueError(f"Missing value for override {token!r}.")
            raw_value = overrides[index + 1]
            index += 2
        node = config
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(raw_value)
    return config
