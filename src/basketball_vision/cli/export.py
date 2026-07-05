"""CLI result export helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def make_json_safe(value: Any) -> Any:
    """Convert common Python objects into JSON-safe values."""
    if is_dataclass(value):
        return make_json_safe(asdict(value))

    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}

    if isinstance(value, list | tuple):
        return [make_json_safe(item) for item in value]

    if hasattr(value, "tolist"):
        return value.tolist()

    if hasattr(value, "__dict__"):
        return make_json_safe(vars(value))

    return value


def export_json(results: Any, output_path: str | Path) -> Path:
    """Export pipeline results to a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(make_json_safe(results), file, indent=2)

    return path
