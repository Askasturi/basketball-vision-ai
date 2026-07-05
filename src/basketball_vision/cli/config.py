"""CLI configuration loading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class CLIConfigError(ValueError):
    """Raised when CLI config loading fails."""


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Load a JSON or YAML config file.

    YAML support requires PyYAML. JSON works without extra dependencies.
    """
    config_path = Path(path)

    if not config_path.exists():
        raise CLIConfigError(f"Config file does not exist: {config_path}")

    suffix = config_path.suffix.lower()

    if suffix == ".json":
        with config_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise CLIConfigError(
                "YAML config files require PyYAML. Install it with: pip install pyyaml"
            ) from exc

        with config_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

    else:
        raise CLIConfigError(
            f"Unsupported config format: {suffix}. Use .json, .yaml, or .yml."
        )

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise CLIConfigError(
            "Config file must contain a top-level object/dictionary.")

    return data
