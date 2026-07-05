"""Tests for CLI export helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass

from basketball_vision.cli.export import export_json, make_json_safe


@dataclass
class DummyResult:
    frame_index: int
    value: str


def test_make_json_safe_with_dataclass():
    result = DummyResult(frame_index=1, value="ok")

    assert make_json_safe(result) == {"frame_index": 1, "value": "ok"}


def test_export_json(tmp_path):
    output_path = tmp_path / "results.json"

    export_json({"frames": [{"frame_index": 1}]}, output_path)

    assert output_path.exists()
    assert json.loads(output_path.read_text()) == {
        "frames": [{"frame_index": 1}]
    }
