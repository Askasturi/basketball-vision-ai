"""Tests for CLI config loading."""

from __future__ import annotations

import json

import pytest

from basketball_vision.cli.config import CLIConfigError, load_config_file


def test_load_json_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(
        {"pipeline": {"save_intermediate": True}}))

    data = load_config_file(config_path)

    assert data == {"pipeline": {"save_intermediate": True}}


def test_missing_config_raises(tmp_path):
    config_path = tmp_path / "missing.json"

    with pytest.raises(CLIConfigError):
        load_config_file(config_path)


def test_unsupported_config_format_raises(tmp_path):
    config_path = tmp_path / "config.txt"
    config_path.write_text("bad")

    with pytest.raises(CLIConfigError):
        load_config_file(config_path)


def test_empty_json_object(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")

    assert load_config_file(config_path) == {}
