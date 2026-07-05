"""Tests for visualization configuration."""

from __future__ import annotations

import pytest

from basketball_vision.visualization import VisualizationConfig


def test_default_config() -> None:
    config = VisualizationConfig()

    assert config.show_boxes is True
    assert config.show_ids is True
    assert config.show_team is True
    assert config.show_numbers is True
    assert config.show_confidence is False
    assert config.show_frame_index is True
    assert config.line_thickness == 2
    assert config.font_scale == 0.6


def test_custom_config() -> None:
    config = VisualizationConfig(
        show_boxes=False,
        show_ids=False,
        show_team=False,
        show_numbers=False,
        show_confidence=True,
        show_frame_index=False,
        line_thickness=4,
        font_scale=1.0,
    )

    assert config.show_boxes is False
    assert config.show_ids is False
    assert config.show_team is False
    assert config.show_numbers is False
    assert config.show_confidence is True
    assert config.show_frame_index is False
    assert config.line_thickness == 4
    assert config.font_scale == 1.0


def test_invalid_line_thickness_raises() -> None:
    with pytest.raises(ValueError, match="line_thickness"):
        VisualizationConfig(line_thickness=0)


def test_invalid_font_scale_raises() -> None:
    with pytest.raises(ValueError, match="font_scale"):
        VisualizationConfig(font_scale=0)
