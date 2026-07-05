"""Tests for OpenCV drawing helpers."""

from __future__ import annotations

import numpy as np

from basketball_vision.visualization.config import VisualizationConfig
from basketball_vision.visualization.drawing import (
    draw_box,
    draw_confidence,
    draw_frame_index,
    draw_label,
    draw_text,
)


def test_draw_box_changes_frame() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    config = VisualizationConfig()

    draw_box(frame, (10, 10, 50, 50), (255, 255, 255), config)

    assert np.any(frame != 0)


def test_draw_text_changes_frame() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    config = VisualizationConfig()

    draw_text(frame, "Test", (10, 50), (255, 255, 255), config)

    assert np.any(frame != 0)


def test_draw_label_changes_frame() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    config = VisualizationConfig()

    draw_label(frame, "ID 1", (10, 40), (255, 255, 255), (0, 0, 255), config)

    assert np.any(frame != 0)


def test_draw_frame_index_changes_frame() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    config = VisualizationConfig()

    draw_frame_index(frame, 12, config, (255, 255, 255))

    assert np.any(frame != 0)


def test_draw_confidence_changes_frame() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    config = VisualizationConfig()

    draw_confidence(frame, 0.95, (10, 50), (255, 255, 255), config)

    assert np.any(frame != 0)
