"""Tests for visualization colors."""

from __future__ import annotations

from basketball_vision.visualization import ColorPalette


def test_color_palette_values_are_bgr_tuples() -> None:
    for color in ColorPalette:
        assert isinstance(color.bgr, tuple)
        assert len(color.bgr) == 3
        assert all(isinstance(channel, int) for channel in color.bgr)
        assert all(0 <= channel <= 255 for channel in color.bgr)


def test_expected_palette_entries_exist() -> None:
    assert ColorPalette.TEAM_A.bgr == (255, 0, 0)
    assert ColorPalette.TEAM_B.bgr == (0, 0, 255)
    assert ColorPalette.REFEREE.bgr == (0, 255, 255)
    assert ColorPalette.UNKNOWN.bgr == (128, 128, 128)
    assert ColorPalette.TEXT.bgr == (255, 255, 255)
    assert ColorPalette.BACKGROUND.bgr == (0, 0, 0)
