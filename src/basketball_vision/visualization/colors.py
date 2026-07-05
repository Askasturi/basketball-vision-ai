"""Color palette used by OpenCV visualization components."""

from __future__ import annotations

from enum import Enum

Color = tuple[int, int, int]


class ColorPalette(Enum):
    """Reusable BGR colors for drawing basketball annotations."""

    TEAM_A = (255, 0, 0)
    TEAM_B = (0, 0, 255)
    REFEREE = (0, 255, 255)
    UNKNOWN = (128, 128, 128)
    TEXT = (255, 255, 255)
    BACKGROUND = (0, 0, 0)

    @property
    def bgr(self) -> Color:
        """Return the OpenCV BGR color tuple."""
        return self.value
