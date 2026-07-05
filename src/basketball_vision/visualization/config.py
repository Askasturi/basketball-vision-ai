"""Configuration for visualization rendering."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VisualizationConfig:
    """Options controlling how basketball results are drawn on frames."""

    show_boxes: bool = True
    show_ids: bool = True
    show_team: bool = True
    show_numbers: bool = True
    show_confidence: bool = False
    show_frame_index: bool = True

    line_thickness: int = 2
    font_scale: float = 0.6

    def __post_init__(self) -> None:
        """Validate visualization configuration values."""
        if self.line_thickness <= 0:
            msg = "line_thickness must be greater than 0"
            raise ValueError(msg)

        if self.font_scale <= 0:
            msg = "font_scale must be greater than 0"
            raise ValueError(msg)
