"""Reusable OpenCV drawing helpers."""

from __future__ import annotations

import cv2
import numpy as np

from basketball_vision.visualization.colors import Color
from basketball_vision.visualization.config import VisualizationConfig

Point = tuple[int, int]
Box = tuple[int, int, int, int]


def draw_box(
    frame: np.ndarray,
    box: Box,
    color: Color,
    config: VisualizationConfig,
) -> None:
    """Draw a rectangle on a frame."""
    x1, y1, x2, y2 = box
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        thickness=config.line_thickness,
    )


def draw_text(
    frame: np.ndarray,
    text: str,
    position: Point,
    color: Color,
    config: VisualizationConfig,
) -> None:
    """Draw text on a frame."""
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        config.font_scale,
        color,
        config.line_thickness,
        cv2.LINE_AA,
    )


def draw_label(
    frame: np.ndarray,
    text: str,
    position: Point,
    text_color: Color,
    background_color: Color,
    config: VisualizationConfig,
) -> None:
    """Draw text with a filled background label."""
    x, y = position
    baseline = 0
    text_size, baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        config.font_scale,
        config.line_thickness,
    )
    text_width, text_height = text_size

    top_left = (x, max(0, y - text_height - baseline - 6))
    bottom_right = (x + text_width + 6, y)

    cv2.rectangle(
        frame,
        top_left,
        bottom_right,
        background_color,
        thickness=-1,
    )

    draw_text(
        frame=frame,
        text=text,
        position=(x + 3, y - baseline - 3),
        color=text_color,
        config=config,
    )


def draw_frame_index(
    frame: np.ndarray,
    frame_index: int,
    config: VisualizationConfig,
    color: Color,
) -> None:
    """Draw the current frame index."""
    draw_label(
        frame=frame,
        text=f"Frame: {frame_index}",
        position=(10, 30),
        text_color=color,
        background_color=(0, 0, 0),
        config=config,
    )


def draw_confidence(
    frame: np.ndarray,
    confidence: float,
    position: Point,
    color: Color,
    config: VisualizationConfig,
) -> None:
    """Draw a confidence value."""
    draw_text(
        frame=frame,
        text=f"{confidence:.2f}",
        position=position,
        color=color,
        config=config,
    )
