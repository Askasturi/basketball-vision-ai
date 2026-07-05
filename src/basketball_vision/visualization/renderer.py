"""Main visualization renderer."""

from __future__ import annotations

from typing import Any

import numpy as np

from basketball_vision.visualization.colors import ColorPalette
from basketball_vision.visualization.config import VisualizationConfig
from basketball_vision.visualization.drawing import (
    draw_box,
    draw_frame_index,
    draw_label,
)


class Renderer:
    """Render basketball analysis results onto video frames."""

    def __init__(self, config: VisualizationConfig | None = None) -> None:
        """Initialize the renderer."""
        self.config = config or VisualizationConfig()

    def render(
        self,
        frame: np.ndarray,
        pipeline_frame_result: object | None = None,
    ) -> np.ndarray:
        """Return an annotated copy of a frame."""
        annotated = frame.copy()

        if pipeline_frame_result is None:
            return annotated

        if self.config.show_frame_index:
            frame_index = self._get_attr(
                pipeline_frame_result, "frame_index", None)
            if frame_index is not None:
                draw_frame_index(
                    frame=annotated,
                    frame_index=int(frame_index),
                    config=self.config,
                    color=ColorPalette.TEXT.bgr,
                )

        objects = self._get_objects(pipeline_frame_result)

        for obj in objects:
            box = self._get_box(obj)
            if box is None:
                continue

            color = self._get_color(obj)

            if self.config.show_boxes:
                draw_box(
                    frame=annotated,
                    box=box,
                    color=color,
                    config=self.config,
                )

            label = self._build_label(obj)
            if label:
                draw_label(
                    frame=annotated,
                    text=label,
                    position=(box[0], max(15, box[1])),
                    text_color=ColorPalette.TEXT.bgr,
                    background_color=color,
                    config=self.config,
                )

        return annotated

    def _get_objects(self, pipeline_frame_result: object) -> list[Any]:
        objects = self._get_attr(pipeline_frame_result, "objects", None)
        if objects is not None:
            return list(objects)

        results = self._get_attr(pipeline_frame_result, "results", None)
        if results is not None:
            return list(results)

        detections = self._get_attr(pipeline_frame_result, "detections", None)
        if detections is not None:
            return list(detections)

        return []

    def _get_box(self, obj: object) -> tuple[int, int, int, int] | None:
        box = self._get_attr(obj, "box", None)
        if box is None:
            box = self._get_attr(obj, "bbox", None)
        if box is None:
            bounding_box = self._get_attr(obj, "bounding_box", None)
            if bounding_box is not None:
                box = bounding_box

        if box is None:
            return None

        if hasattr(box, "x1"):
            return (
                int(self._get_attr(box, "x1", 0)),
                int(self._get_attr(box, "y1", 0)),
                int(self._get_attr(box, "x2", 0)),
                int(self._get_attr(box, "y2", 0)),
            )

        if isinstance(box, (list, tuple)) and len(box) == 4:
            return tuple(int(value) for value in box)

        return None

    def _build_label(self, obj: object) -> str:
        parts: list[str] = []

        if self.config.show_ids:
            track_id = self._get_attr(obj, "track_id", None)
            if track_id is None:
                track_id = self._get_attr(obj, "id", None)
            if track_id is not None:
                parts.append(f"ID {track_id}")

        if self.config.show_team:
            team = self._get_attr(obj, "team", None)
            if team is not None:
                parts.append(str(team))

        if self.config.show_numbers:
            number = self._get_attr(obj, "number", None)
            if number is None:
                number = self._get_attr(obj, "jersey_number", None)
            if number is not None:
                parts.append(f"#{number}")

        if self.config.show_confidence:
            confidence = self._get_attr(obj, "confidence", None)
            if confidence is not None:
                parts.append(f"{float(confidence):.2f}")

        return " ".join(parts)

    def _get_color(self, obj: object) -> tuple[int, int, int]:
        team = str(self._get_attr(obj, "team", "")).lower()

        if "a" in team or "home" in team:
            return ColorPalette.TEAM_A.bgr
        if "b" in team or "away" in team:
            return ColorPalette.TEAM_B.bgr
        if "ref" in team:
            return ColorPalette.REFEREE.bgr

        return ColorPalette.UNKNOWN.bgr

    @staticmethod
    def _get_attr(obj: object, name: str, default: Any) -> Any:
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)
