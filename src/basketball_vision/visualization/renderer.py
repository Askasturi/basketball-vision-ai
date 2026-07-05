"""Main visualization renderer."""

from __future__ import annotations

from typing import Any

import cv2
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
                pipeline_frame_result, "frame_index", None
            )
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

        self._draw_analytics_overlay(annotated, pipeline_frame_result)

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

    def _draw_analytics_overlay(
        self,
        frame: np.ndarray,
        pipeline_frame_result: object,
    ) -> None:
        analytics = self._get_attr(pipeline_frame_result, "analytics", None)
        if analytics is None:
            analytics = self._get_attr(
                pipeline_frame_result,
                "analytics_result",
                None,
            )

        if analytics is None:
            return

        game = self._get_attr(analytics, "game", None)
        if game is None:
            return

        frame_index = self._get_attr(
            pipeline_frame_result, "frame_index", None)
        players = self._get_attr(game, "number_of_players", 0)
        team_counts = self._get_attr(game, "team_counts", {})
        fps = self._extract_fps(game)

        overlay_lines = [
            f"Players: {players}",
            *self._team_overlay_lines(team_counts),
        ]

        if frame_index is not None:
            overlay_lines.append(f"Frame: {int(frame_index)}")

        overlay_lines.append(f"FPS: {fps:g}")

        self._draw_overlay_lines(frame=frame, lines=overlay_lines)

    @staticmethod
    def _team_overlay_lines(team_counts: object) -> list[str]:
        if not isinstance(team_counts, dict):
            return []

        lines: list[str] = []
        for team, count in sorted(team_counts.items()):
            team_name = str(team).replace("_", " ").title()
            lines.append(f"{team_name} Team: {count}")

        return lines

    @staticmethod
    def _extract_fps(game: object) -> float:
        total_frames = float(getattr(game, "total_frames", 0) or 0)
        video_seconds = float(getattr(game, "video_seconds", 0.0) or 0.0)

        if video_seconds <= 0:
            return 0.0

        return total_frames / video_seconds

    def _draw_overlay_lines(self, frame: np.ndarray, lines: list[str]) -> None:
        if not lines:
            return

        x = 10
        y = 45 if self.config.show_frame_index else 20
        line_height = 22
        padding = 6
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 1

        widths: list[int] = []
        heights: list[int] = []

        for line in lines:
            size, _ = cv2.getTextSize(line, font, scale, thickness)
            widths.append(size[0])
            heights.append(size[1])

        box_width = max(widths) + padding * 2
        box_height = len(lines) * line_height + padding

        cv2.rectangle(
            frame,
            (x - padding, y - 16),
            (x + box_width, y + box_height - 12),
            ColorPalette.UNKNOWN.bgr,
            -1,
        )

        for index, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (x, y + index * line_height),
                font,
                scale,
                ColorPalette.TEXT.bgr,
                thickness,
                cv2.LINE_AA,
            )

    @staticmethod
    def _get_attr(obj: object, name: str, default: Any) -> Any:
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)
