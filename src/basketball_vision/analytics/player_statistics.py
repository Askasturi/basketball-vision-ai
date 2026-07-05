"""Player statistics calculation."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Any

from basketball_vision.analytics.statistics_result import PlayerStatistics


@dataclass(slots=True)
class PlayerStatisticsAccumulator:
    """Mutable accumulator for one tracked player's statistics."""

    track_id: int
    fps: float
    team: str | None = None
    jersey_number: str | None = None
    frames_seen: int = 0
    first_frame: int | None = None
    last_frame: int | None = None
    confidence_total: float = 0.0
    area_total: float = 0.0
    distance_travelled_pixels: float = 0.0
    maximum_speed: float = 0.0
    _last_center: tuple[float, float] | None = field(default=None, init=False)

    def update(
        self,
        *,
        frame_index: int,
        track: Any,
        team: str | None = None,
        jersey_number: str | None = None,
    ) -> None:
        """Update player statistics from one frame."""
        self.frames_seen += 1

        if self.first_frame is None:
            self.first_frame = frame_index
        self.last_frame = frame_index

        if team is not None:
            self.team = str(team)

        if jersey_number is not None:
            self.jersey_number = str(jersey_number)

        detection = getattr(track, "detection", track)
        confidence = float(getattr(detection, "confidence", 0.0))
        self.confidence_total += confidence

        box = self._extract_box(detection)
        if box is None:
            return

        x1, y1, x2, y2 = box
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        self.area_total += width * height

        center = (x1 + width / 2.0, y1 + height / 2.0)
        if self._last_center is not None:
            distance = hypot(
                center[0] - self._last_center[0],
                center[1] - self._last_center[1],
            )
            self.distance_travelled_pixels += distance
            speed = distance * self.fps
            self.maximum_speed = max(self.maximum_speed, speed)

        self._last_center = center

    def to_result(self) -> PlayerStatistics:
        """Convert accumulated statistics to an immutable result."""
        first_frame = self.first_frame if self.first_frame is not None else 0
        last_frame = self.last_frame if self.last_frame is not None else first_frame
        frames_seen = max(self.frames_seen, 1)

        return PlayerStatistics(
            track_id=self.track_id,
            team=self.team,
            jersey_number=self.jersey_number,
            frames_seen=self.frames_seen,
            first_frame=first_frame,
            last_frame=last_frame,
            time_visible=self.frames_seen / self.fps,
            average_detection_confidence=self.confidence_total / frames_seen,
            average_bounding_box_area=self.area_total / frames_seen,
            distance_travelled_pixels=self.distance_travelled_pixels,
            average_speed=self.distance_travelled_pixels
            / max(self.time_visible_seconds, 1 / self.fps),
            maximum_speed=self.maximum_speed,
        )

    @property
    def time_visible_seconds(self) -> float:
        """Return visible time in seconds."""
        return self.frames_seen / self.fps if self.frames_seen else 0.0

    @staticmethod
    def _extract_box(obj: Any) -> tuple[float, float, float, float] | None:
        box = getattr(obj, "box", None)
        if box is None:
            box = getattr(obj, "bbox", None)
        if box is None:
            box = getattr(obj, "bounding_box", None)

        if box is None:
            return None

        if hasattr(box, "x1"):
            return (
                float(getattr(box, "x1")),
                float(getattr(box, "y1")),
                float(getattr(box, "x2")),
                float(getattr(box, "y2")),
            )

        if isinstance(box, (list, tuple)) and len(box) == 4:
            return tuple(float(value) for value in box)

        return None
