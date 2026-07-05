"""Simple color-based player team classifier."""

from typing import Any

import numpy as np

from basketball_vision.classification.base_classifier import BasePlayerClassifier
from basketball_vision.classification.config import ColorTeamClassifierConfig
from basketball_vision.classification.player_team import (
    ClassificationResult,
    PlayerTeamAssignment,
)
from basketball_vision.classification.types import TeamLabel


class ColorTeamClassifier(BasePlayerClassifier):
    """Classify players by comparing crop average color to team colors."""

    def __init__(self, config: ColorTeamClassifierConfig | None = None) -> None:
        super().__init__(config or ColorTeamClassifierConfig())
        self.config: ColorTeamClassifierConfig

    def classify(self, frame: np.ndarray, tracking_result: Any) -> ClassificationResult:
        """Classify each active tracked player in a frame."""
        self._validate_frame(frame)

        assignments: list[PlayerTeamAssignment] = []

        for track in self._get_tracks(tracking_result):
            if not self._should_classify_track(track):
                continue

            crop = self._crop_track(frame, track)
            if crop is None:
                assignment = PlayerTeamAssignment(
                    track_id=self._get_track_id(track),
                    team=TeamLabel.UNKNOWN,
                    confidence=0.0,
                    track=track,
                )
                assignments.append(assignment)
                continue

            average_color = self._average_rgb(crop)
            team, confidence = self._classify_color(average_color)

            assignments.append(
                PlayerTeamAssignment(
                    track_id=self._get_track_id(track),
                    team=team,
                    confidence=confidence,
                    track=track,
                )
            )

        return ClassificationResult(
            frame_index=self._get_frame_index(tracking_result),
            timestamp=self._get_timestamp(tracking_result),
            assignments=tuple(assignments),
        )

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            msg = "frame must be a numpy array"
            raise TypeError(msg)

        if frame.ndim != 3:
            msg = "frame must have shape (height, width, channels)"
            raise ValueError(msg)

        if frame.shape[2] < 3:
            msg = "frame must have at least 3 color channels"
            raise ValueError(msg)

    @staticmethod
    def _get_tracks(tracking_result: Any) -> tuple[Any, ...]:
        tracks = getattr(tracking_result, "tracks", ())
        return tuple(tracks)

    @staticmethod
    def _get_frame_index(tracking_result: Any) -> int:
        return int(getattr(tracking_result, "frame_index", 0))

    @staticmethod
    def _get_timestamp(tracking_result: Any) -> float | None:
        return getattr(tracking_result, "timestamp", None)

    @staticmethod
    def _get_track_id(track: Any) -> int:
        return int(getattr(track, "track_id"))

    def _should_classify_track(self, track: Any) -> bool:
        state = getattr(track, "state", None)
        state_name = getattr(state, "name", str(state)).upper()

        if state_name == "LOST" and not self.config.include_lost_tracks:
            return False

        if state_name == "REMOVED" and not self.config.include_removed_tracks:
            return False

        return True

    def _crop_track(self, frame: np.ndarray, track: Any) -> np.ndarray | None:
        box = self._get_track_box(track)
        if box is None:
            return None

        height, width = frame.shape[:2]
        padding = self.config.crop_padding

        x1 = max(0, int(round(box[0])) - padding)
        y1 = max(0, int(round(box[1])) - padding)
        x2 = min(width, int(round(box[2])) + padding)
        y2 = min(height, int(round(box[3])) + padding)

        if x2 <= x1 or y2 <= y1:
            return None

        area = (x2 - x1) * (y2 - y1)
        if area < self.config.min_crop_area:
            return None

        return frame[y1:y2, x1:x2, :3]

    @staticmethod
    def _get_track_box(track: Any) -> tuple[float, float, float, float] | None:
        """Get xyxy box from Track.

        Compatibility note:
        Detection uses `bounding_box`, not `bbox`.
        """

        direct_box = getattr(track, "bounding_box", None)
        if direct_box is not None:
            return ColorTeamClassifier._box_to_xyxy(direct_box)

        detection = getattr(track, "detection", None)
        if detection is not None:
            detection_box = getattr(detection, "bounding_box", None)
            if detection_box is not None:
                return ColorTeamClassifier._box_to_xyxy(detection_box)

        return None

    @staticmethod
    def _box_to_xyxy(box: Any) -> tuple[float, float, float, float]:
        if all(hasattr(box, attr) for attr in ("x1", "y1", "x2", "y2")):
            return (
                float(box.x1),
                float(box.y1),
                float(box.x2),
                float(box.y2),
            )

        if all(hasattr(box, attr) for attr in ("left", "top", "right", "bottom")):
            return (
                float(box.left),
                float(box.top),
                float(box.right),
                float(box.bottom),
            )

        if isinstance(box, (tuple, list)) and len(box) == 4:
            # type: ignore[return-value]
            return tuple(float(value) for value in box)

        msg = "bounding_box must expose x1/y1/x2/y2 or be a 4-value tuple/list"
        raise TypeError(msg)

    @staticmethod
    def _average_rgb(crop: np.ndarray) -> tuple[float, float, float]:
        average = crop.reshape(-1, crop.shape[-1]).mean(axis=0)
        return float(average[0]), float(average[1]), float(average[2])

    def _classify_color(
        self,
        color: tuple[float, float, float],
    ) -> tuple[TeamLabel, float]:
        team_a_distance = self._color_distance(color, self.config.team_a_color)
        team_b_distance = self._color_distance(color, self.config.team_b_color)

        best_distance = min(team_a_distance, team_b_distance)

        if best_distance > self.config.unknown_distance_threshold:
            return TeamLabel.UNKNOWN, 0.0

        team = (
            TeamLabel.TEAM_A
            if team_a_distance <= team_b_distance
            else TeamLabel.TEAM_B
        )

        max_distance = 441.6729559300637
        confidence = 1.0 - min(best_distance / max_distance, 1.0)

        if confidence < self.config.min_confidence:
            return TeamLabel.UNKNOWN, confidence

        return team, confidence

    @staticmethod
    def _color_distance(
        color: tuple[float, float, float],
        target: tuple[int, int, int],
    ) -> float:
        return float(
            np.linalg.norm(
                np.array(color, dtype=np.float32)
                - np.array(target, dtype=np.float32)
            )
        )
