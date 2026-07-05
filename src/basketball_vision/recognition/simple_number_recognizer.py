
"""Simple deterministic placeholder jersey number recognizer."""

from __future__ import annotations

from typing import Any

import numpy as np

from basketball_vision.recognition.base_recognizer import BaseNumberRecognizer
from basketball_vision.recognition.config import RecognitionConfig
from basketball_vision.recognition.exceptions import RecognitionInputError
from basketball_vision.recognition.number import (
    PlayerNumberRecognition,
    RecognitionResult,
)
from basketball_vision.recognition.types import NumberRecognitionStatus


class SimpleNumberRecognizer(BaseNumberRecognizer):
    """Deterministic recognizer for testing and pipeline integration.

    This recognizer does not perform OCR. It returns configured jersey numbers
    by track ID and marks all other active tracks as unknown.
    """

    def __init__(self, config: RecognitionConfig | None = None) -> None:
        super().__init__(config=config)

    def recognize(
        self,
        frame: np.ndarray,
        tracking_result: Any,
        classification_result: Any | None = None,
    ) -> RecognitionResult:
        """Recognize player numbers for tracks in a frame."""
        self._validate_frame(frame)

        recognitions: list[PlayerNumberRecognition] = []

        for track in self._get_tracks(tracking_result):
            if not self._should_include_track(track):
                continue

            track_id = self._get_track_id(track)
            self._crop_track(frame, track)

            team_assignment = self._find_team_assignment(
                track_id=track_id,
                classification_result=classification_result,
            )

            number = self.config.track_id_to_number.get(track_id)

            if number is None:
                recognitions.append(
                    PlayerNumberRecognition(
                        track_id=track_id,
                        number=None,
                        confidence=0.0,
                        status=NumberRecognitionStatus.UNKNOWN,
                        track=track,
                        team_assignment=team_assignment,
                    )
                )
                continue

            recognitions.append(
                PlayerNumberRecognition(
                    track_id=track_id,
                    number=number,
                    confidence=1.0,
                    status=NumberRecognitionStatus.RECOGNIZED,
                    track=track,
                    team_assignment=team_assignment,
                )
            )

        return RecognitionResult(
            frame_index=self._get_frame_index(tracking_result),
            timestamp=self._get_timestamp(tracking_result),
            recognitions=tuple(recognitions),
        )

    def _validate_frame(self, frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            msg = "frame must be a numpy array"
            raise RecognitionInputError(msg)

        if frame.ndim not in {2, 3}:
            msg = "frame must be a 2D grayscale or 3D color image"
            raise RecognitionInputError(msg)

        if frame.size == 0:
            msg = "frame must not be empty"
            raise RecognitionInputError(msg)

    def _get_tracks(self, tracking_result: Any) -> tuple[Any, ...]:
        tracks = getattr(tracking_result, "tracks", None)

        if tracks is None:
            msg = "tracking_result must have a tracks attribute"
            raise RecognitionInputError(msg)

        return tuple(tracks)

    def _get_track_id(self, track: Any) -> int:
        track_id = getattr(track, "track_id", None)

        if not isinstance(track_id, int):
            msg = "track must have an integer track_id"
            raise RecognitionInputError(msg)

        return track_id

    def _get_frame_index(self, tracking_result: Any) -> int:
        frame_index = getattr(tracking_result, "frame_index", 0)

        if frame_index is None:
            return 0

        return int(frame_index)

    def _get_timestamp(self, tracking_result: Any) -> float | None:
        timestamp = getattr(tracking_result, "timestamp", None)

        if timestamp is None:
            return None

        return float(timestamp)

    def _should_include_track(self, track: Any) -> bool:
        status = getattr(track, "status", None)

        if status is None:
            return True

        status_value = getattr(status, "value", str(status)).lower()

        if "lost" in status_value:
            return self.config.include_lost_tracks

        if "removed" in status_value:
            return self.config.include_removed_tracks

        return True

    def _crop_track(self, frame: np.ndarray, track: Any) -> np.ndarray:
        bounding_box = self._get_bounding_box(track)

        x1 = max(0, int(round(getattr(bounding_box, "x1"))))
        y1 = max(0, int(round(getattr(bounding_box, "y1"))))
        x2 = min(frame.shape[1], int(round(getattr(bounding_box, "x2"))))
        y2 = min(frame.shape[0], int(round(getattr(bounding_box, "y2"))))

        if x2 <= x1 or y2 <= y1:
            return frame[0:0, 0:0]

        return frame[y1:y2, x1:x2]

    def _get_bounding_box(self, track: Any) -> Any:
        bounding_box = getattr(track, "bounding_box", None)

        if bounding_box is not None:
            return bounding_box

        detection = getattr(track, "detection", None)
        if detection is not None:
            bounding_box = getattr(detection, "bounding_box", None)
            if bounding_box is not None:
                return bounding_box

        msg = "track must expose bounding_box or detection.bounding_box"
        raise RecognitionInputError(msg)

    def _find_team_assignment(
        self,
        track_id: int,
        classification_result: Any | None,
    ) -> Any | None:
        if classification_result is None:
            return None

        assignments = getattr(classification_result, "assignments", None)
        if assignments is None:
            assignments = getattr(classification_result, "classifications", None)

        if assignments is None:
            return None

        for assignment in assignments:
            assignment_track_id = getattr(assignment, "track_id", None)
            if assignment_track_id == track_id:
                return assignment

        return None
