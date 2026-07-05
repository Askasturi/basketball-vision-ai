
"""Data models for player number recognition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from basketball_vision.recognition.exceptions import RecognitionConfigurationError
from basketball_vision.recognition.types import NumberRecognitionStatus


@dataclass(frozen=True, slots=True)
class PlayerNumberRecognition:
    """Number recognition output for one tracked player."""

    track_id: int
    number: str | None
    confidence: float
    status: NumberRecognitionStatus
    track: Any
    team_assignment: Any | None = None

    def __post_init__(self) -> None:
        """Validate player number recognition data."""
        if self.track_id < 0:
            msg = "track_id must be non-negative"
            raise RecognitionConfigurationError(msg)

        if not 0.0 <= self.confidence <= 1.0:
            msg = "confidence must be between 0.0 and 1.0"
            raise RecognitionConfigurationError(msg)

        if self.status is NumberRecognitionStatus.RECOGNIZED and not self.number:
            msg = "recognized results must include a number"
            raise RecognitionConfigurationError(msg)

        if self.number is not None and not self.number.isdigit():
            msg = "number must contain only digits"
            raise RecognitionConfigurationError(msg)


@dataclass(frozen=True, slots=True)
class RecognitionResult:
    """Number recognition result for one video frame."""

    frame_index: int
    timestamp: float | None
    recognitions: tuple[PlayerNumberRecognition, ...]

    def __post_init__(self) -> None:
        """Validate recognition result."""
        if self.frame_index < 0:
            msg = "frame_index must be non-negative"
            raise RecognitionConfigurationError(msg)

        if self.timestamp is not None and self.timestamp < 0:
            msg = "timestamp must be non-negative when provided"
            raise RecognitionConfigurationError(msg)
