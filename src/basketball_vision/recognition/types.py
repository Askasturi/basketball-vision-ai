
"""Recognition-related enums."""

from __future__ import annotations

from enum import StrEnum


class NumberRecognitionStatus(StrEnum):
    """Status for a player number recognition result."""

    RECOGNIZED = "recognized"
    UNKNOWN = "unknown"
    LOW_CONFIDENCE = "low_confidence"


class NumberRecognizerType(StrEnum):
    """Supported number recognizer backend types."""

    SIMPLE = "simple"
