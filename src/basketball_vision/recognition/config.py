
"""Configuration for player number recognition."""

from __future__ import annotations

from dataclasses import dataclass, field

from basketball_vision.recognition.exceptions import RecognitionConfigurationError
from basketball_vision.recognition.types import NumberRecognizerType


@dataclass(frozen=True, slots=True)
class RecognitionConfig:
    """Base configuration for number recognition."""

    recognizer_type: NumberRecognizerType = NumberRecognizerType.SIMPLE
    confidence_threshold: float = 0.50
    include_lost_tracks: bool = False
    include_removed_tracks: bool = False
    valid_min_number: int = 0
    valid_max_number: int = 99
    track_id_to_number: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate recognition configuration."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            msg = "confidence_threshold must be between 0.0 and 1.0"
            raise RecognitionConfigurationError(msg)

        if self.valid_min_number < 0:
            msg = "valid_min_number must be non-negative"
            raise RecognitionConfigurationError(msg)

        if self.valid_max_number < self.valid_min_number:
            msg = "valid_max_number must be greater than or equal to valid_min_number"
            raise RecognitionConfigurationError(msg)

        for track_id, number in self.track_id_to_number.items():
            if not isinstance(track_id, int):
                msg = "track_id_to_number keys must be integers"
                raise RecognitionConfigurationError(msg)

            if track_id < 0:
                msg = "track_id_to_number keys must be non-negative"
                raise RecognitionConfigurationError(msg)

            self._validate_number(number)

    def _validate_number(self, number: str) -> None:
        if not isinstance(number, str):
            msg = "recognized numbers must be strings"
            raise RecognitionConfigurationError(msg)

        if not number.isdigit():
            msg = "recognized numbers must contain only digits"
            raise RecognitionConfigurationError(msg)

        value = int(number)
        if not self.valid_min_number <= value <= self.valid_max_number:
            msg = (
                "recognized number must be between "
                f"{self.valid_min_number} and {self.valid_max_number}"
            )
            raise RecognitionConfigurationError(msg)
