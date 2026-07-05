"""Configuration objects for player team classification."""

from dataclasses import dataclass, field
from typing import Any

from basketball_vision.classification.exceptions import InvalidClassificationConfigError

RGBColor = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class ClassificationConfig:
    """Base configuration for player classification."""

    min_confidence: float = 0.0
    include_lost_tracks: bool = False
    include_removed_tracks: bool = False

    def __post_init__(self) -> None:
        """Validate base classification config."""
        if not 0.0 <= self.min_confidence <= 1.0:
            msg = "min_confidence must be between 0.0 and 1.0"
            raise InvalidClassificationConfigError(msg)


@dataclass(frozen=True, slots=True)
class ColorTeamClassifierConfig(ClassificationConfig):
    """Configuration for color-based team classification.

    Colors are RGB tuples.
    """

    team_a_color: RGBColor = (255, 0, 0)
    team_b_color: RGBColor = (0, 0, 255)
    unknown_distance_threshold: float = 140.0
    crop_padding: int = 0
    min_crop_area: int = 16
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate color classifier config."""
        ClassificationConfig.__post_init__(self)

        self._validate_color(self.team_a_color, "team_a_color")
        self._validate_color(self.team_b_color, "team_b_color")

        if self.unknown_distance_threshold < 0:
            msg = "unknown_distance_threshold must be non-negative"
            raise InvalidClassificationConfigError(msg)

        if self.crop_padding < 0:
            msg = "crop_padding must be non-negative"
            raise InvalidClassificationConfigError(msg)

        if self.min_crop_area <= 0:
            msg = "min_crop_area must be positive"
            raise InvalidClassificationConfigError(msg)

    @staticmethod
    def _validate_color(color: RGBColor, name: str) -> None:
        if len(color) != 3:
            msg = f"{name} must contain exactly 3 RGB values"
            raise InvalidClassificationConfigError(msg)

        for value in color:
            if not isinstance(value, int):
                msg = f"{name} values must be integers"
                raise InvalidClassificationConfigError(msg)

            if not 0 <= value <= 255:
                msg = f"{name} values must be between 0 and 255"
                raise InvalidClassificationConfigError(msg)
