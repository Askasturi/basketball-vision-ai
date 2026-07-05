"""Configuration objects for the unified basketball vision pipeline."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Configuration for BasketballVisionPipeline."""

    enable_tracking: bool = True
    enable_classification: bool = True
    enable_recognition: bool = True
    require_detector: bool = True

    detector_config: Any | None = None
    tracker_config: Any | None = None
    classifier_config: Any | None = None
    recognizer_config: Any | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "enable_tracking",
            "enable_classification",
            "enable_recognition",
            "require_detector",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, bool):
                msg = f"{field_name} must be a bool."
                raise TypeError(msg)
