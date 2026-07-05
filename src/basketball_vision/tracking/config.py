from dataclasses import dataclass

from basketball_vision.tracking.exceptions import TrackerConfigurationError


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    iou_threshold: float = 0.3
    max_missed_frames: int = 30
    min_hits: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise TrackerConfigurationError(
                "iou_threshold must be between 0.0 and 1.0")

        if self.max_missed_frames < 0:
            raise TrackerConfigurationError("max_missed_frames must be >= 0")

        if self.min_hits < 1:
            raise TrackerConfigurationError("min_hits must be >= 1")
