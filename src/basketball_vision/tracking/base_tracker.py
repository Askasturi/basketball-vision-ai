from abc import ABC, abstractmethod

from basketball_vision.detection import DetectionResult
from basketball_vision.tracking.config import TrackerConfig
from basketball_vision.tracking.track import TrackingResult


class BaseTracker(ABC):
    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()

    @abstractmethod
    def update(self, detection_result: DetectionResult) -> TrackingResult:
        """Update tracker state from a detection result."""

    @abstractmethod
    def reset(self) -> None:
        """Reset tracker state."""
