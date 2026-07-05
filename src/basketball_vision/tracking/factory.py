from basketball_vision.tracking.base_tracker import BaseTracker
from basketball_vision.tracking.config import TrackerConfig
from basketball_vision.tracking.exceptions import TrackerNotFoundError
from basketball_vision.tracking.simple_tracker import SimpleTracker
from basketball_vision.tracking.types import TrackerType


class TrackerFactory:
    @staticmethod
    def create(
        tracker_type: TrackerType | str,
        config: TrackerConfig | None = None,
    ) -> BaseTracker:
        try:
            normalized_type = TrackerType(tracker_type)
        except ValueError as error:
            raise TrackerNotFoundError(
                f"Unsupported tracker type: {tracker_type}"
            ) from error

        if normalized_type == TrackerType.SIMPLE:
            return SimpleTracker(config=config)

        raise TrackerNotFoundError(f"Unsupported tracker type: {tracker_type}")
