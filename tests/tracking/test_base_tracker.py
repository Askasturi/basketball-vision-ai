from basketball_vision.detection import DetectionResult
from basketball_vision.tracking import BaseTracker, TrackerConfig, TrackingResult


class DummyTracker(BaseTracker):
    def update(self, detection_result: DetectionResult) -> TrackingResult:
        return TrackingResult(
            frame_index=detection_result.frame_index,
            timestamp=detection_result.timestamp,
            tracks=(),
        )

    def reset(self) -> None:
        return None


def test_base_tracker_uses_default_config() -> None:
    tracker = DummyTracker()

    assert isinstance(tracker.config, TrackerConfig)


def test_base_tracker_accepts_custom_config() -> None:
    config = TrackerConfig(iou_threshold=0.7)
    tracker = DummyTracker(config=config)

    assert tracker.config == config
