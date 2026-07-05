import pytest

from basketball_vision.tracking import (
    SimpleTracker,
    TrackerConfig,
    TrackerFactory,
    TrackerNotFoundError,
    TrackerType,
)


def test_tracker_factory_creates_simple_tracker_from_enum() -> None:
    tracker = TrackerFactory.create(TrackerType.SIMPLE)

    assert isinstance(tracker, SimpleTracker)


def test_tracker_factory_creates_simple_tracker_from_string() -> None:
    tracker = TrackerFactory.create("simple")

    assert isinstance(tracker, SimpleTracker)


def test_tracker_factory_passes_config() -> None:
    config = TrackerConfig(iou_threshold=0.6)

    tracker = TrackerFactory.create(TrackerType.SIMPLE, config=config)

    assert tracker.config == config


def test_tracker_factory_rejects_unknown_tracker_type() -> None:
    with pytest.raises(TrackerNotFoundError):
        TrackerFactory.create("unknown")
