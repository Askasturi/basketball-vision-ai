import pytest

from basketball_vision.tracking import TrackerConfig, TrackerConfigurationError


def test_default_tracker_config() -> None:
    config = TrackerConfig()

    assert config.iou_threshold == 0.3
    assert config.max_missed_frames == 30
    assert config.min_hits == 1


def test_custom_tracker_config() -> None:
    config = TrackerConfig(iou_threshold=0.5, max_missed_frames=10, min_hits=2)

    assert config.iou_threshold == 0.5
    assert config.max_missed_frames == 10
    assert config.min_hits == 2


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_invalid_iou_threshold_raises(threshold: float) -> None:
    with pytest.raises(TrackerConfigurationError):
        TrackerConfig(iou_threshold=threshold)


def test_negative_max_missed_frames_raises() -> None:
    with pytest.raises(TrackerConfigurationError):
        TrackerConfig(max_missed_frames=-1)


def test_invalid_min_hits_raises() -> None:
    with pytest.raises(TrackerConfigurationError):
        TrackerConfig(min_hits=0)
