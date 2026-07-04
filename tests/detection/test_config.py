import pytest

from basketball_vision.detection import DetectorConfig


def test_default_config():
    config = DetectorConfig()

    assert config.confidence_threshold == 0.25
    assert config.iou_threshold == 0.45
    assert config.device == "cpu"
    assert config.image_size == 640
    assert config.batch_size == 1
    assert not config.half_precision


def test_invalid_confidence():
    with pytest.raises(ValueError):
        DetectorConfig(confidence_threshold=1.5)


def test_invalid_iou():
    with pytest.raises(ValueError):
        DetectorConfig(iou_threshold=-0.1)


def test_invalid_image_size():
    with pytest.raises(ValueError):
        DetectorConfig(image_size=0)


def test_invalid_batch_size():
    with pytest.raises(ValueError):
        DetectorConfig(batch_size=0)
