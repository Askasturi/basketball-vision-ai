"""Tests for YOLO detector configuration."""

import pytest

from basketball_vision.detection import YOLODetectorConfig


def test_default_configuration() -> None:
    """Test default configuration values."""

    config = YOLODetectorConfig()

    assert config.model_path == "yolov8n.pt"
    assert config.class_ids is None
    assert config.agnostic_nms is False
    assert config.max_detections == 300
    assert config.verbose is False


def test_custom_configuration() -> None:
    """Test custom configuration."""

    config = YOLODetectorConfig(
        model_path="models/basketball.pt",
        class_ids=(0, 1),
        agnostic_nms=True,
        max_detections=100,
        verbose=True,
    )

    assert config.model_path == "models/basketball.pt"
    assert config.class_ids == (0, 1)
    assert config.agnostic_nms is True
    assert config.max_detections == 100
    assert config.verbose is True


def test_empty_model_path_raises() -> None:
    """Empty model path should raise."""

    with pytest.raises(ValueError):
        YOLODetectorConfig(model_path="")


def test_invalid_max_detections() -> None:
    """Maximum detections must be positive."""

    with pytest.raises(ValueError):
        YOLODetectorConfig(max_detections=0)


def test_empty_class_ids_raises() -> None:
    """Empty class_ids should raise."""

    with pytest.raises(ValueError):
        YOLODetectorConfig(class_ids=())


def test_negative_class_id_raises() -> None:
    """Negative class IDs should raise."""

    with pytest.raises(ValueError):
        YOLODetectorConfig(class_ids=(0, -1))


def test_inherits_detector_config() -> None:
    """YOLO config inherits DetectorConfig."""

    config = YOLODetectorConfig()

    assert config.confidence_threshold == 0.25
    assert config.iou_threshold == 0.45
    assert config.image_size == 640
    assert config.batch_size == 1
