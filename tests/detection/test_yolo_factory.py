"""Tests for YOLO detector factory integration."""

from basketball_vision.detection import (
    DetectorFactory,
    DetectorType,
    YOLODetector,
    YOLODetectorConfig,
)


def test_yolo_detector_factory_registration() -> None:
    """Factory should create YOLODetector when registered."""

    DetectorFactory.register(
        DetectorType.YOLO,
        lambda config: YOLODetector(config),  # type: ignore[arg-type]
    )

    detector = DetectorFactory.create(
        DetectorType.YOLO,
        YOLODetectorConfig(),
    )

    assert isinstance(detector, YOLODetector)
