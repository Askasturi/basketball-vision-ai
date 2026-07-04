import pytest

from basketball_vision.detection import (
    BoundingBox,
    Detection,
    DetectionResult,
)


def test_bounding_box_dimensions():
    box = BoundingBox(10, 20, 30, 60)

    assert box.width == 20
    assert box.height == 40
    assert box.area == 800


def test_bounding_box_center():
    box = BoundingBox(0, 0, 10, 20)

    assert box.center == (5.0, 10.0)


def test_invalid_box():
    with pytest.raises(ValueError):
        BoundingBox(10, 0, 5, 20)


def test_detection_confidence_validation():
    box = BoundingBox(0, 0, 10, 10)

    with pytest.raises(ValueError):
        Detection(
            bounding_box=box,
            confidence=2.0,
            class_id=0,
            class_name="player",
        )


def test_detection_result():
    box = BoundingBox(0, 0, 10, 10)

    detection = Detection(
        bounding_box=box,
        confidence=0.95,
        class_id=0,
        class_name="player",
    )

    result = DetectionResult(
        frame_index=5,
        timestamp=0.2,
        detections=(detection,),
    )

    assert result.num_detections == 1
    assert not result.is_empty()


def test_empty_result():
    result = DetectionResult(
        frame_index=0,
        timestamp=0.0,
    )

    assert result.is_empty()
