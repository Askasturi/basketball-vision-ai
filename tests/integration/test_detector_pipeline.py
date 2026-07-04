"""Integration tests for the detection pipeline."""

from basketball_vision.detection import DetectionResult


def test_detection_result_contract() -> None:
    """DetectionResult exposes the expected API."""

    result = DetectionResult(
        frame_index=0,
        timestamp=0.0,
        detections=(),
    )

    assert result.is_empty()
    assert result.num_detections == 0
