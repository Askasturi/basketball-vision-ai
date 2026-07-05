import numpy as np
import pytest

from basketball_vision.pipeline import PipelineFrameResult


def test_pipeline_frame_result_stores_values() -> None:
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    result = PipelineFrameResult(
        frame_index=2,
        timestamp=1.5,
        frame=frame,
        detections="detections",
        tracking="tracking",
        classification="classification",
        recognition="recognition",
    )

    assert result.frame_index == 2
    assert result.timestamp == 1.5
    assert result.frame is frame
    assert result.detections == "detections"
    assert result.tracking == "tracking"
    assert result.classification == "classification"
    assert result.recognition == "recognition"


def test_pipeline_frame_result_rejects_negative_frame_index() -> None:
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        PipelineFrameResult(frame_index=-1, timestamp=None, frame=frame)


def test_pipeline_frame_result_rejects_negative_timestamp() -> None:
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        PipelineFrameResult(frame_index=0, timestamp=-1.0, frame=frame)


def test_pipeline_frame_result_rejects_non_numpy_frame() -> None:
    with pytest.raises(TypeError):
        PipelineFrameResult(frame_index=0, timestamp=None, frame="not-frame")
