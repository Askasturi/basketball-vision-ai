import numpy as np
import pytest

from basketball_vision.pipeline import (
    BasketballVisionPipeline,
    PipelineConfig,
    PipelineConfigurationError,
    PipelineExecutionError,
)


class MockDetector:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def detect(self, frame: np.ndarray) -> str:
        self.calls.append("detect")
        assert isinstance(frame, np.ndarray)
        return "detections"


class MockTracker:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def update(self, detections: str) -> str:
        self.calls.append("track")
        assert detections == "detections"
        return "tracking"


class MockClassifier:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def classify(self, tracking: str) -> str:
        self.calls.append("classify")
        assert tracking == "tracking"
        return "classification"


class MockRecognizer:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def recognize(
        self,
        tracking_result: str,
        classification_result: str | None = None,
    ) -> str:
        self.calls.append("recognize")
        assert tracking_result == "tracking"
        assert classification_result in {"classification", None}
        return "recognition"


class FailingDetector:
    def detect(self, frame: np.ndarray) -> str:
        raise RuntimeError("detector failed")


def test_pipeline_process_frame_full_flow() -> None:
    calls: list[str] = []
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    pipeline = BasketballVisionPipeline(
        detector=MockDetector(calls),
        tracker=MockTracker(calls),
        classifier=MockClassifier(calls),
        recognizer=MockRecognizer(calls),
    )

    result = pipeline.process_frame(frame, frame_index=3, timestamp=2.5)

    assert calls == ["detect", "track", "classify", "recognize"]
    assert result.frame_index == 3
    assert result.timestamp == 2.5
    assert result.frame is frame
    assert result.detections == "detections"
    assert result.tracking == "tracking"
    assert result.classification == "classification"
    assert result.recognition == "recognition"


def test_pipeline_tracking_disabled_skips_later_stages() -> None:
    calls: list[str] = []
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    pipeline = BasketballVisionPipeline(
        detector=MockDetector(calls),
        tracker=None,
        classifier=MockClassifier(calls),
        recognizer=MockRecognizer(calls),
        config=PipelineConfig(enable_tracking=False),
    )

    result = pipeline.process_frame(frame)

    assert calls == ["detect"]
    assert result.detections == "detections"
    assert result.tracking is None
    assert result.classification is None
    assert result.recognition is None


def test_pipeline_classification_disabled_recognition_still_runs() -> None:
    calls: list[str] = []
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    pipeline = BasketballVisionPipeline(
        detector=MockDetector(calls),
        tracker=MockTracker(calls),
        classifier=None,
        recognizer=MockRecognizer(calls),
        config=PipelineConfig(enable_classification=False),
    )

    result = pipeline.process_frame(frame)

    assert calls == ["detect", "track", "recognize"]
    assert result.classification is None
    assert result.recognition == "recognition"


def test_pipeline_recognition_disabled() -> None:
    calls: list[str] = []
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    pipeline = BasketballVisionPipeline(
        detector=MockDetector(calls),
        tracker=MockTracker(calls),
        classifier=MockClassifier(calls),
        recognizer=None,
        config=PipelineConfig(enable_recognition=False),
    )

    result = pipeline.process_frame(frame)

    assert calls == ["detect", "track", "classify"]
    assert result.recognition is None


def test_pipeline_process_video() -> None:
    calls: list[str] = []
    frames = [
        np.zeros((10, 10, 3), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.uint8),
    ]

    pipeline = BasketballVisionPipeline(
        detector=MockDetector(calls),
        tracker=MockTracker(calls),
        classifier=MockClassifier(calls),
        recognizer=MockRecognizer(calls),
    )

    results = pipeline.process_video(frames)

    assert len(results) == 2
    assert results[0].frame_index == 0
    assert results[1].frame_index == 1


def test_pipeline_requires_detector_by_default() -> None:
    with pytest.raises(PipelineConfigurationError):
        BasketballVisionPipeline(detector=None)


def test_pipeline_allows_missing_detector_when_not_required() -> None:
    pipeline = BasketballVisionPipeline(
        detector=None,
        config=PipelineConfig(
            require_detector=False,
            enable_tracking=False,
            enable_classification=False,
            enable_recognition=False,
        ),
    )

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    result = pipeline.process_frame(frame)

    assert result.detections is None


def test_pipeline_requires_tracker_when_tracking_enabled() -> None:
    with pytest.raises(PipelineConfigurationError):
        BasketballVisionPipeline(
            detector=object(),
            tracker=None,
            config=PipelineConfig(enable_tracking=True),
        )


def test_pipeline_wraps_stage_errors() -> None:
    pipeline = BasketballVisionPipeline(
        detector=FailingDetector(),
        config=PipelineConfig(
            enable_tracking=False,
            enable_classification=False,
            enable_recognition=False,
        ),
    )

    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(PipelineExecutionError):
        pipeline.process_frame(frame)
