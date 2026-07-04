"""Tests for the abstract detector interface."""

from basketball_vision.detection import BaseDetector, DetectionResult


class DummyDetector(BaseDetector):
    """Dummy detector implementation."""

    def load_model(self) -> None:
        self._loaded = True

    def predict(self, image) -> DetectionResult:
        return DetectionResult(
            frame_index=0,
            timestamp=0.0,
        )

    def warmup(self) -> None:
        pass

    def close(self) -> None:
        self._loaded = False


def test_context_manager() -> None:
    """Context manager loads and unloads the detector."""

    detector = DummyDetector()

    assert not detector.is_loaded

    with detector as loaded:
        assert loaded.is_loaded

    assert not detector.is_loaded


def test_predict_returns_detection_result() -> None:
    """Predict returns a DetectionResult."""

    detector = DummyDetector()

    detector.load_model()

    result = detector.predict(None)

    assert isinstance(result, DetectionResult)
