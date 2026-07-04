from basketball_vision.detection import BaseDetector, DetectionResult


class DummyDetector(BaseDetector):
    def load_model(self):
        self._loaded = True

    def predict(self, image):
        return DetectionResult(
            frame_index=0,
            timestamp=0.0,
        )

    def warmup(self):
        pass

    def close(self):
        self._loaded = False


def test_context_manager():
    detector = DummyDetector()

    assert not detector.is_loaded

    with detector as d:
        assert d.is_loaded

    assert not detector.is_loaded


def test_predict_returns_detection_result():
    detector = DummyDetector()

    detector.load_model()

    result = detector.predict(None)

    assert isinstance(result, DetectionResult)
