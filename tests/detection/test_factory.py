"""Tests for the detector factory."""

from basketball_vision.detection import (
    BaseDetector,
    DetectionResult,
    DetectorConfig,
    DetectorFactory,
    DetectorType,
)


class DummyDetector(BaseDetector):
    """Dummy detector used for testing."""

    def __init__(self, config: DetectorConfig) -> None:
        super().__init__()
        self.config = config

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


def setup_function() -> None:
    """Clear the registry before each test."""
    DetectorFactory.clear()


def test_register_and_create_detector() -> None:
    """A registered detector can be created."""

    DetectorFactory.register(
        DetectorType.YOLO,
        DummyDetector,
    )

    detector = DetectorFactory.create(
        DetectorType.YOLO,
        DetectorConfig(),
    )

    assert isinstance(detector, DummyDetector)


def test_unregister_detector() -> None:
    """A detector can be unregistered."""

    DetectorFactory.register(
        DetectorType.YOLO,
        DummyDetector,
    )

    DetectorFactory.unregister(
        DetectorType.YOLO,
    )

    try:
        DetectorFactory.create(
            DetectorType.YOLO,
            DetectorConfig(),
        )
    except ValueError:
        pass
    else:
        assert False


def test_clear_registry() -> None:
    """The registry can be cleared."""

    DetectorFactory.register(
        DetectorType.YOLO,
        DummyDetector,
    )

    DetectorFactory.clear()

    try:
        DetectorFactory.create(
            DetectorType.YOLO,
            DetectorConfig(),
        )
    except ValueError:
        pass
    else:
        assert False


def test_unknown_detector() -> None:
    """Unknown detector types raise an error."""

    try:
        DetectorFactory.create(
            DetectorType.YOLO,
            DetectorConfig(),
        )
    except ValueError:
        pass
    else:
        assert False
