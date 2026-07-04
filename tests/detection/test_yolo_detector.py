"""Tests for YOLO detector implementation."""

from __future__ import annotations

import numpy as np
import pytest

import basketball_vision.detection.yolo_detector as yolo_detector_module
from basketball_vision.detection import YOLODetector, YOLODetectorConfig
from basketball_vision.detection.exceptions import (
    InferenceError,
    ModelLoadError,
    ModelNotLoadedError,
)


class FakeTensor:
    """Minimal tensor-like object for tests."""

    def __init__(self, value):
        self._value = value

    def item(self):
        """Return scalar value."""
        return self._value


class FakeXYXY:
    """Minimal coordinate container."""

    def __init__(self, coords):
        self._coords = coords

    def tolist(self):
        """Return coordinates."""
        return self._coords


class FakeBox:
    """Fake YOLO box."""

    def __init__(self, coords, confidence, class_id):
        self.xyxy = [FakeXYXY(coords)]
        self.conf = FakeTensor(confidence)
        self.cls = FakeTensor(class_id)


class FakeResults:
    """Fake YOLO results object."""

    names = {
        0: "person",
        1: "basketball",
    }

    boxes = [
        FakeBox([10, 20, 110, 220], 0.95, 0),
        FakeBox([250, 300, 280, 330], 0.88, 1),
    ]


class FakeYOLO:
    """Fake Ultralytics YOLO model."""

    instances = []

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.predict_calls = []
        FakeYOLO.instances.append(self)

    def predict(self, source, **kwargs):
        """Fake prediction."""
        self.predict_calls.append(
            {
                "source": source,
                "kwargs": kwargs,
            }
        )
        return [FakeResults()]


class EmptyYOLO(FakeYOLO):
    """Fake model returning no results."""

    def predict(self, source, **kwargs):
        """Fake empty prediction."""
        self.predict_calls.append(
            {
                "source": source,
                "kwargs": kwargs,
            }
        )
        return []


class FailingYOLO(FakeYOLO):
    """Fake model that fails during inference."""

    def predict(self, source, **kwargs):
        """Raise during prediction."""
        raise RuntimeError("inference failed")


class LoadFailingYOLO:
    """Fake model that fails during construction."""

    def __init__(self, model_path: str):
        raise RuntimeError(f"could not load {model_path}")


@pytest.fixture(autouse=True)
def clear_fake_instances() -> None:
    """Clear fake model instances before each test."""
    FakeYOLO.instances.clear()


@pytest.fixture
def image() -> np.ndarray:
    """Return a dummy image."""
    return np.zeros((64, 64, 3), dtype=np.uint8)


def test_initial_state() -> None:
    """Detector starts unloaded."""

    detector = YOLODetector(YOLODetectorConfig())

    assert detector.is_loaded is False
    assert detector.config.model_path == "yolov8n.pt"
    assert detector.model_name == "yolov8n.pt"


def test_model_property_raises_before_load() -> None:
    """Accessing model before loading raises."""

    detector = YOLODetector(YOLODetectorConfig())

    with pytest.raises(ModelNotLoadedError):
        _ = detector.model


def test_load_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Load model initializes YOLO backend."""

    monkeypatch.setattr(yolo_detector_module, "YOLO", FakeYOLO)

    detector = YOLODetector(YOLODetectorConfig(model_path="custom.pt"))
    detector.load_model()

    assert detector.is_loaded is True
    assert detector.model.model_path == "custom.pt"


def test_load_model_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling load_model twice should not reload."""

    monkeypatch.setattr(yolo_detector_module, "YOLO", FakeYOLO)

    detector = YOLODetector(YOLODetectorConfig())
    detector.load_model()
    detector.load_model()

    assert len(FakeYOLO.instances) == 1


def test_load_model_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model construction failure raises ModelLoadError."""

    monkeypatch.setattr(yolo_detector_module, "YOLO", LoadFailingYOLO)

    detector = YOLODetector(YOLODetectorConfig(model_path="bad.pt"))

    with pytest.raises(ModelLoadError):
        detector.load_model()

    assert detector.is_loaded is False


def test_warmup_requires_loaded_model() -> None:
    """Warmup before loading raises."""

    detector = YOLODetector(YOLODetectorConfig())

    with pytest.raises(ModelNotLoadedError):
        detector.warmup()


def test_warmup_runs_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Warmup runs one dummy prediction."""

    monkeypatch.setattr(yolo_detector_module, "YOLO", FakeYOLO)

    detector = YOLODetector(YOLODetectorConfig(image_size=32))
    detector.load_model()
    detector.warmup()

    assert len(detector.model.predict_calls) == 1
    call = detector.model.predict_calls[0]
    assert call["source"].shape == (32, 32, 3)


def test_predict_requires_loaded_model(image: np.ndarray) -> None:
    """Predict before loading raises."""

    detector = YOLODetector(YOLODetectorConfig())

    with pytest.raises(ModelNotLoadedError):
        detector.predict(image)


def test_predict_returns_detection_result(
    monkeypatch: pytest.MonkeyPatch,
    image: np.ndarray,
) -> None:
    """Predict converts YOLO results into DetectionResult."""

    monkeypatch.setattr(yolo_detector_module, "YOLO", FakeYOLO)

    detector = YOLODetector(YOLODetectorConfig())
    detector.load_model()

    result = detector.predict(
        image,
        frame_index=12,
        timestamp=3.5,
    )

    assert result.frame_index == 12
    assert result.timestamp == 3.5
    assert result.num_detections == 2
    assert result.detections[0].class_name == "person"
    assert result.detections[1].class_name == "basketball"


def test_predict_empty_results(
    monkeypatch: pytest.MonkeyPatch,
    image: np.ndarray,
) -> None:
    """Empty YOLO result list returns an empty DetectionResult."""

    monkeypatch.setattr(yolo_detector_module, "YOLO", EmptyYOLO)

    detector = YOLODetector(YOLODetectorConfig())
    detector.load_model()

    result = detector.predict(
        image,
        frame_index=7,
        timestamp=1.25,
    )

    assert result.frame_index == 7
    assert result.timestamp == 1.25
    assert result.is_empty()


def test_predict_failure_raises_inference_error(
    monkeypatch: pytest.MonkeyPatch,
    image: np.ndarray,
) -> None:
    """Inference failures are wrapped in InferenceError."""

    monkeypatch.setattr(yolo_detector_module, "YOLO", FailingYOLO)

    detector = YOLODetector(YOLODetectorConfig())
    detector.load_model()

    with pytest.raises(InferenceError):
        detector.predict(image)


def test_close_unloads_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Close releases the model."""

    monkeypatch.setattr(yolo_detector_module, "YOLO", FakeYOLO)

    detector = YOLODetector(YOLODetectorConfig())
    detector.load_model()
    detector.close()

    assert detector.is_loaded is False

    with pytest.raises(ModelNotLoadedError):
        _ = detector.model


def test_context_manager_loads_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Context manager loads and closes the detector."""

    monkeypatch.setattr(yolo_detector_module, "YOLO", FakeYOLO)

    detector = YOLODetector(YOLODetectorConfig())

    with detector as active_detector:
        assert active_detector.is_loaded is True

    assert detector.is_loaded is False


def test_prediction_kwargs() -> None:
    """Prediction kwargs should reflect configuration."""

    config = YOLODetectorConfig(
        confidence_threshold=0.5,
        iou_threshold=0.6,
        image_size=320,
        class_ids=(0, 1),
        agnostic_nms=True,
        max_detections=50,
        verbose=True,
        half_precision=True,
    )

    detector = YOLODetector(config)

    kwargs = detector._prediction_kwargs()

    assert kwargs["imgsz"] == 320
    assert kwargs["conf"] == 0.5
    assert kwargs["iou"] == 0.6
    assert kwargs["classes"] == (0, 1)
    assert kwargs["agnostic_nms"] is True
    assert kwargs["max_det"] == 50
    assert kwargs["verbose"] is True
    assert kwargs["half"] is True
