from basketball_vision.pipeline import BasketballVisionPipelineFactory, PipelineConfig


class DummyDetector:
    def detect(self, frame):
        return "detections"


class DummyTracker:
    def update(self, detections):
        return "tracking"


class DummyClassifier:
    def classify(self, tracking):
        return "classification"


class DummyRecognizer:
    def recognize(self, tracking_result, classification_result=None):
        return "recognition"


def test_pipeline_factory_create(monkeypatch) -> None:
    monkeypatch.setattr(
        BasketballVisionPipelineFactory,
        "_create_detector",
        staticmethod(lambda config: DummyDetector()),
    )
    monkeypatch.setattr(
        BasketballVisionPipelineFactory,
        "_create_tracker",
        staticmethod(lambda config: DummyTracker()),
    )
    monkeypatch.setattr(
        BasketballVisionPipelineFactory,
        "_create_classifier",
        staticmethod(lambda config: DummyClassifier()),
    )
    monkeypatch.setattr(
        BasketballVisionPipelineFactory,
        "_create_recognizer",
        staticmethod(lambda config: DummyRecognizer()),
    )

    pipeline = BasketballVisionPipelineFactory.create()

    assert isinstance(pipeline.detector, DummyDetector)
    assert isinstance(pipeline.tracker, DummyTracker)
    assert isinstance(pipeline.classifier, DummyClassifier)
    assert isinstance(pipeline.recognizer, DummyRecognizer)


def test_pipeline_factory_respects_disabled_stages(monkeypatch) -> None:
    monkeypatch.setattr(
        BasketballVisionPipelineFactory,
        "_create_detector",
        staticmethod(lambda config: DummyDetector()),
    )

    pipeline = BasketballVisionPipelineFactory.create(
        PipelineConfig(
            enable_tracking=False,
            enable_classification=False,
            enable_recognition=False,
        )
    )

    assert isinstance(pipeline.detector, DummyDetector)
    assert pipeline.tracker is None
    assert pipeline.classifier is None
    assert pipeline.recognizer is None
