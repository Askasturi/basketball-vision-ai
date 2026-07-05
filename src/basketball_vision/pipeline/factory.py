"""Factory for creating the unified basketball vision pipeline."""

from typing import Any

from basketball_vision.pipeline.config import PipelineConfig
from basketball_vision.pipeline.pipeline import BasketballVisionPipeline


class BasketballVisionPipelineFactory:
    """Factory for BasketballVisionPipeline."""

    @staticmethod
    def create(config: PipelineConfig | None = None) -> BasketballVisionPipeline:
        """Create a BasketballVisionPipeline with default project components."""
        pipeline_config = config or PipelineConfig()

        detector = BasketballVisionPipelineFactory._create_detector(
            pipeline_config.detector_config
        )
        tracker = (
            BasketballVisionPipelineFactory._create_tracker(
                pipeline_config.tracker_config
            )
            if pipeline_config.enable_tracking
            else None
        )
        classifier = (
            BasketballVisionPipelineFactory._create_classifier(
                pipeline_config.classifier_config
            )
            if pipeline_config.enable_tracking and pipeline_config.enable_classification
            else None
        )
        recognizer = (
            BasketballVisionPipelineFactory._create_recognizer(
                pipeline_config.recognizer_config
            )
            if pipeline_config.enable_tracking and pipeline_config.enable_recognition
            else None
        )

        return BasketballVisionPipeline(
            detector=detector,
            tracker=tracker,
            classifier=classifier,
            recognizer=recognizer,
            config=pipeline_config,
        )

    @staticmethod
    def _call_factory(factory: Any, config: Any | None) -> Any:
        if hasattr(factory, "create"):
            return factory.create(config) if config is not None else factory.create()
        if config is not None:
            return factory(config)
        return factory()

    @staticmethod
    def _create_detector(config: Any | None) -> Any:
        from basketball_vision.detection.factory import DetectorFactory

        return BasketballVisionPipelineFactory._call_factory(DetectorFactory, config)

    @staticmethod
    def _create_tracker(config: Any | None) -> Any:
        from basketball_vision.tracking.factory import TrackerFactory

        return BasketballVisionPipelineFactory._call_factory(TrackerFactory, config)

    @staticmethod
    def _create_classifier(config: Any | None) -> Any:
        from basketball_vision.classification.factory import ClassificationFactory

        return BasketballVisionPipelineFactory._call_factory(
            ClassificationFactory,
            config,
        )

    @staticmethod
    def _create_recognizer(config: Any | None) -> Any:
        from basketball_vision.recognition.factory import RecognitionFactory

        return BasketballVisionPipelineFactory._call_factory(RecognitionFactory, config)
