"""Unified Basketball Vision pipeline."""

from collections.abc import Iterable
from typing import Any

import numpy as np

from basketball_vision.pipeline.config import PipelineConfig
from basketball_vision.pipeline.exceptions import (
    PipelineConfigurationError,
    PipelineExecutionError,
)
from basketball_vision.pipeline.results import PipelineFrameResult


class BasketballVisionPipeline:
    """Orchestrates detection, tracking, classification, and recognition."""

    def __init__(
        self,
        detector: Any | None,
        tracker: Any | None = None,
        classifier: Any | None = None,
        recognizer: Any | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.config = config or PipelineConfig()

        self.detector = detector
        self.tracker = tracker
        self.classifier = classifier
        self.recognizer = recognizer

        self._validate_components()

    def _validate_components(self) -> None:
        if self.config.require_detector and self.detector is None:
            msg = "A detector is required when require_detector=True."
            raise PipelineConfigurationError(msg)

        if self.detector is not None and not hasattr(self.detector, "detect"):
            msg = "detector must provide a detect(frame) method."
            raise PipelineConfigurationError(msg)

        if self.config.enable_tracking:
            if self.tracker is None:
                msg = "A tracker is required when enable_tracking=True."
                raise PipelineConfigurationError(msg)
            if not hasattr(self.tracker, "update"):
                msg = "tracker must provide an update(detections) method."
                raise PipelineConfigurationError(msg)

        if self.config.enable_tracking and self.config.enable_classification:
            if self.classifier is None:
                msg = "A classifier is required when classification is enabled."
                raise PipelineConfigurationError(msg)
            if not hasattr(self.classifier, "classify"):
                msg = "classifier must provide a classify(tracking) method."
                raise PipelineConfigurationError(msg)

        if self.config.enable_tracking and self.config.enable_recognition:
            if self.recognizer is None:
                msg = "A recognizer is required when recognition is enabled."
                raise PipelineConfigurationError(msg)
            if not hasattr(self.recognizer, "recognize"):
                msg = "recognizer must provide a recognize(...) method."
                raise PipelineConfigurationError(msg)

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        timestamp: float | None = None,
    ) -> PipelineFrameResult:
        """Process a single frame through the enabled pipeline stages."""
        if not isinstance(frame, np.ndarray):
            msg = "frame must be a numpy.ndarray."
            raise TypeError(msg)

        try:
            detections = self.detector.detect(
                frame) if self.detector is not None else None

            tracking = None
            if self.config.enable_tracking and self.tracker is not None:
                tracking = self.tracker.update(detections)

            classification = None
            if (
                self.config.enable_tracking
                and tracking is not None
                and self.config.enable_classification
                and self.classifier is not None
            ):
                classification = self.classifier.classify(tracking)

            recognition = None
            if (
                self.config.enable_tracking
                and tracking is not None
                and self.config.enable_recognition
                and self.recognizer is not None
            ):
                recognition = self.recognizer.recognize(
                    tracking_result=tracking,
                    classification_result=classification,
                )

            return PipelineFrameResult(
                frame_index=frame_index,
                timestamp=timestamp,
                frame=frame,
                detections=detections,
                tracking=tracking,
                classification=classification,
                recognition=recognition,
            )

        except PipelineExecutionError:
            raise
        except Exception as exc:
            msg = f"Pipeline execution failed: {exc}"
            raise PipelineExecutionError(msg) from exc

    def process_video(self, frames: Iterable[np.ndarray]) -> list[PipelineFrameResult]:
        """Process an iterable of frames."""
        return [
            self.process_frame(frame=frame, frame_index=index)
            for index, frame in enumerate(frames)
        ]
