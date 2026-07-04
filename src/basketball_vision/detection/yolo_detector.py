"""Ultralytics YOLO detector implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

from basketball_vision.detection.base_detector import BaseDetector
from basketball_vision.detection.converters import YOLOConverter
from basketball_vision.detection.detections import DetectionResult
from basketball_vision.detection.exceptions import (
    InferenceError,
    ModelLoadError,
    ModelNotLoadedError,
)
from basketball_vision.detection.yolo_config import YOLODetectorConfig


class YOLODetector(BaseDetector):
    """YOLO implementation of the BaseDetector interface.

    This class encapsulates all Ultralytics-specific functionality.
    Downstream modules only receive DetectionResult objects.
    """

    def __init__(self, config: YOLODetectorConfig) -> None:
        """Initialize the detector."""
        super().__init__()
        self._config = config
        self._model: YOLO | None = None

    @property
    def config(self) -> YOLODetectorConfig:
        """Return the detector configuration."""
        return self._config

    @property
    def model(self) -> YOLO:
        """Return the loaded YOLO model."""
        if self._model is None:
            raise ModelNotLoadedError("YOLO model has not been loaded.")

        return self._model

    @property
    def model_name(self) -> str:
        """Return the configured model path."""
        return self._config.model_path

    def load_model(self) -> None:
        """Load the configured YOLO model."""
        if self._loaded:
            return

        try:
            model_path = Path(self._config.model_path)
            self._model = YOLO(str(model_path))
            self._loaded = True
        except Exception as exc:
            self._model = None
            self._loaded = False
            raise ModelLoadError(
                f"Failed to load YOLO model from '{self._config.model_path}'."
            ) from exc

    def warmup(self) -> None:
        """Warm up the detector with a dummy inference."""
        if not self._loaded:
            raise ModelNotLoadedError(
                "Cannot warm up before loading the model."
            )

        dummy_image = np.zeros(
            (
                self._config.image_size,
                self._config.image_size,
                3,
            ),
            dtype=np.uint8,
        )

        try:
            self.model.predict(
                source=dummy_image,
                **self._prediction_kwargs(),
            )
        except Exception as exc:
            raise InferenceError("YOLO warmup inference failed.") from exc

    def predict(
        self,
        image: np.ndarray,
        *,
        frame_index: int = 0,
        timestamp: float = 0.0,
    ) -> DetectionResult:
        """Run object detection on an image."""
        if not self._loaded:
            raise ModelNotLoadedError(
                "Model must be loaded before calling predict()."
            )

        try:
            results = self._run_inference(image)
        except Exception as exc:
            raise InferenceError("YOLO inference failed.") from exc

        if not results:
            return DetectionResult(
                frame_index=frame_index,
                timestamp=timestamp,
                detections=(),
            )

        return YOLOConverter.to_detection_result(
            results=results[0],
            frame_index=frame_index,
            timestamp=timestamp,
        )

    def close(self) -> None:
        """Release detector resources."""
        self._model = None
        self._loaded = False

    def reload(self) -> None:
        """Reload the configured model."""
        self.close()
        self.load_model()

    def _run_inference(self, image: np.ndarray) -> list[Any]:
        """Execute YOLO inference."""
        return self.model.predict(
            source=image,
            **self._prediction_kwargs(),
        )

    def _prediction_kwargs(self) -> dict[str, Any]:
        """Return Ultralytics prediction keyword arguments."""
        kwargs: dict[str, Any] = {
            "imgsz": self._config.image_size,
            "conf": self._config.confidence_threshold,
            "iou": self._config.iou_threshold,
            "device": self._config.device.value,
            "classes": self._config.class_ids,
            "agnostic_nms": self._config.agnostic_nms,
            "max_det": self._config.max_detections,
            "verbose": self._config.verbose,
        }

        if self._config.half_precision:
            kwargs["half"] = True

        return kwargs

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        return (
            f"{self.__class__.__name__}("
            f"model_path={self._config.model_path!r}, "
            f"loaded={self._loaded})"
        )
