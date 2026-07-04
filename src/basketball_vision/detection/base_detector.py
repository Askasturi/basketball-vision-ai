"""Abstract base class for object detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod

from basketball_vision.detection.detections import DetectionResult


class BaseDetector(ABC):
    """Abstract interface for all detector implementations."""

    def __init__(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Return whether the detector has been initialized."""
        return self._loaded

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights and initialize inference resources."""

    @abstractmethod
    def predict(self, image) -> DetectionResult:
        """Run inference on a single image."""

    @abstractmethod
    def warmup(self) -> None:
        """Warm up the detector for consistent inference performance."""

    @abstractmethod
    def close(self) -> None:
        """Release all detector resources."""

    def __enter__(self):
        self.load_model()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
