"""Factory for detector implementations."""

from __future__ import annotations

from collections.abc import Callable

from basketball_vision.detection.base_detector import BaseDetector
from basketball_vision.detection.config import DetectorConfig
from basketball_vision.detection.types import DetectorType


class DetectorFactory:
    """Factory responsible for creating detector instances."""

    _registry: dict[
        DetectorType,
        Callable[[DetectorConfig], BaseDetector],
    ] = {}

    @classmethod
    def register(
        cls,
        detector_type: DetectorType,
        constructor: Callable[[DetectorConfig], BaseDetector],
    ) -> None:
        """Register a detector implementation."""
        cls._registry[detector_type] = constructor

    @classmethod
    def unregister(cls, detector_type: DetectorType) -> None:
        """Remove a registered detector."""
        cls._registry.pop(detector_type, None)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered detector implementations."""
        cls._registry.clear()

    @classmethod
    def create(
        cls,
        detector_type: DetectorType,
        config: DetectorConfig,
    ) -> BaseDetector:
        """Create a detector instance."""

        try:
            constructor = cls._registry[detector_type]
        except KeyError as exc:
            raise ValueError(
                f"No detector registered for '{detector_type}'."
            ) from exc

        return constructor(config)
