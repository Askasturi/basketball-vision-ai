"""Detection framework."""

from .base_detector import BaseDetector
from .config import DetectorConfig
from .detections import BoundingBox, Detection, DetectionResult

__all__ = [
    "BaseDetector",
    "BoundingBox",
    "Detection",
    "DetectionResult",
    "DetectorConfig",
]
