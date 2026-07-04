"""Detection framework."""

from .base_detector import BaseDetector
from .detections import BoundingBox, Detection, DetectionResult

__all__ = [
    "BaseDetector",
    "BoundingBox",
    "Detection",
    "DetectionResult",
]
