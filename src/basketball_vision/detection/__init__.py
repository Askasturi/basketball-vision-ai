"""Detection framework."""

from .base_detector import BaseDetector
from .config import DetectorConfig
from .detections import BoundingBox, Detection, DetectionResult
from .factory import DetectorFactory
from .types import DetectorType, DeviceType
from .yolo_config import YOLODetectorConfig

__all__ = [
    "BaseDetector",
    "BoundingBox",
    "Detection",
    "DetectionResult",
    "DetectorConfig",
    "YOLODetectorConfig",
    "DetectorFactory",
    "DeviceType",
    "DetectorType",
]
