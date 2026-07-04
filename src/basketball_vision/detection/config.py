"""Configuration models for object detectors."""

from __future__ import annotations

from dataclasses import dataclass

from basketball_vision.detection.types import DeviceType


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    """Configuration for object detectors.

    Attributes:
        confidence_threshold:
            Minimum confidence score required to keep a detection.
        iou_threshold:
            Intersection over Union (IoU) threshold used during
            Non-Maximum Suppression.
        device:
            Target device used for inference.
        image_size:
            Input image size used by the detector.
        batch_size:
            Number of images processed in one inference batch.
        half_precision:
            Whether FP16 inference should be enabled when supported.
    """

    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: DeviceType = DeviceType.CPU
    image_size: int = 640
    batch_size: int = 1
    half_precision: bool = False

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between 0.0 and 1.0."
            )

        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError(
                "iou_threshold must be between 0.0 and 1.0."
            )

        if self.image_size <= 0:
            raise ValueError(
                "image_size must be greater than zero."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )
