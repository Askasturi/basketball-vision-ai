"""Configuration models for object detectors."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    """Configuration for object detectors."""

    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "cpu"
    image_size: int = 640
    batch_size: int = 1
    half_precision: bool = False

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between 0 and 1."
            )

        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError(
                "iou_threshold must be between 0 and 1."
            )

        if self.image_size <= 0:
            raise ValueError(
                "image_size must be greater than zero."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )
