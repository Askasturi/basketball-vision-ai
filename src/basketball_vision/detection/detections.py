"""Immutable data models for object detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned bounding box."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 <= self.x1:
            raise ValueError("x2 must be greater than x1.")

        if self.y2 <= self.y1:
            raise ValueError("y2 must be greater than y1.")

    @property
    def width(self) -> float:
        """Width of the bounding box."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Height of the bounding box."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Area of the bounding box."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """Center point of the bounding box."""
        return (
            self.x1 + self.width / 2,
            self.y1 + self.height / 2,
        )


@dataclass(frozen=True, slots=True)
class Detection:
    """Represents one detected object."""

    bounding_box: BoundingBox
    confidence: float
    class_id: int
    class_name: str
    tracking_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Collection of detections for one frame."""

    frame_index: int
    timestamp: float
    detections: tuple[Detection, ...] = ()

    @property
    def num_detections(self) -> int:
        """Return the number of detections."""
        return len(self.detections)

    def is_empty(self) -> bool:
        """Return True if no detections are present."""
        return self.num_detections == 0
