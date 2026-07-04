"""YOLO-specific detector configuration."""

from __future__ import annotations

from dataclasses import dataclass

from basketball_vision.detection.config import DetectorConfig


@dataclass(frozen=True, slots=True)
class YOLODetectorConfig(DetectorConfig):
    """Configuration for the YOLO detector.

    Extends the generic DetectorConfig with YOLO-specific options while
    preserving the common detector interface.
    """

    model_path: str = "yolov8n.pt"

    class_ids: tuple[int, ...] | None = None

    agnostic_nms: bool = False

    max_detections: int = 300

    verbose: bool = False

    def __post_init__(self) -> None:
        """Validate configuration."""
        DetectorConfig.__post_init__(self)

        if not self.model_path.strip():
            raise ValueError(
                "model_path cannot be empty."
            )

        if self.max_detections <= 0:
            raise ValueError(
                "max_detections must be greater than zero."
            )

        if self.class_ids is not None:

            if len(self.class_ids) == 0:
                raise ValueError(
                    "class_ids cannot be empty."
                )

            if any(class_id < 0 for class_id in self.class_ids):
                raise ValueError(
                    "class_ids must contain non-negative integers."
                )
