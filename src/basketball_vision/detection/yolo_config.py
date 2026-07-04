"""YOLO-specific detector configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from basketball_vision.detection.config import DetectorConfig


@dataclass(frozen=True, slots=True)
class YOLODetectorConfig(DetectorConfig):
    """Configuration for the YOLO detector.

    Extends the generic DetectorConfig with options that are specific
    to Ultralytics YOLO while preserving the common detector interface.

    Attributes:
        model_path:
            Path to a YOLO model weights file.

        class_ids:
            Optional list of class IDs to detect.
            None means detect every supported class.

        agnostic_nms:
            Whether Non-Maximum Suppression ignores class labels.

        max_detections:
            Maximum number of detections returned for one image.

        verbose:
            Enables Ultralytics logging output.
    """

    model_path: str = "yolov8n.pt"

    class_ids: list[int] | None = field(default=None)

    agnostic_nms: bool = False

    max_detections: int = 300

    verbose: bool = False

    def __post_init__(self) -> None:
        """Validate YOLO configuration."""
        super().__post_init__()

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
                    "class_ids cannot be an empty list."
                )

            if any(class_id < 0 for class_id in self.class_ids):
                raise ValueError(
                    "class_ids must contain non-negative integers."
                )
