"""Utilities for converting detector outputs into framework models."""

from __future__ import annotations

from basketball_vision.detection.detections import (
    BoundingBox,
    Detection,
    DetectionResult,
)


class YOLOConverter:
    """Converts Ultralytics YOLO predictions into framework models."""

    @staticmethod
    def to_detection_result(
        *,
        results,
        frame_index: int,
        timestamp: float,
    ) -> DetectionResult:
        """Convert a YOLO Results object into a DetectionResult.

        Args:
            results:
                Ultralytics Results instance.

            frame_index:
                Index of the processed frame.

            timestamp:
                Timestamp of the processed frame.

        Returns:
            DetectionResult containing immutable detection objects.
        """

        detections: list[Detection] = []

        boxes = getattr(results, "boxes", None)

        if boxes is None:
            return DetectionResult(
                frame_index=frame_index,
                timestamp=timestamp,
                detections=(),
            )

        names = getattr(results, "names", {})

        for box in boxes:

            x1, y1, x2, y2 = (
                float(v)
                for v in box.xyxy[0].tolist()
            )

            bbox = BoundingBox(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )

            class_id = int(box.cls.item())

            detection = Detection(
                bounding_box=bbox,
                confidence=float(box.conf.item()),
                class_id=class_id,
                class_name=names.get(
                    class_id,
                    str(class_id),
                ),
            )

            detections.append(detection)

        return DetectionResult(
            frame_index=frame_index,
            timestamp=timestamp,
            detections=tuple(detections),
        )
