"""Example demonstrating YOLO object detection."""

from basketball_vision.detection import (
    DetectorFactory,
    DetectorType,
    YOLODetectorConfig,
)
from basketball_vision.video import VideoLoader


def main() -> None:
    """Run YOLO detection on a video."""

    config = YOLODetectorConfig(
        model_path="yolov8n.pt",
    )

    detector = DetectorFactory.create(
        DetectorType.YOLO,
        config,
    )

    detector.load_model()

    with VideoLoader("tests/assets/sample.mp4") as video:
        for frame in video.frames():
            result = detector.predict(
                frame.image,
                frame_index=frame.index,
                timestamp=frame.timestamp,
            )

            print(
                f"Frame {result.frame_index}: "
                f"{result.num_detections} detections"
            )

    detector.close()


if __name__ == "__main__":
    main()
