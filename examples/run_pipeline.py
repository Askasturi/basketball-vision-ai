"""Run the Basketball Vision AI pipeline with real YOLO or mock detections."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from basketball_vision.classification import ColorTeamClassifier
from basketball_vision.detection import BoundingBox, Detection, DetectionResult
from basketball_vision.recognition import SimpleNumberRecognizer
from basketball_vision.tracking import SimpleTracker
from basketball_vision.visualization import Renderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Basketball Vision AI pipeline.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/demo/real_pipeline.mp4")
    )
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def load_yolo_model(model_path: str) -> Any:
    from ultralytics import YOLO

    return YOLO(model_path)


def get_mock_detections(frame: np.ndarray) -> list[dict[str, Any]]:
    h, w = frame.shape[:2]

    return [
        {
            "bbox": [int(w * 0.20), int(h * 0.25), int(w * 0.35), int(h * 0.80)],
            "class_name": "player",
            "confidence": 0.95,
        },
        {
            "bbox": [int(w * 0.60), int(h * 0.22), int(w * 0.78), int(h * 0.82)],
            "class_name": "player",
            "confidence": 0.92,
        },
        {
            "bbox": [int(w * 0.46), int(h * 0.12), int(w * 0.54), int(h * 0.25)],
            "class_name": "ball",
            "confidence": 0.88,
        },
    ]


def get_yolo_detections(
    frame: np.ndarray, model: Any, confidence: float
) -> list[dict[str, Any]]:
    results = model(frame, conf=confidence, verbose=False)
    detections: list[dict[str, Any]] = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            class_id = int(box.cls[0])
            raw_name = str(result.names.get(class_id, class_id))
            score = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            if raw_name == "person":
                class_name = "player"
            elif raw_name in {"sports ball", "ball", "basketball"}:
                class_name = "ball"
            else:
                continue

            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "class_name": class_name,
                    "confidence": score,
                }
            )

    return detections


def to_detection_result(
    raw_detections: list[dict[str, Any]],
    frame_index: int,
    timestamp: float,
) -> DetectionResult:
    detections: list[Detection] = []

    for raw in raw_detections:
        x1, y1, x2, y2 = raw["bbox"]
        class_name = raw["class_name"]

        detections.append(
            Detection(
                bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                confidence=float(raw["confidence"]),
                class_id=0 if class_name == "player" else 1,
                class_name=class_name,
            )
        )

    return DetectionResult(
        frame_index=frame_index,
        timestamp=timestamp,
        detections=tuple(detections),
    )


def run_pipeline(args: argparse.Namespace) -> Path:
    if not args.input.exists():
        raise FileNotFoundError(f"Input video not found: {args.input}")

    tracker = SimpleTracker()
    team_classifier = ColorTeamClassifier()
    number_recognizer = SimpleNumberRecognizer()
    renderer = Renderer()

    yolo_model = None if args.mock else load_yolo_model(args.model)

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_index = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_index += 1

        detections = (
            get_mock_detections(frame)
            if args.mock
            else get_yolo_detections(frame, yolo_model, args.confidence)
        )

        detection_result = to_detection_result(
            raw_detections=detections,
            frame_index=frame_index,
            timestamp=frame_index / fps,
        )
        tracking_result = tracker.update(detection_result)
        classification_result = team_classifier.classify(
            frame=frame, tracking_result=tracking_result
        )
        recognition_result = number_recognizer.recognize(
            frame=frame, tracking_result=tracking_result
        )

        rendered = renderer.render(
            frame=frame,
            pipeline_frame_result={
                "frame_index": frame_index,
                "objects": detections,
                "tracking": tracking_result,
                "classification": classification_result,
                "recognition": recognition_result,
            },
        )

        writer.write(rendered)

        if args.max_frames and frame_index >= args.max_frames:
            break

    cap.release()
    writer.release()

    if frame_index == 0:
        raise RuntimeError("No frames were processed.")

    return args.output


def main() -> None:
    args = parse_args()
    output = run_pipeline(args)
    print(f"Annotated video exported to: {output}")


if __name__ == "__main__":
    main()
