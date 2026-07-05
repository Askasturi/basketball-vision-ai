
"""Full mock pipeline demo for Basketball Vision AI.

Pipeline:
Video/frame generation -> detection -> tracking -> team classification
-> number recognition -> visualization -> export.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

OUTPUT_PATH = Path("outputs/demo/full_pipeline_demo.mp4")


@dataclass(frozen=True)
class DemoObject:
    """One mocked pipeline object."""

    track_id: int
    class_name: str
    bbox: tuple[int, int, int, int]
    confidence: float
    team: str
    number: str


def create_demo_frame(
    frame_index: int,
    width: int = 960,
    height: int = 540,
) -> np.ndarray:
    """Create one synthetic basketball-style frame."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (35, 115, 55)

    cv2.rectangle(frame, (60, 70), (900, 470), (210, 210, 210), 3)
    cv2.circle(frame, (width // 2, height // 2), 80, (210, 210, 210), 2)
    cv2.line(frame, (width // 2, 70), (width // 2, 470), (210, 210, 210), 2)

    cv2.putText(
        frame,
        f"Basketball Vision AI Full Pipeline Demo | Frame {frame_index}",
        (35, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def create_mock_pipeline_result(frame_index: int) -> dict[str, object]:
    """Create mocked detections after tracking, classification, and OCR."""
    offset = frame_index * 5

    objects = [
        DemoObject(
            track_id=1,
            class_name="player",
            bbox=(140 + offset, 190, 220 + offset, 360),
            confidence=0.94,
            team="Home",
            number="23",
        ),
        DemoObject(
            track_id=2,
            class_name="player",
            bbox=(420 - offset // 2, 170, 505 - offset // 2, 350),
            confidence=0.91,
            team="Away",
            number="11",
        ),
        DemoObject(
            track_id=3,
            class_name="player",
            bbox=(650 - offset, 200, 735 - offset, 375),
            confidence=0.89,
            team="Home",
            number="7",
        ),
        DemoObject(
            track_id=4,
            class_name="ball",
            bbox=(515 + offset // 2, 250, 550 + offset // 2, 285),
            confidence=0.87,
            team="N/A",
            number="N/A",
        ),
    ]

    return {
        "frame_index": frame_index,
        "objects": objects,
    }


def draw_pipeline_result(
    frame: np.ndarray,
    pipeline_result: dict[str, object],
) -> np.ndarray:
    """Draw mocked full-pipeline results on a frame."""
    rendered = frame.copy()
    objects = pipeline_result["objects"]

    for obj in objects:
        if not isinstance(obj, DemoObject):
            continue

        x1, y1, x2, y2 = obj.bbox
        label = (
            f"ID {obj.track_id} | {obj.class_name} | "
            f"{obj.team} | #{obj.number} | {obj.confidence:.2f}"
        )

        cv2.rectangle(rendered, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.rectangle(rendered, (x1, y1 - 28), (x1 + 420, y1), (0, 0, 0), -1)
        cv2.putText(
            rendered,
            label,
            (x1 + 6, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        rendered,
        (
            "Video -> Detection -> Tracking -> Team Classification "
            "-> Number Recognition -> Visualization -> Export"
        ),
        (35, 510),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return rendered


def ensure_output_dir(path: Path = OUTPUT_PATH) -> Path:
    """Create the output directory and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_demo_video(
    output_path: Path = OUTPUT_PATH,
    frame_count: int = 24,
    fps: int = 12,
    width: int = 960,
    height: int = 540,
) -> Path:
    """Generate and save the full pipeline demo video."""
    output_path = ensure_output_dir(output_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    try:
        for frame_index in range(frame_count):
            frame = create_demo_frame(frame_index, width=width, height=height)
            result = create_mock_pipeline_result(frame_index)
            rendered = draw_pipeline_result(frame, result)
            writer.write(rendered)
    finally:
        writer.release()

    return output_path


def main() -> None:
    """Run the full pipeline demo."""
    output_path = write_demo_video()
    print(f"Saved full pipeline demo video to: {output_path}")


if __name__ == "__main__":
    main()
