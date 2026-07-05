
"""Demo script for rendering basketball vision annotations on a sample frame."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from basketball_vision.visualization import Renderer, VisualizationConfig


def main() -> None:
    output_dir = Path("outputs/demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = np.full((720, 1280, 3), 245, dtype=np.uint8)

    detections = [
        {
            "bbox": [180, 160, 380, 560],
            "confidence": 0.94,
            "class_name": "player",
            "track_id": 7,
            "team": "home",
            "number": "23",
        },
        {
            "bbox": [760, 190, 940, 570],
            "confidence": 0.91,
            "class_name": "player",
            "track_id": 12,
            "team": "away",
            "number": "11",
        },
        {
            "bbox": [565, 310, 610, 355],
            "confidence": 0.88,
            "class_name": "ball",
        },
    ]

    renderer = Renderer(VisualizationConfig(show_frame_index=True))
    pipeline_result = {'frame_index': 1, 'objects': detections}
    rendered = renderer.render(frame=frame, pipeline_frame_result=pipeline_result)

    output_path = output_dir / "render_demo.jpg"
    cv2.imwrite(str(output_path), rendered)

    print(f"Saved demo image to: {output_path}")


if __name__ == "__main__":
    main()
