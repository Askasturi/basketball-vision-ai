"""Utilities for rendering annotated videos."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

from basketball_vision.visualization.renderer import Renderer


def write_annotated_video(
    frames: Iterable[np.ndarray],
    results: Iterable[object],
    output_path: str | Path,
    fps: float,
    renderer: Renderer | None = None,
) -> Path:
    """Write annotated frames to a video file."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    active_renderer = renderer or Renderer()
    writer: cv2.VideoWriter | None = None

    try:
        for frame, result in zip(frames, results, strict=False):
            annotated = active_renderer.render(frame, result)

            if writer is None:
                height, width = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(output),
                    fourcc,
                    fps,
                    (width, height),
                )

            writer.write(annotated)

    finally:
        if writer is not None:
            writer.release()

    return output
