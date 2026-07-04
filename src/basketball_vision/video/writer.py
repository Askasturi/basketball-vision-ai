"""
Utilities for writing video files.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from basketball_vision.video.exceptions import VideoWriteError


class VideoWriter:
    """
    Production-quality wrapper around OpenCV VideoWriter.
    """

    def __init__(
        self,
        output_path: str | Path,
        width: int,
        height: int,
        fps: float,
        codec: str = "mp4v",
    ) -> None:
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*codec)

        self._writer = cv2.VideoWriter(
            str(self._path),
            fourcc,
            fps,
            (width, height),
        )

        if not self._writer.isOpened():
            raise VideoWriteError(
                f"Unable to create video writer: {self._path}"
            )

    def write(self, frame: np.ndarray) -> None:
        """
        Write a single frame.
        """
        if frame is None:
            raise VideoWriteError("Cannot write a None frame.")

        self._writer.write(frame)

    def release(self) -> None:
        """
        Release writer resources.
        """
        self._writer.release()

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
