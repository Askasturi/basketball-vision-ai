"""
Video loading utilities.

This module provides the VideoLoader class, which is responsible for
opening videos, extracting metadata, and managing the underlying
OpenCV VideoCapture resource.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from basketball_vision.video.exceptions import VideoOpenError
from basketball_vision.video.metadata import VideoMetadata


class VideoLoader:
    """
    Load a video file and expose its metadata and capture object.

    The VideoLoader is responsible for opening the video,
    extracting metadata, and releasing resources safely.

    Example:
        with VideoLoader("game.mp4") as loader:
            metadata = loader.metadata
            capture = loader.capture
    """

    def __init__(self, video_path: str | Path) -> None:
        self._path = Path(video_path)

        if not self._path.exists():
            raise VideoOpenError(f"Video file does not exist: {self._path}")

        self._capture = cv2.VideoCapture(str(self._path))

        if not self._capture.isOpened():
            raise VideoOpenError(f"Unable to open video: {self._path}")

        self._metadata = self._read_metadata()

    @property
    def path(self) -> Path:
        """Return the video path."""
        return self._path

    @property
    def capture(self) -> cv2.VideoCapture:
        """Return the underlying OpenCV VideoCapture object."""
        return self._capture

    @property
    def metadata(self) -> VideoMetadata:
        """Return metadata for the loaded video."""
        return self._metadata

    def release(self) -> None:
        """Release the underlying VideoCapture resource."""
        if self._capture.isOpened():
            self._capture.release()

    def __enter__(self) -> VideoLoader:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()

    def _read_metadata(self) -> VideoMetadata:
        """
        Read metadata from the video.

        Returns:
            VideoMetadata containing video properties.
        """
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = int(self._capture.get(cv2.CAP_PROP_FOURCC))

        codec = "".join(
            chr((fourcc >> (8 * i)) & 0xFF)
            for i in range(4)
        ).strip()

        duration = frame_count / fps if fps > 0 else 0.0

        return VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            codec=codec,
            duration_seconds=duration,
        )
