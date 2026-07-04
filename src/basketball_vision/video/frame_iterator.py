"""
Iterator for reading frames from a video.
"""

from __future__ import annotations

from collections.abc import Iterator

from basketball_vision.video.frame import VideoFrame
from basketball_vision.video.loader import VideoLoader


class FrameIterator(Iterator[VideoFrame]):
    """
    Iterate over frames in a video.

    Each iteration returns a VideoFrame containing
    the image, frame index, and timestamp.
    """

    def __init__(self, loader: VideoLoader) -> None:
        self._loader = loader
        self._capture = loader.capture
        self._fps = loader.metadata.fps
        self._frame_index = 0

    def __iter__(self) -> FrameIterator:
        return self

    def __next__(self) -> VideoFrame:
        success, image = self._capture.read()

        if not success:
            raise StopIteration

        frame = VideoFrame(
            index=self._frame_index,
            timestamp=self._frame_index / self._fps,
            image=image,
        )

        self._frame_index += 1

        return frame
