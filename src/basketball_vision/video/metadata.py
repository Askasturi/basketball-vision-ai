"""
Immutable metadata model for video files.

This module defines the VideoMetadata dataclass, which stores
basic properties of a video in a single object.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """
    Immutable container for video metadata.

    Attributes:
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frames per second.
        frame_count: Total number of frames.
        codec: FourCC codec string.
        duration_seconds: Total duration of the video.
    """

    width: int
    height: int
    fps: float
    frame_count: int
    codec: str
    duration_seconds: float

    @property
    def resolution(self) -> tuple[int, int]:
        """
        Return the video resolution as (width, height).

        Returns:
            tuple[int, int]: Video resolution.
        """
        return self.width, self.height

    @property
    def aspect_ratio(self) -> float:
        """
        Return the width-to-height aspect ratio.

        Returns:
            float: Aspect ratio.

        Raises:
            ZeroDivisionError:
                If the video height is zero.
        """
        return self.width / self.height