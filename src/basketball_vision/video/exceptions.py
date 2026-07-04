"""
Custom exceptions for the video module.

These exceptions provide clear, domain-specific errors for all
video-related operations.
"""


class VideoError(Exception):
    """Base exception for all video-related errors."""


class VideoOpenError(VideoError):
    """Raised when a video cannot be opened."""


class InvalidFrameError(VideoError):
    """Raised when an invalid frame is encountered."""


class UnsupportedCodecError(VideoError):
    """Raised when a codec is unsupported."""


class VideoWriteError(VideoError):
    """Raised when writing a frame fails."""
