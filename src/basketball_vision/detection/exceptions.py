"""Exceptions for the detection subsystem."""


class DetectionError(Exception):
    """Base exception for all detection-related errors."""


class ModelNotLoadedError(DetectionError):
    """Raised when inference is attempted before a model is loaded."""


class ModelLoadError(DetectionError):
    """Raised when a detector model cannot be loaded."""


class InferenceError(DetectionError):
    """Raised when inference fails."""
