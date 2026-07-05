class TrackingError(Exception):
    """Base exception for tracking errors."""


class TrackerConfigurationError(TrackingError):
    """Raised when tracker configuration is invalid."""


class TrackerNotFoundError(TrackingError):
    """Raised when an unsupported tracker type is requested."""
