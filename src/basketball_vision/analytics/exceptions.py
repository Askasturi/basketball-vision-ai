"""Exceptions for basketball analytics."""


class AnalyticsError(Exception):
    """Base exception for analytics errors."""


class AnalyticsConfigurationError(AnalyticsError):
    """Raised when analytics configuration is invalid."""


class AnalyticsExportError(AnalyticsError):
    """Raised when analytics export fails."""
