
"""Custom exceptions for player number recognition."""

from __future__ import annotations


class RecognitionError(Exception):
    """Base exception for recognition errors."""


class RecognitionConfigurationError(RecognitionError):
    """Raised when recognition configuration is invalid."""


class RecognitionInputError(RecognitionError):
    """Raised when recognition inputs are invalid."""
