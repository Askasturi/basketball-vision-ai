"""Classification-specific exceptions."""


class ClassificationError(Exception):
    """Base error for classification failures."""


class InvalidClassificationConfigError(ClassificationError):
    """Raised when a classification config is invalid."""


class ClassifierNotRegisteredError(ClassificationError):
    """Raised when a requested classifier type is not registered."""
