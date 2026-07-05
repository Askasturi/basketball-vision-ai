"""Pipeline-specific exceptions."""


class PipelineError(Exception):
    """Base exception for pipeline errors."""


class PipelineConfigurationError(PipelineError):
    """Raised when pipeline configuration or components are invalid."""


class PipelineExecutionError(PipelineError):
    """Raised when a pipeline stage fails during execution."""
