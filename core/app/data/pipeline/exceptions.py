"""Pipeline-specific exceptions."""


class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass


class StageNotFoundError(PipelineError):
    """Raised when a stage is not registered."""
    pass


class StageExecutionError(PipelineError):
    """Raised when a stage fails during execution."""
    pass


class DependencyError(PipelineError):
    """Raised when stage dependencies cannot be resolved."""
    pass


class PipelineConfigurationError(PipelineError):
    """Raised when pipeline configuration is invalid."""
    pass


class CheckpointError(PipelineError):
    """Raised when checkpoint operations fail."""
    pass


class ParallelExecutionError(PipelineError):
    """Raised when parallel execution fails."""
    pass


class RetryExhaustedError(PipelineError):
    """Raised when all retry attempts are exhausted."""
    pass