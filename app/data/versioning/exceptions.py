"""
Custom exceptions for the versioning framework.
"""


class VersioningError(Exception):
    """Base exception for versioning errors."""
    pass


class VersionNotFoundError(VersioningError):
    """Raised when a requested version is not found."""
    pass


class VersionConflictError(VersioningError):
    """Raised when there's a version conflict."""
    pass


class ChecksumMismatchError(VersioningError):
    """Raised when checksums don't match."""
    pass


class DatasetNotFoundError(VersioningError):
    """Raised when a dataset is not found."""
    pass


class ArtifactNotFoundError(VersioningError):
    """Raised when an artifact is not found."""
    pass


class InvalidVersionError(VersioningError):
    """Raised when an invalid version is specified."""
    pass


class RegistryError(VersioningError):
    """Raised for registry-related errors."""
    pass


class StorageError(VersioningError):
    """Raised for storage-related errors."""
    pass


class LineageError(VersioningError):
    """Raised for lineage-related errors."""
    pass


class DuplicateEntityError(VersioningError):
    """Raised when attempting to create a duplicate entity."""
    pass


class UnsupportedFormatError(VersioningError):
    """Raised when a file format is not supported."""
    pass


class ValidationError(VersioningError):
    """Raised when validation fails."""
    pass


class RollbackError(VersioningError):
    """Raised when rollback fails."""
    pass