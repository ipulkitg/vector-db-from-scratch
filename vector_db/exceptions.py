"""Custom exception hierarchy for the vector database."""
from __future__ import annotations

from typing import Any
from uuid import UUID


class VectorDBError(Exception):
    """Base exception for all vector database errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


# Resource errors (404)
class ResourceNotFoundError(VectorDBError):
    """Raised when a requested resource does not exist."""

    def __init__(self, resource_type: str, resource_id: UUID | str) -> None:
        message = f"{resource_type} {resource_id} not found"
        super().__init__(message, {"resource_type": resource_type, "resource_id": str(resource_id)})
        self.resource_type = resource_type
        self.resource_id = resource_id


class LibraryNotFoundError(ResourceNotFoundError):
    """Raised when a library is not found."""

    def __init__(self, library_id: UUID) -> None:
        super().__init__("Library", library_id)


class DocumentNotFoundError(ResourceNotFoundError):
    """Raised when a document is not found."""

    def __init__(self, document_id: UUID) -> None:
        super().__init__("Document", document_id)


class ChunkNotFoundError(ResourceNotFoundError):
    """Raised when a chunk is not found."""

    def __init__(self, chunk_id: UUID) -> None:
        super().__init__("Chunk", chunk_id)


# Validation errors (400)
class ValidationError(VectorDBError):
    """Raised when input validation fails."""

    pass


class DimensionMismatchError(ValidationError):
    """Raised when vector dimensions don't match library requirements."""

    def __init__(self, expected: int, actual: int, context: str = "") -> None:
        message = f"Dimension mismatch: expected {expected}, got {actual}"
        if context:
            message = f"{context}: {message}"
        super().__init__(
            message,
            {"expected_dimension": expected, "actual_dimension": actual, "context": context},
        )
        self.expected = expected
        self.actual = actual


class InvalidEmbeddingError(ValidationError):
    """Raised when an embedding is invalid (empty, non-numeric, etc.)."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Invalid embedding: {reason}")


class InvalidMetricError(ValidationError):
    """Raised when an unsupported distance metric is requested."""

    def __init__(self, metric: str, supported: list[str]) -> None:
        message = f"Unsupported metric '{metric}'. Supported: {', '.join(supported)}"
        super().__init__(message, {"metric": metric, "supported_metrics": supported})


class InvalidIndexKindError(ValidationError):
    """Raised when an unsupported index kind is requested."""

    def __init__(self, index_kind: str, supported: list[str]) -> None:
        message = f"Unsupported index kind '{index_kind}'. Supported: {', '.join(supported)}"
        super().__init__(message, {"index_kind": index_kind, "supported_kinds": supported})


# Conflict errors (409)
class ConflictError(VectorDBError):
    """Raised when an operation conflicts with existing state."""

    pass


class ResourceAlreadyExistsError(ConflictError):
    """Raised when attempting to create a resource that already exists."""

    def __init__(self, resource_type: str, resource_id: UUID) -> None:
        message = f"{resource_type} {resource_id} already exists"
        super().__init__(message, {"resource_type": resource_type, "resource_id": str(resource_id)})


class ImmutableFieldError(ConflictError):
    """Raised when attempting to modify an immutable field."""

    def __init__(self, field_name: str, reason: str = "") -> None:
        message = f"Cannot modify field '{field_name}'"
        if reason:
            message = f"{message}: {reason}"
        super().__init__(message, {"field": field_name, "reason": reason})


class DimensionChangeError(ConflictError):
    """Raised when attempting to change embedding dimension with existing chunks."""

    def __init__(self, library_id: UUID, chunk_count: int) -> None:
        message = f"Cannot change embedding dimension: library {library_id} has {chunk_count} chunks"
        super().__init__(
            message, {"library_id": str(library_id), "chunk_count": chunk_count}
        )


# Storage errors (500)
class StorageError(VectorDBError):
    """Base class for storage-related errors."""

    pass


class PersistenceError(StorageError):
    """Raised when disk persistence operations fail."""

    def __init__(self, operation: str, path: str, original_error: Exception | None = None) -> None:
        message = f"Persistence error during {operation}: {path}"
        if original_error:
            message = f"{message} ({type(original_error).__name__}: {original_error})"
        super().__init__(
            message,
            {
                "operation": operation,
                "path": path,
                "original_error": str(original_error) if original_error else None,
            },
        )
        self.original_error = original_error


class IndexError(StorageError):
    """Raised when vector index operations fail."""

    def __init__(self, operation: str, reason: str) -> None:
        message = f"Index error during {operation}: {reason}"
        super().__init__(message, {"operation": operation, "reason": reason})


# Search errors (400)
class SearchError(VectorDBError):
    """Raised when search operations fail."""

    pass


class InvalidSearchParameterError(SearchError):
    """Raised when search parameters are invalid."""

    def __init__(self, parameter: str, value: Any, reason: str) -> None:
        message = f"Invalid search parameter '{parameter}': {reason}"
        super().__init__(
            message, {"parameter": parameter, "value": str(value), "reason": reason}
        )


__all__ = [
    "VectorDBError",
    # Resource errors
    "ResourceNotFoundError",
    "LibraryNotFoundError",
    "DocumentNotFoundError",
    "ChunkNotFoundError",
    # Validation errors
    "ValidationError",
    "DimensionMismatchError",
    "InvalidEmbeddingError",
    "InvalidMetricError",
    "InvalidIndexKindError",
    # Conflict errors
    "ConflictError",
    "ResourceAlreadyExistsError",
    "ImmutableFieldError",
    "DimensionChangeError",
    # Storage errors
    "StorageError",
    "PersistenceError",
    "IndexError",
    # Search errors
    "SearchError",
    "InvalidSearchParameterError",
]
