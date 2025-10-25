from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

MetadataValue = Union[str, int, float, bool]
Metadata = Dict[str, MetadataValue]


class BaseEntity(BaseModel):
    """Base class shared across all vector database entities."""

    id: UUID = Field(default_factory=uuid4, description="Unique identifier")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp in UTC",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last mutation timestamp in UTC",
    )

    model_config = ConfigDict(validate_assignment=True)

    def update_timestamp(self) -> None:
        """Refresh the mutation timestamp."""
        self.updated_at = datetime.now(timezone.utc)


class Chunk(BaseEntity):
    """
    Atomic piece of text and its embedding. This is what gets indexed/searched.
    """

    document_id: UUID = Field(..., description="Identifier of the parent document")
    text: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Chunk content",
    )
    embedding: List[float] = Field(
        ...,
        min_length=1,
        description="Vector embedding for the chunk",
    )
    metadata: Metadata = Field(
        default_factory=dict,
        description="Additional structured information for filtering",
    )
    chunk_index: int = Field(
        ...,
        ge=0,
        description="Stable position of the chunk inside its document",
    )

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, value: List[float]) -> List[float]:
        if not value:
            raise ValueError("Embedding cannot be empty")
        if not all(isinstance(component, (int, float)) for component in value):
            raise ValueError("Embedding must contain only numeric values")
        return [float(component) for component in value]

    @property
    def embedding_dimension(self) -> int:
        """Convenience accessor for the embedding dimensionality."""
        return len(self.embedding)

    model_config = ConfigDict(validate_assignment=True)


class Document(BaseEntity):
    """Represents the logical document that groups chunks together."""

    library_id: UUID = Field(..., description="Identifier of the parent library")
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human readable document name",
    )
    metadata: Metadata = Field(
        default_factory=dict,
        description="Structured metadata accessible for filtering",
    )
    chunk_count: int = Field(0, ge=0, description="Number of chunks stored")

    def increment_chunk_count(self) -> None:
        self.chunk_count += 1
        self.update_timestamp()

    def decrement_chunk_count(self) -> None:
        self.chunk_count = max(0, self.chunk_count - 1)
        self.update_timestamp()

    model_config = ConfigDict(validate_assignment=True)


class Library(BaseEntity):
    """Collection of documents that share indexing constraints."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Library name",
    )
    description: Optional[str] = Field(
        None,
        max_length=1_000,
        description="Optional library description",
    )
    metadata: Metadata = Field(
        default_factory=dict,
        description="Structured metadata for the library",
    )
    document_count: int = Field(0, ge=0, description="Stored document count")
    chunk_count: int = Field(0, ge=0, description="Total number of chunks")
    embedding_dimension: int = Field(
        ...,
        gt=0,
        description="Dimension enforced for every chunk embedding",
    )
    distance_metric: Literal["cosine", "euclidean", "dot_product"] = Field(
        default="cosine",
        description="Similarity metric to use for comparisons",
    )
    index_kind: Literal["flat", "random_projection"] = Field(
        default="flat",
        description="Index implementation used for vector search",
    )

    def validate_chunk_embedding(self, embedding: List[float]) -> None:
        """Ensure a chunk embedding matches the library dimension."""
        if len(embedding) != self.embedding_dimension:
            raise ValueError(
                (
                    "Embedding dimension mismatch: expected "
                    f"{self.embedding_dimension}, got {len(embedding)}"
                )
            )

    def increment_document_count(self) -> None:
        self.document_count += 1
        self.update_timestamp()

    def decrement_document_count(self) -> None:
        self.document_count = max(0, self.document_count - 1)
        self.update_timestamp()

    def add_chunks(self, count: int) -> None:
        if count < 0:
            raise ValueError("Chunk increments must be positive")
        self.chunk_count += count
        self.update_timestamp()

    def remove_chunks(self, count: int) -> None:
        if count < 0:
            raise ValueError("Chunk decrements must be positive")
        self.chunk_count = max(0, self.chunk_count - count)
        self.update_timestamp()

    model_config = ConfigDict(validate_assignment=True)


__all__ = ["BaseEntity", "Chunk", "Document", "Library"]
