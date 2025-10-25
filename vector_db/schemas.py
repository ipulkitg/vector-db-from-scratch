from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Generic, List, Literal, Optional, TypeVar, Union
from uuid import UUID

from pydantic import BaseModel, Field

MetadataValue = Union[str, int, float, bool]
Metadata = Dict[str, MetadataValue]

T = TypeVar("T")


class LibraryCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(default=None, max_length=1_000)
    metadata: Metadata = Field(default_factory=dict)
    embedding_dimension: int = Field(..., gt=0)
    distance_metric: Literal["cosine", "euclidean", "dot_product"] = Field(default="cosine")
    index_kind: Literal["flat", "random_projection"] = Field(default="flat")


class LibraryUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    description: Optional[str] = Field(default=None, max_length=1_000)
    metadata: Optional[Metadata] = None
    distance_metric: Optional[Literal["cosine", "euclidean", "dot_product"]] = None
    index_kind: Optional[Literal["flat", "random_projection"]] = None


class DocumentCreate(BaseModel):
    library_id: UUID
    name: str = Field(..., min_length=1, max_length=255)
    metadata: Metadata = Field(default_factory=dict)


class DocumentUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    metadata: Optional[Metadata] = None


class ChunkCreate(BaseModel):
    document_id: UUID
    text: str = Field(..., min_length=1, max_length=10_000)
    embedding: List[float] = Field(..., min_length=1)
    metadata: Metadata = Field(default_factory=dict)
    chunk_index: int = Field(..., ge=0)


class ChunkUpdate(BaseModel):
    text: Optional[str] = Field(default=None, min_length=1, max_length=10_000)
    embedding: Optional[List[float]] = Field(default=None, min_length=1)
    metadata: Optional[Metadata] = None
    chunk_index: Optional[int] = Field(default=None, ge=0)


class ChunkSearchRequest(BaseModel):
    query_vector: List[float] = Field(..., min_length=1)
    k: int = Field(default=10, gt=0)
    metadata_filters: Optional[Metadata] = None


class ChunkSearchResult(BaseModel):
    chunk_id: UUID
    distance: float


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response wrapper."""

    items: List[T]
    total: int
    skip: int
    limit: int
    has_more: bool

    @classmethod
    def paginate(cls, items: List[T], skip: int, limit: int) -> "PaginatedResponse[T]":
        """Create a paginated response from a list of items."""
        total = len(items)
        paginated_items = items[skip : skip + limit]
        has_more = skip + limit < total
        return cls(items=paginated_items, total=total, skip=skip, limit=limit, has_more=has_more)


class ChunkBatchCreate(BaseModel):
    """Batch creation of chunks for a single document."""

    document_id: UUID
    chunks: List[ChunkCreate] = Field(..., min_length=1, max_length=1000)


class ChunkBatchCreateResult(BaseModel):
    """Result of batch chunk creation."""

    created_count: int
    chunk_ids: List[UUID]


if TYPE_CHECKING:
    from .entities import Chunk, Document, Library


class LibraryResponse(BaseModel):
    id: UUID
    name: str
    description: Optional[str]
    metadata: Metadata
    document_count: int
    chunk_count: int
    embedding_dimension: int
    distance_metric: Literal["cosine", "euclidean", "dot_product"]
    index_kind: Literal["flat", "random_projection"]


class DocumentResponse(BaseModel):
    id: UUID
    library_id: UUID
    name: str
    metadata: Metadata
    chunk_count: int


class ChunkResponse(BaseModel):
    id: UUID
    document_id: UUID
    text: str
    metadata: Metadata
    chunk_index: int


def library_to_response(library: "Library") -> LibraryResponse:
    return LibraryResponse(
        id=library.id,
        name=library.name,
        description=library.description,
        metadata=library.metadata,
        document_count=library.document_count,
        chunk_count=library.chunk_count,
        embedding_dimension=library.embedding_dimension,
        distance_metric=library.distance_metric,
        index_kind=library.index_kind,
    )


def document_to_response(document: "Document") -> DocumentResponse:
    from .entities import Document

    return DocumentResponse(
        id=document.id,
        library_id=document.library_id,
        name=document.name,
        metadata=document.metadata,
        chunk_count=document.chunk_count,
    )


def chunk_to_response(chunk: "Chunk") -> ChunkResponse:
    from .entities import Chunk

    return ChunkResponse(
        id=chunk.id,
        document_id=chunk.document_id,
        text=chunk.text,
        metadata=chunk.metadata,
        chunk_index=chunk.chunk_index,
    )


__all__ = [
    "LibraryCreate",
    "LibraryUpdate",
    "LibraryResponse",
    "DocumentCreate",
    "DocumentUpdate",
    "DocumentResponse",
    "ChunkCreate",
    "ChunkUpdate",
    "ChunkResponse",
    "ChunkSearchRequest",
    "ChunkSearchResult",
    "PaginatedResponse",
    "ChunkBatchCreate",
    "ChunkBatchCreateResult",
    "library_to_response",
    "document_to_response",
    "chunk_to_response",
]
