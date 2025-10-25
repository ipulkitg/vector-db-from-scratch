from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from .entities import Chunk, Document, Library, Metadata
from .repositories import ChunkRepository, DocumentRepository, LibraryRepository
from .vector_store import DistanceResult
from .schemas import (
    ChunkCreate,
    ChunkUpdate,
    DocumentCreate,
    DocumentUpdate,
    LibraryCreate,
    LibraryUpdate,
)


class LibraryService:
    """Application-facing operations for library management."""

    def __init__(self, repository: LibraryRepository) -> None:
        self._repository = repository

    def create_library(self, data: LibraryCreate) -> Library:
        library = Library(**data.model_dump())
        return self._repository.add(library)

    def get_library(self, library_id: UUID) -> Optional[Library]:
        return self._repository.get(library_id)

    def list_libraries(self) -> List[Library]:
        return self._repository.list()

    def update_library(self, library_id: UUID, data: LibraryUpdate) -> Library:
        existing = self._require_library(library_id)
        updates = data.model_dump(exclude_unset=True)
        updated = existing.model_copy(update=updates)
        return self._repository.update(library_id, updated)

    def delete_library(self, library_id: UUID) -> bool:
        return self._repository.delete(library_id)

    def _require_library(self, library_id: UUID) -> Library:
        library = self._repository.get(library_id)
        if not library:
            raise ValueError(f"Library {library_id} not found")
        return library


class DocumentService:
    """Business logic for document CRUD operations."""

    def __init__(self, repository: DocumentRepository, library_repository: LibraryRepository) -> None:
        self._repository = repository
        self._libraries = library_repository

    def create_document(self, data: DocumentCreate) -> Document:
        self._ensure_library(data.library_id)
        document = Document(**data.model_dump())
        return self._repository.add(document)

    def get_document(self, document_id: UUID) -> Optional[Document]:
        return self._repository.get(document_id)

    def list_documents(self, library_id: Optional[UUID] = None) -> List[Document]:
        return self._repository.list(library_id=library_id)

    def update_document(self, document_id: UUID, data: DocumentUpdate) -> Document:
        document = self._require_document(document_id)
        updates = data.model_dump(exclude_unset=True)
        updated = document.model_copy(update=updates)
        return self._repository.update(document_id, updated)

    def delete_document(self, document_id: UUID) -> bool:
        return self._repository.delete(document_id)

    def _ensure_library(self, library_id: UUID) -> Library:
        library = self._libraries.get(library_id)
        if not library:
            raise ValueError(f"Library {library_id} not found")
        return library

    def _require_document(self, document_id: UUID) -> Document:
        document = self._repository.get(document_id)
        if not document:
            raise ValueError(f"Document {document_id} not found")
        return document


class ChunkService:
    """Business logic for chunk management and similarity search."""

    def __init__(
        self,
        repository: ChunkRepository,
        document_repository: DocumentRepository,
        library_repository: LibraryRepository,
    ) -> None:
        self._repository = repository
        self._documents = document_repository
        self._libraries = library_repository

    def create_chunk(self, data: ChunkCreate) -> Chunk:
        document = self._ensure_document(data.document_id)
        library = self._ensure_library(document.library_id)
        library.validate_chunk_embedding(data.embedding)
        chunk = Chunk(**data.model_dump())
        return self._repository.add(chunk)

    def get_chunk(self, chunk_id: UUID) -> Optional[Chunk]:
        return self._repository.get(chunk_id)

    def list_chunks(
        self,
        *,
        document_id: Optional[UUID] = None,
        library_id: Optional[UUID] = None,
    ) -> List[Chunk]:
        return self._repository.list(document_id=document_id, library_id=library_id)

    def update_chunk(self, chunk_id: UUID, data: ChunkUpdate) -> Chunk:
        chunk = self._require_chunk(chunk_id)
        document = self._ensure_document(chunk.document_id)
        library = self._ensure_library(document.library_id)

        updates = data.model_dump(exclude_unset=True)
        embedding = updates.get("embedding")
        if embedding is not None:
            library.validate_chunk_embedding(embedding)

        updated = chunk.model_copy(update=updates)
        return self._repository.update(chunk_id, updated)

    def delete_chunk(self, chunk_id: UUID) -> bool:
        return self._repository.delete(chunk_id)

    def search_chunks(
        self,
        library_id: UUID,
        query_vector: List[float],
        k: int = 10,
        metadata_filters: Optional[Metadata] = None,
    ) -> List[DistanceResult]:
        library = self._ensure_library(library_id)
        library.validate_chunk_embedding(query_vector)
        return self._repository.search(
            library_id,
            query_vector,
            k,
            metadata_filters=metadata_filters,
        )

    def _ensure_document(self, document_id: UUID) -> Document:
        document = self._documents.get(document_id)
        if not document:
            raise ValueError(f"Document {document_id} not found")
        return document

    def _ensure_library(self, library_id: UUID) -> Library:
        library = self._libraries.get(library_id)
        if not library:
            raise ValueError(f"Library {library_id} not found")
        return library

    def _require_chunk(self, chunk_id: UUID) -> Chunk:
        chunk = self._repository.get(chunk_id)
        if not chunk:
            raise ValueError(f"Chunk {chunk_id} not found")
        return chunk


__all__ = ["LibraryService", "DocumentService", "ChunkService"]
