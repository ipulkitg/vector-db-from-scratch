from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol
from uuid import UUID

from .disk_store import DiskVectorStore
from .entities import Chunk, Document, Library, Metadata
from .vector_store import DistanceResult, VectorStore


class LibraryRepository(Protocol):
    def add(self, library: Library) -> Library: ...

    def get(self, library_id: UUID) -> Optional[Library]: ...

    def list(self) -> List[Library]: ...

    def update(self, library_id: UUID, library: Library) -> Library: ...

    def delete(self, library_id: UUID) -> bool: ...


class DocumentRepository(Protocol):
    def add(self, document: Document) -> Document: ...

    def get(self, document_id: UUID) -> Optional[Document]: ...

    def list(self, library_id: Optional[UUID] = None) -> List[Document]: ...

    def update(self, document_id: UUID, document: Document) -> Document: ...

    def delete(self, document_id: UUID) -> bool: ...


class ChunkRepository(Protocol):
    def add(self, chunk: Chunk) -> Chunk: ...

    def get(self, chunk_id: UUID) -> Optional[Chunk]: ...

    def list(
        self,
        document_id: Optional[UUID] = None,
        library_id: Optional[UUID] = None,
    ) -> List[Chunk]: ...

    def update(self, chunk_id: UUID, chunk: Chunk) -> Chunk: ...

    def delete(self, chunk_id: UUID) -> bool: ...

    def search(
        self,
        library_id: UUID,
        query_vector: List[float],
        k: int = 10,
        metadata_filters: Optional[Metadata] = None,
    ) -> List[DistanceResult]: ...


@dataclass(slots=True)
class VectorStoreLibraryRepository(LibraryRepository):
    store: VectorStore

    def add(self, library: Library) -> Library:
        return self.store.add_library(library)

    def get(self, library_id: UUID) -> Optional[Library]:
        return self.store.get_library(library_id)

    def list(self) -> List[Library]:
        return self.store.list_libraries()

    def update(self, library_id: UUID, library: Library) -> Library:
        return self.store.update_library(library_id, library)

    def delete(self, library_id: UUID) -> bool:
        return self.store.delete_library(library_id)


@dataclass(slots=True)
class VectorStoreDocumentRepository(DocumentRepository):
    store: VectorStore

    def add(self, document: Document) -> Document:
        return self.store.add_document(document)

    def get(self, document_id: UUID) -> Optional[Document]:
        return self.store.get_document(document_id)

    def list(self, library_id: Optional[UUID] = None) -> List[Document]:
        return self.store.list_documents(library_id=library_id)

    def update(self, document_id: UUID, document: Document) -> Document:
        return self.store.update_document(document_id, document)

    def delete(self, document_id: UUID) -> bool:
        return self.store.delete_document(document_id)


@dataclass(slots=True)
class VectorStoreChunkRepository(ChunkRepository):
    store: VectorStore

    def add(self, chunk: Chunk) -> Chunk:
        return self.store.add_chunk(chunk)

    def get(self, chunk_id: UUID) -> Optional[Chunk]:
        return self.store.get_chunk(chunk_id)

    def list(
        self,
        document_id: Optional[UUID] = None,
        library_id: Optional[UUID] = None,
    ) -> List[Chunk]:
        return self.store.list_chunks(document_id=document_id, library_id=library_id)

    def update(self, chunk_id: UUID, chunk: Chunk) -> Chunk:
        return self.store.update_chunk(chunk_id, chunk)

    def delete(self, chunk_id: UUID) -> bool:
        return self.store.delete_chunk(chunk_id)

    def search(
        self,
        library_id: UUID,
        query_vector: List[float],
        k: int = 10,
        metadata_filters: Optional[Metadata] = None,
    ) -> List[DistanceResult]:
        return self.store.search(library_id, query_vector, k, metadata_filters=metadata_filters)


@dataclass(slots=True)
class DiskLibraryRepository(LibraryRepository):
    store: DiskVectorStore

    def add(self, library: Library) -> Library:
        return self.store.add_library(library)

    def get(self, library_id: UUID) -> Optional[Library]:
        return self.store.get_library(library_id)

    def list(self) -> List[Library]:
        return self.store.list_libraries()

    def update(self, library_id: UUID, library: Library) -> Library:
        return self.store.update_library(library_id, library)

    def delete(self, library_id: UUID) -> bool:
        return self.store.delete_library(library_id)


@dataclass(slots=True)
class DiskDocumentRepository(DocumentRepository):
    store: DiskVectorStore

    def add(self, document: Document) -> Document:
        return self.store.add_document(document)

    def get(self, document_id: UUID) -> Optional[Document]:
        return self.store.get_document(document_id)

    def list(self, library_id: Optional[UUID] = None) -> List[Document]:
        return self.store.list_documents(library_id=library_id)

    def update(self, document_id: UUID, document: Document) -> Document:
        return self.store.update_document(document_id, document)

    def delete(self, document_id: UUID) -> bool:
        return self.store.delete_document(document_id)


@dataclass(slots=True)
class DiskChunkRepository(ChunkRepository):
    store: DiskVectorStore

    def add(self, chunk: Chunk) -> Chunk:
        return self.store.add_chunk(chunk)

    def get(self, chunk_id: UUID) -> Optional[Chunk]:
        return self.store.get_chunk(chunk_id)

    def list(
        self,
        document_id: Optional[UUID] = None,
        library_id: Optional[UUID] = None,
    ) -> List[Chunk]:
        return self.store.list_chunks(document_id=document_id, library_id=library_id)

    def update(self, chunk_id: UUID, chunk: Chunk) -> Chunk:
        return self.store.update_chunk(chunk_id, chunk)

    def delete(self, chunk_id: UUID) -> bool:
        return self.store.delete_chunk(chunk_id)

    def search(
        self,
        library_id: UUID,
        query_vector: List[float],
        k: int = 10,
        metadata_filters: Optional[Metadata] = None,
    ) -> List[DistanceResult]:
        # Note: DiskVectorStore search doesn't support metadata_filters yet
        # This matches the in-memory implementation
        return self.store.search(library_id, query_vector, k)


__all__ = [
    "LibraryRepository",
    "DocumentRepository",
    "ChunkRepository",
    "VectorStoreLibraryRepository",
    "VectorStoreDocumentRepository",
    "VectorStoreChunkRepository",
    "DiskLibraryRepository",
    "DiskDocumentRepository",
    "DiskChunkRepository",
]
