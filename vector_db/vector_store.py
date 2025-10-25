from __future__ import annotations

from collections import defaultdict
from threading import RLock
from typing import Dict, List, Optional, Set
from uuid import UUID

import numpy as np

from .entities import Chunk, Document, Library, Metadata
from .indexes import DistanceResult, FlatIndex, RandomProjectionIndex, VectorIndex


class VectorStore:
    """
    Thread-safe in-memory store for libraries, documents, and chunks,
    bundling a simple vector index per library.
    """

    def __init__(self) -> None:
        self._libraries: Dict[UUID, Library] = {}
        self._documents: Dict[UUID, Document] = {}
        self._chunks: Dict[UUID, Chunk] = {}

        self._docs_by_library: Dict[UUID, Set[UUID]] = defaultdict(set)
        self._chunks_by_document: Dict[UUID, Set[UUID]] = defaultdict(set)
        self._chunks_by_library: Dict[UUID, Set[UUID]] = defaultdict(set)

        self._vector_index: Dict[UUID, VectorIndex] = {}
        self._lock = RLock()

    # Library operations
    def add_library(self, library: Library) -> Library:
        with self._lock:
            if library.id in self._libraries:
                raise ValueError(f"Library {library.id} already exists")

            self._libraries[library.id] = library
            self._docs_by_library[library.id] = set()
            self._chunks_by_library[library.id] = set()
            self._vector_index[library.id] = self._build_index(library)
            return library

    def get_library(self, library_id: UUID) -> Optional[Library]:
        with self._lock:
            return self._libraries.get(library_id)

    def list_libraries(self) -> List[Library]:
        with self._lock:
            return list(self._libraries.values())

    def update_library(self, library_id: UUID, updated_library: Library) -> Library:
        with self._lock:
            if library_id not in self._libraries:
                raise ValueError(f"Library {library_id} not found")
            if updated_library.id != library_id:
                raise ValueError("Library ID cannot change")

            existing = self._libraries[library_id]
            dimension_changed = (
                existing.embedding_dimension != updated_library.embedding_dimension
            )
            index_changed = existing.index_kind != updated_library.index_kind
            if existing.chunk_count > 0 and (dimension_changed or index_changed):
                raise ValueError("Cannot change index settings with existing chunks")

            if dimension_changed or index_changed:
                self._vector_index[library_id] = self._build_index(updated_library)

            updated_library.update_timestamp()
            self._libraries[library_id] = updated_library
            return updated_library

    def delete_library(self, library_id: UUID) -> bool:
        with self._lock:
            if library_id not in self._libraries:
                return False

            document_ids = list(self._docs_by_library[library_id])
            for document_id in document_ids:
                self.delete_document(document_id)

            del self._libraries[library_id]
            self._docs_by_library.pop(library_id, None)
            self._chunks_by_library.pop(library_id, None)
            self._vector_index.pop(library_id, None)
            return True

    # Document operations
    def add_document(self, document: Document) -> Document:
        with self._lock:
            if document.library_id not in self._libraries:
                raise ValueError(f"Library {document.library_id} not found")
            if document.id in self._documents:
                raise ValueError(f"Document {document.id} already exists")

            self._documents[document.id] = document
            self._docs_by_library[document.library_id].add(document.id)
            self._chunks_by_document[document.id] = set()

            library = self._libraries[document.library_id]
            library.increment_document_count()
            return document

    def get_document(self, document_id: UUID) -> Optional[Document]:
        with self._lock:
            return self._documents.get(document_id)

    def list_documents(self, library_id: Optional[UUID] = None) -> List[Document]:
        with self._lock:
            if library_id is not None:
                doc_ids = self._docs_by_library.get(library_id, set())
                return [self._documents[doc_id] for doc_id in doc_ids]
            return list(self._documents.values())

    def update_document(self, document_id: UUID, updated_document: Document) -> Document:
        with self._lock:
            if document_id not in self._documents:
                raise ValueError(f"Document {document_id} not found")
            if updated_document.id != document_id:
                raise ValueError("Document ID cannot change")

            existing = self._documents[document_id]
            if updated_document.library_id != existing.library_id:
                raise ValueError("Document library cannot change")

            updated_document.update_timestamp()
            self._documents[document_id] = updated_document
            return updated_document

    def delete_document(self, document_id: UUID) -> bool:
        with self._lock:
            if document_id not in self._documents:
                return False

            document = self._documents[document_id]
            chunk_ids = list(self._chunks_by_document.get(document_id, set()))
            for chunk_id in chunk_ids:
                self.delete_chunk(chunk_id)

            del self._documents[document_id]
            self._docs_by_library[document.library_id].discard(document_id)
            self._chunks_by_document.pop(document_id, None)

            library = self._libraries[document.library_id]
            library.decrement_document_count()
            return True

    # Chunk operations
    def add_chunk(self, chunk: Chunk) -> Chunk:
        with self._lock:
            if chunk.document_id not in self._documents:
                raise ValueError(f"Document {chunk.document_id} not found")
            if chunk.id in self._chunks:
                raise ValueError(f"Chunk {chunk.id} already exists")

            document = self._documents[chunk.document_id]
            library = self._libraries[document.library_id]
            library.validate_chunk_embedding(chunk.embedding)

            self._chunks[chunk.id] = chunk
            self._chunks_by_document[chunk.document_id].add(chunk.id)
            self._chunks_by_library[document.library_id].add(chunk.id)

            vector = np.asarray(chunk.embedding, dtype=np.float32)
            self._vector_index[document.library_id].add_vector(chunk.id, vector)

            document.increment_chunk_count()
            library.add_chunks(1)
            return chunk

    def get_chunk(self, chunk_id: UUID) -> Optional[Chunk]:
        with self._lock:
            return self._chunks.get(chunk_id)

    def list_chunks(
        self, document_id: Optional[UUID] = None, library_id: Optional[UUID] = None
    ) -> List[Chunk]:
        with self._lock:
            if document_id is not None:
                chunk_ids = self._chunks_by_document.get(document_id, set())
                return [self._chunks[cid] for cid in chunk_ids]
            if library_id is not None:
                chunk_ids = self._chunks_by_library.get(library_id, set())
                return [self._chunks[cid] for cid in chunk_ids]
            return list(self._chunks.values())

    def update_chunk(self, chunk_id: UUID, updated_chunk: Chunk) -> Chunk:
        with self._lock:
            if chunk_id not in self._chunks:
                raise ValueError(f"Chunk {chunk_id} not found")
            if updated_chunk.id != chunk_id:
                raise ValueError("Chunk ID cannot change")

            existing = self._chunks[chunk_id]
            if updated_chunk.document_id != existing.document_id:
                raise ValueError("Chunk document cannot change")

            document = self._documents[existing.document_id]
            library = self._libraries[document.library_id]
            library.validate_chunk_embedding(updated_chunk.embedding)

            if existing.embedding != updated_chunk.embedding:
                vector = np.asarray(updated_chunk.embedding, dtype=np.float32)
                self._vector_index[document.library_id].update_vector(chunk_id, vector)

            updated_chunk.update_timestamp()
            self._chunks[chunk_id] = updated_chunk
            return updated_chunk

    def delete_chunk(self, chunk_id: UUID) -> bool:
        with self._lock:
            if chunk_id not in self._chunks:
                return False

            chunk = self._chunks[chunk_id]
            document = self._documents[chunk.document_id]
            library = self._libraries[document.library_id]

            self._vector_index[document.library_id].remove_vector(chunk_id)
            del self._chunks[chunk_id]
            self._chunks_by_document[chunk.document_id].discard(chunk_id)
            self._chunks_by_library[document.library_id].discard(chunk_id)

            document.decrement_chunk_count()
            library.remove_chunks(1)
            return True

    # Search operations
    def search(
        self,
        library_id: UUID,
        query_vector: List[float],
        k: int = 10,
        metadata_filters: Optional[Metadata] = None,
    ) -> List[DistanceResult]:
        with self._lock:
            if library_id not in self._libraries:
                raise ValueError(f"Library {library_id} not found")
            if k <= 0:
                raise ValueError("k must be positive")

            library = self._libraries[library_id]
            library.validate_chunk_embedding(query_vector)
            query = np.asarray(query_vector, dtype=np.float32)

            index = self._vector_index.get(library_id)
            if index is None:
                return []
            allowed_ids: Optional[Set[UUID]] = None
            if metadata_filters:
                allowed_ids = {
                    chunk_id
                    for chunk_id in self._chunks_by_library.get(library_id, set())
                    if self._metadata_matches(self._chunks[chunk_id], metadata_filters)
                }
                if not allowed_ids:
                    return []
            return index.search(query, k, library.distance_metric, allowed_ids=allowed_ids)

    def _metadata_matches(self, chunk: Chunk, filters: Metadata) -> bool:
        for key, expected in filters.items():
            if chunk.metadata.get(key) != expected:
                return False
        return True

    def _build_index(self, library: Library) -> VectorIndex:
        if library.index_kind == "flat":
            return FlatIndex(dimension=library.embedding_dimension)
        if library.index_kind == "random_projection":
            return RandomProjectionIndex(dimension=library.embedding_dimension)
        raise ValueError(f"Unsupported index kind: {library.index_kind}")
