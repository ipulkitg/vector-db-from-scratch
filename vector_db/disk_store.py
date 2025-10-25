"""Disk-based persistent storage for vector database entities."""
from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional, Set, Union
from uuid import UUID

import numpy as np

from .entities import Chunk, Document, Library
from .indexes import DistanceResult, FlatIndex, RandomProjectionIndex

VectorIndexType = Union[FlatIndex, RandomProjectionIndex]


class DiskVectorStore:
    """
    Thread-safe disk-backed store for libraries, documents, and chunks.
    All entities are persisted to JSON files, vector indexes to binary files.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self._ensure_directories()

        # In-memory cache (lazy loaded)
        self._libraries: Dict[UUID, Library] = {}
        self._documents: Dict[UUID, Document] = {}
        self._chunks: Dict[UUID, Chunk] = {}
        self._vector_index: Dict[UUID, VectorIndexType] = {}

        self._lock = RLock()
        self._loaded = False

    def _ensure_directories(self) -> None:
        """Create necessary directory structure."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "libraries").mkdir(exist_ok=True)
        (self.data_dir / "documents").mkdir(exist_ok=True)
        (self.data_dir / "chunks").mkdir(exist_ok=True)
        (self.data_dir / "indexes").mkdir(exist_ok=True)

    def _load_all(self) -> None:
        """Load all entities from disk into memory (lazy initialization)."""
        if self._loaded:
            return

        with self._lock:
            if self._loaded:  # Double-check after acquiring lock
                return

            # Load libraries
            library_dir = self.data_dir / "libraries"
            if library_dir.exists():
                for lib_file in library_dir.glob("*.json"):
                    with open(lib_file, "r") as f:
                        data = json.load(f)
                        library = Library(**data)
                        self._libraries[library.id] = library

                        # Load corresponding index
                        self._load_index(library.id, library.index_kind, library.embedding_dimension)

            # Load documents
            doc_dir = self.data_dir / "documents"
            if doc_dir.exists():
                for doc_file in doc_dir.glob("*.json"):
                    with open(doc_file, "r") as f:
                        data = json.load(f)
                        document = Document(**data)
                        self._documents[document.id] = document

            # Load chunks
            chunk_dir = self.data_dir / "chunks"
            if chunk_dir.exists():
                for chunk_file in chunk_dir.glob("*.json"):
                    with open(chunk_file, "r") as f:
                        data = json.load(f)
                        chunk = Chunk(**data)
                        self._chunks[chunk.id] = chunk

            self._loaded = True

    def _load_index(self, library_id: UUID, index_kind: str, dimension: int) -> None:
        """Load a vector index from disk."""
        index_path = self.data_dir / "indexes" / str(library_id)

        if index_kind == "flat":
            if index_path.with_suffix(".json").exists():
                self._vector_index[library_id] = FlatIndex.load(index_path)
            else:
                self._vector_index[library_id] = FlatIndex(dimension=dimension)
        elif index_kind == "random_projection":
            if index_path.with_suffix(".json").exists():
                self._vector_index[library_id] = RandomProjectionIndex.load(index_path)
            else:
                self._vector_index[library_id] = RandomProjectionIndex(dimension=dimension)
        else:
            raise ValueError(f"Unknown index kind: {index_kind}")

    def _save_library(self, library: Library) -> None:
        """Persist a library to disk."""
        path = self.data_dir / "libraries" / f"{library.id}.json"
        with open(path, "w") as f:
            json.dump(library.model_dump(mode="json"), f, indent=2)

    def _save_document(self, document: Document) -> None:
        """Persist a document to disk."""
        path = self.data_dir / "documents" / f"{document.id}.json"
        with open(path, "w") as f:
            json.dump(document.model_dump(mode="json"), f, indent=2)

    def _save_chunk(self, chunk: Chunk) -> None:
        """Persist a chunk to disk."""
        path = self.data_dir / "chunks" / f"{chunk.id}.json"
        with open(path, "w") as f:
            json.dump(chunk.model_dump(mode="json"), f, indent=2)

    def _save_index(self, library_id: UUID) -> None:
        """Persist a vector index to disk."""
        if library_id in self._vector_index:
            index_path = self.data_dir / "indexes" / str(library_id)
            self._vector_index[library_id].save(index_path)

    def _delete_library_file(self, library_id: UUID) -> None:
        """Delete library file from disk."""
        path = self.data_dir / "libraries" / f"{library_id}.json"
        path.unlink(missing_ok=True)

        # Delete index files
        index_path = self.data_dir / "indexes" / str(library_id)
        index_path.with_suffix(".json").unlink(missing_ok=True)
        index_path.with_suffix(".npy").unlink(missing_ok=True)
        index_path.with_suffix(".projections.npy").unlink(missing_ok=True)

    def _delete_document_file(self, document_id: UUID) -> None:
        """Delete document file from disk."""
        path = self.data_dir / "documents" / f"{document_id}.json"
        path.unlink(missing_ok=True)

    def _delete_chunk_file(self, chunk_id: UUID) -> None:
        """Delete chunk file from disk."""
        path = self.data_dir / "chunks" / f"{chunk_id}.json"
        path.unlink(missing_ok=True)

    # Library operations
    def add_library(self, library: Library) -> Library:
        with self._lock:
            self._load_all()

            if library.id in self._libraries:
                raise ValueError(f"Library {library.id} already exists")

            self._libraries[library.id] = library

            # Create appropriate index
            if library.index_kind == "flat":
                self._vector_index[library.id] = FlatIndex(dimension=library.embedding_dimension)
            elif library.index_kind == "random_projection":
                self._vector_index[library.id] = RandomProjectionIndex(
                    dimension=library.embedding_dimension
                )
            else:
                raise ValueError(f"Unknown index kind: {library.index_kind}")

            self._save_library(library)
            self._save_index(library.id)
            return library

    def get_library(self, library_id: UUID) -> Optional[Library]:
        with self._lock:
            self._load_all()
            return self._libraries.get(library_id)

    def list_libraries(self) -> List[Library]:
        with self._lock:
            self._load_all()
            return list(self._libraries.values())

    def update_library(self, library_id: UUID, updated_library: Library) -> Library:
        with self._lock:
            self._load_all()

            if library_id not in self._libraries:
                raise ValueError(f"Library {library_id} not found")
            if updated_library.id != library_id:
                raise ValueError("Library ID cannot change")

            existing = self._libraries[library_id]

            # Dimension change validation
            if (
                existing.chunk_count > 0
                and existing.embedding_dimension != updated_library.embedding_dimension
            ):
                raise ValueError("Cannot change embedding dimension with existing chunks")

            # Recreate index if dimension or index_kind changed
            if (
                existing.embedding_dimension != updated_library.embedding_dimension
                or existing.index_kind != updated_library.index_kind
            ):
                if updated_library.index_kind == "flat":
                    self._vector_index[library_id] = FlatIndex(
                        dimension=updated_library.embedding_dimension
                    )
                elif updated_library.index_kind == "random_projection":
                    self._vector_index[library_id] = RandomProjectionIndex(
                        dimension=updated_library.embedding_dimension
                    )

            updated_library.update_timestamp()
            self._libraries[library_id] = updated_library
            self._save_library(updated_library)
            self._save_index(library_id)
            return updated_library

    def delete_library(self, library_id: UUID) -> bool:
        with self._lock:
            self._load_all()

            if library_id not in self._libraries:
                return False

            # Cascade delete documents and chunks
            documents_to_delete = [
                doc_id for doc_id, doc in self._documents.items()
                if doc.library_id == library_id
            ]
            for document_id in documents_to_delete:
                self.delete_document(document_id)

            # Delete library
            del self._libraries[library_id]
            self._vector_index.pop(library_id, None)
            self._delete_library_file(library_id)
            return True

    # Document operations
    def add_document(self, document: Document) -> Document:
        with self._lock:
            self._load_all()

            if document.library_id not in self._libraries:
                raise ValueError(f"Library {document.library_id} not found")
            if document.id in self._documents:
                raise ValueError(f"Document {document.id} already exists")

            self._documents[document.id] = document

            library = self._libraries[document.library_id]
            library.increment_document_count()
            self._save_library(library)
            self._save_document(document)
            return document

    def get_document(self, document_id: UUID) -> Optional[Document]:
        with self._lock:
            self._load_all()
            return self._documents.get(document_id)

    def list_documents(self, library_id: Optional[UUID] = None) -> List[Document]:
        with self._lock:
            self._load_all()
            if library_id is not None:
                return [doc for doc in self._documents.values() if doc.library_id == library_id]
            return list(self._documents.values())

    def update_document(self, document_id: UUID, updated_document: Document) -> Document:
        with self._lock:
            self._load_all()

            if document_id not in self._documents:
                raise ValueError(f"Document {document_id} not found")
            if updated_document.id != document_id:
                raise ValueError("Document ID cannot change")

            existing = self._documents[document_id]
            if updated_document.library_id != existing.library_id:
                raise ValueError("Document library cannot change")

            updated_document.update_timestamp()
            self._documents[document_id] = updated_document
            self._save_document(updated_document)
            return updated_document

    def delete_document(self, document_id: UUID) -> bool:
        with self._lock:
            self._load_all()

            if document_id not in self._documents:
                return False

            document = self._documents[document_id]

            # Cascade delete chunks
            chunks_to_delete = [
                chunk_id for chunk_id, chunk in self._chunks.items()
                if chunk.document_id == document_id
            ]
            for chunk_id in chunks_to_delete:
                self.delete_chunk(chunk_id)

            # Delete document
            del self._documents[document_id]

            library = self._libraries[document.library_id]
            library.decrement_document_count()
            self._save_library(library)
            self._delete_document_file(document_id)
            return True

    # Chunk operations
    def add_chunk(self, chunk: Chunk) -> Chunk:
        with self._lock:
            self._load_all()

            if chunk.document_id not in self._documents:
                raise ValueError(f"Document {chunk.document_id} not found")
            if chunk.id in self._chunks:
                raise ValueError(f"Chunk {chunk.id} already exists")

            document = self._documents[chunk.document_id]
            library = self._libraries[document.library_id]
            library.validate_chunk_embedding(chunk.embedding)

            self._chunks[chunk.id] = chunk

            # Add to vector index
            vector = np.asarray(chunk.embedding, dtype=np.float32)
            self._vector_index[document.library_id].add_vector(chunk.id, vector)

            # Update counts
            document.increment_chunk_count()
            library.add_chunks(1)

            self._save_chunk(chunk)
            self._save_document(document)
            self._save_library(library)
            self._save_index(document.library_id)
            return chunk

    def get_chunk(self, chunk_id: UUID) -> Optional[Chunk]:
        with self._lock:
            self._load_all()
            return self._chunks.get(chunk_id)

    def list_chunks(
        self, document_id: Optional[UUID] = None, library_id: Optional[UUID] = None
    ) -> List[Chunk]:
        with self._lock:
            self._load_all()
            if document_id is not None:
                return [c for c in self._chunks.values() if c.document_id == document_id]
            if library_id is not None:
                doc_ids = {
                    doc.id for doc in self._documents.values() if doc.library_id == library_id
                }
                return [c for c in self._chunks.values() if c.document_id in doc_ids]
            return list(self._chunks.values())

    def update_chunk(self, chunk_id: UUID, updated_chunk: Chunk) -> Chunk:
        with self._lock:
            self._load_all()

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

            # Update vector index if embedding changed
            if existing.embedding != updated_chunk.embedding:
                vector = np.asarray(updated_chunk.embedding, dtype=np.float32)
                self._vector_index[document.library_id].update_vector(chunk_id, vector)
                self._save_index(document.library_id)

            updated_chunk.update_timestamp()
            self._chunks[chunk_id] = updated_chunk
            self._save_chunk(updated_chunk)
            return updated_chunk

    def delete_chunk(self, chunk_id: UUID) -> bool:
        with self._lock:
            self._load_all()

            if chunk_id not in self._chunks:
                return False

            chunk = self._chunks[chunk_id]
            document = self._documents[chunk.document_id]
            library = self._libraries[document.library_id]

            # Remove from vector index
            self._vector_index[document.library_id].remove_vector(chunk_id)

            # Delete chunk
            del self._chunks[chunk_id]

            # Update counts
            document.decrement_chunk_count()
            library.remove_chunks(1)

            self._delete_chunk_file(chunk_id)
            self._save_document(document)
            self._save_library(library)
            self._save_index(document.library_id)
            return True

    # Search operations
    def search(self, library_id: UUID, query_vector: List[float], k: int = 10) -> List[DistanceResult]:
        with self._lock:
            self._load_all()

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
            return index.search(query, k, library.distance_metric)


__all__ = ["DiskVectorStore"]
