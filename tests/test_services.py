from __future__ import annotations

import pytest

from uuid import uuid4

from vector_db.repositories import (
    VectorStoreChunkRepository,
    VectorStoreDocumentRepository,
    VectorStoreLibraryRepository,
)
from vector_db.schemas import (
    ChunkCreate,
    ChunkUpdate,
    DocumentCreate,
    DocumentUpdate,
    LibraryCreate,
    LibraryUpdate,
)
from vector_db.services import ChunkService, DocumentService, LibraryService
from vector_db.vector_store import VectorStore


@pytest.fixture()
def store() -> VectorStore:
    return VectorStore()


@pytest.fixture()
def repositories(store: VectorStore):
    return {
        "libraries": VectorStoreLibraryRepository(store),
        "documents": VectorStoreDocumentRepository(store),
        "chunks": VectorStoreChunkRepository(store),
    }


def test_library_service_crud_flow(repositories) -> None:
    service = LibraryService(repositories["libraries"])
    create_payload = LibraryCreate(name="Service Library", embedding_dimension=3)

    created = service.create_library(create_payload)
    assert service.get_library(created.id) is not None
    assert created.id in {lib.id for lib in service.list_libraries()}

    update_payload = LibraryUpdate(description="Updated desc")
    updated = service.update_library(created.id, update_payload)
    assert updated.description == "Updated desc"  # type: ignore[union-attr]

    assert service.delete_library(created.id) is True
    assert service.get_library(created.id) is None


def test_document_service_requires_existing_library(repositories) -> None:
    service = DocumentService(repositories["documents"], repositories["libraries"])
    payload = DocumentCreate(library_id=uuid4(), name="Doc")

    with pytest.raises(ValueError, match="Library"):
        service.create_document(payload)


def test_document_service_crud_flow(repositories) -> None:
    library_service = LibraryService(repositories["libraries"])
    document_service = DocumentService(repositories["documents"], repositories["libraries"])

    library = library_service.create_library(LibraryCreate(name="Docs Library", embedding_dimension=3))

    document = document_service.create_document(
        DocumentCreate(library_id=library.id, name="Doc 1")
    )
    assert document_service.get_document(document.id) is not None

    document_service.update_document(document.id, DocumentUpdate(name="Doc 1 Updated"))
    assert document_service.get_document(document.id).name == "Doc 1 Updated"

    assert document_service.delete_document(document.id)
    assert document_service.get_document(document.id) is None


def test_chunk_service_search_flow(repositories) -> None:
    library_service = LibraryService(repositories["libraries"])
    document_service = DocumentService(repositories["documents"], repositories["libraries"])
    chunk_service = ChunkService(
        repositories["chunks"], repositories["documents"], repositories["libraries"]
    )

    library = library_service.create_library(LibraryCreate(name="Chunk Library", embedding_dimension=3))
    document = document_service.create_document(
        DocumentCreate(library_id=library.id, name="Chunk Doc")
    )
    chunk = chunk_service.create_chunk(
        ChunkCreate(
            document_id=document.id,
            text="Chunk data",
            embedding=[1.0, 0.0, 0.0],
            metadata={},
            chunk_index=0,
        )
    )
    chunks_for_document = chunk_service.list_chunks(document_id=document.id)
    assert chunk.id in {c.id for c in chunks_for_document}

    results = chunk_service.search_chunks(library.id, [1.0, 0.0, 0.0], k=1)
    assert results[0][0] == chunk.id


def test_chunk_service_search_with_metadata_filters(repositories) -> None:
    library_service = LibraryService(repositories["libraries"])
    document_service = DocumentService(repositories["documents"], repositories["libraries"])
    chunk_service = ChunkService(
        repositories["chunks"], repositories["documents"], repositories["libraries"]
    )

    library = library_service.create_library(LibraryCreate(name="Filtered", embedding_dimension=3))
    document = document_service.create_document(DocumentCreate(library_id=library.id, name="Doc"))
    alpha_chunk = chunk_service.create_chunk(
        ChunkCreate(
            document_id=document.id,
            text="Alpha",
            embedding=[1.0, 0.0, 0.0],
            metadata={"category": "alpha"},
            chunk_index=0,
        )
    )
    chunk_service.create_chunk(
        ChunkCreate(
            document_id=document.id,
            text="Beta",
            embedding=[0.0, 1.0, 0.0],
            metadata={"category": "beta"},
            chunk_index=1,
        )
    )

    filtered_results = chunk_service.search_chunks(
        library.id,
        [1.0, 0.0, 0.0],
        k=2,
        metadata_filters={"category": "alpha"},
    )
    assert [chunk_id for chunk_id, _ in filtered_results] == [alpha_chunk.id]
