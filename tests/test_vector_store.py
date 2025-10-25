from __future__ import annotations

from typing import List, Optional

import pytest

from vector_db.entities import Chunk, Document, Library
from vector_db.vector_store import VectorStore


@pytest.fixture()
def store() -> VectorStore:
    return VectorStore()


@pytest.fixture()
def library(store: VectorStore) -> Library:
    library = Library(name="Test Library", embedding_dimension=3)
    store.add_library(library)
    return library


@pytest.fixture()
def document(store: VectorStore, library: Library) -> Document:
    document = Document(library_id=library.id, name="Doc 1")
    store.add_document(document)
    return document


def make_chunk(
    document: Document,
    embedding: List[float],
    index: int = 0,
    metadata: Optional[dict] = None,
) -> Chunk:
    payload = metadata or {"order": index}
    return Chunk(
        document_id=document.id,
        text=f"Chunk {index}",
        embedding=embedding,
        metadata=payload,
        chunk_index=index,
    )


def test_add_and_get_library(store: VectorStore) -> None:
    library = Library(name="Library A", embedding_dimension=4)
    result = store.add_library(library)

    assert result.id == library.id
    assert store.get_library(library.id) is library
    assert library.id in {lib.id for lib in store.list_libraries()}


def test_add_library_twice_raises(store: VectorStore, library: Library) -> None:
    with pytest.raises(ValueError):
        store.add_library(library)


def test_update_library_prevents_dimension_change_with_chunks(
    store: VectorStore, library: Library, document: Document
) -> None:
    chunk = make_chunk(document, [0.1, 0.1, 0.1])
    store.add_chunk(chunk)

    updated = library.model_copy(update={"embedding_dimension": 5})
    with pytest.raises(ValueError):
        store.update_library(library.id, updated)


def test_update_library_without_chunks_allows_changes(store: VectorStore, library: Library) -> None:
    updated = library.model_copy(update={"description": "New description"})
    store.update_library(library.id, updated)

    assert store.get_library(library.id).description == "New description"  # type: ignore[union-attr]


def test_update_library_prevents_index_change_with_chunks(
    store: VectorStore, library: Library, document: Document
) -> None:
    chunk = make_chunk(document, [0.1, 0.1, 0.1])
    store.add_chunk(chunk)

    updated = library.model_copy(update={"index_kind": "random_projection"})
    with pytest.raises(ValueError):
        store.update_library(library.id, updated)


def test_add_document_updates_library_count(store: VectorStore, library: Library) -> None:
    document = Document(library_id=library.id, name="Doc")
    store.add_document(document)

    assert store.get_document(document.id) is document
    assert library.document_count == 1
    assert document.id in {doc.id for doc in store.list_documents(library.id)}


def test_delete_document_cascades_chunks(store: VectorStore, library: Library, document: Document) -> None:
    chunk = make_chunk(document, [0.1, 0.2, 0.3])
    store.add_chunk(chunk)

    assert store.delete_document(document.id)
    assert store.get_chunk(chunk.id) is None
    assert library.document_count == 0
    assert library.chunk_count == 0


def test_add_chunk_updates_counts_and_indexes(
    store: VectorStore, library: Library, document: Document
) -> None:
    chunk = make_chunk(document, [0.2, 0.2, 0.2])
    store.add_chunk(chunk)

    assert store.get_chunk(chunk.id) is chunk
    assert document.chunk_count == 1
    assert library.chunk_count == 1
    assert chunk.id in {c.id for c in store.list_chunks(document_id=document.id)}
    assert chunk.id in {c.id for c in store.list_chunks(library_id=library.id)}


def test_add_chunk_with_wrong_dimension_fails(store: VectorStore, document: Document) -> None:
    chunk = make_chunk(document, [0.1, 0.2], index=1)

    with pytest.raises(ValueError):
        store.add_chunk(chunk)


def test_update_chunk_updates_vector_index(store: VectorStore, document: Document, library: Library) -> None:
    chunk = make_chunk(document, [0.1, 0.2, 0.3])
    store.add_chunk(chunk)

    updated = chunk.model_copy(update={"embedding": [1.0, 1.0, 1.0]})
    store.update_chunk(chunk.id, updated)

    # New embedding should dominate the search
    results = store.search(library.id, [1.0, 1.0, 1.0], k=1)
    assert results[0][0] == chunk.id


def test_delete_chunk_updates_counts(store: VectorStore, document: Document, library: Library) -> None:
    chunk = make_chunk(document, [0.2, 0.3, 0.4])
    store.add_chunk(chunk)

    assert store.delete_chunk(chunk.id)
    assert store.get_chunk(chunk.id) is None
    assert document.chunk_count == 0
    assert library.chunk_count == 0


def test_search_returns_closest_chunks(store: VectorStore, document: Document, library: Library) -> None:
    chunk_a = make_chunk(document, [1.0, 0.0, 0.0], index=0)
    chunk_b = make_chunk(document, [0.0, 1.0, 0.0], index=1)
    store.add_chunk(chunk_a)
    store.add_chunk(chunk_b)

    results = store.search(library.id, [0.9, 0.1, 0.0], k=2)
    ordered_chunk_ids = [chunk_id for chunk_id, _ in results]
    assert ordered_chunk_ids[0] == chunk_a.id
    assert ordered_chunk_ids[1] == chunk_b.id


def test_search_with_metadata_filters_returns_subset(
    store: VectorStore, document: Document, library: Library
) -> None:
    chunk_a = make_chunk(document, [1.0, 0.0, 0.0], index=0, metadata={"tag": "alpha"})
    chunk_b = make_chunk(document, [0.0, 1.0, 0.0], index=1, metadata={"tag": "beta"})
    store.add_chunk(chunk_a)
    store.add_chunk(chunk_b)

    results = store.search(
        library.id,
        [1.0, 0.0, 0.0],
        k=2,
        metadata_filters={"tag": "alpha"},
    )
    assert [chunk_id for chunk_id, _ in results] == [chunk_a.id]


def test_search_with_metadata_filters_handles_no_matches(
    store: VectorStore, document: Document, library: Library
) -> None:
    chunk = make_chunk(document, [1.0, 0.0, 0.0], index=0, metadata={"tag": "alpha"})
    store.add_chunk(chunk)

    results = store.search(
        library.id,
        [1.0, 0.0, 0.0],
        k=1,
        metadata_filters={"tag": "beta"},
    )
    assert results == []


def test_delete_library_cascades_all_children(store: VectorStore) -> None:
    library = Library(name="Cascade", embedding_dimension=3)
    store.add_library(library)
    document = Document(library_id=library.id, name="Doc")
    store.add_document(document)
    chunk = make_chunk(document, [0.3, 0.3, 0.3])
    store.add_chunk(chunk)

    assert store.delete_library(library.id)
    assert store.list_libraries() == []
    assert store.list_documents() == []
    assert store.list_chunks() == []


def test_library_with_random_projection_index_supports_search(store: VectorStore) -> None:
    library = Library(
        name="Approx",
        embedding_dimension=3,
        index_kind="random_projection",
    )
    store.add_library(library)
    document = Document(library_id=library.id, name="Doc")
    store.add_document(document)
    chunk = make_chunk(document, [1.0, 0.0, 0.0])
    store.add_chunk(chunk)

    results = store.search(library.id, [1.0, 0.0, 0.0], k=1)
    assert results[0][0] == chunk.id
