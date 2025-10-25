from __future__ import annotations

from datetime import timezone
from typing import List
from uuid import UUID, uuid4

import pytest

from vector_db.entities import BaseEntity, Chunk, Document, Library


def test_base_entity_initializes_with_uuid_and_timestamps() -> None:
    entity = BaseEntity()

    assert isinstance(entity.id, UUID)
    assert entity.created_at.tzinfo == timezone.utc
    assert entity.updated_at.tzinfo == timezone.utc
    assert entity.updated_at >= entity.created_at


def test_base_entity_update_timestamp_moves_forward() -> None:
    entity = BaseEntity()
    initial_updated = entity.updated_at

    entity.update_timestamp()

    assert entity.updated_at > initial_updated


def make_chunk(embedding: List[float]) -> Chunk:
    return Chunk(
        document_id=uuid4(),
        text="Vector databases enable semantic search.",
        embedding=embedding,
        metadata={"page": 1},
        chunk_index=0,
    )


def test_chunk_rejects_empty_or_non_numeric_embedding() -> None:
    with pytest.raises(ValueError):
        make_chunk([])

    with pytest.raises(ValueError):
        make_chunk([0.1, "oops"])  # type: ignore[list-item]


def test_chunk_reports_embedding_dimension() -> None:
    chunk = make_chunk([0.1, 0.2, 0.3])

    assert chunk.embedding_dimension == 3


def test_document_chunk_count_helpers() -> None:
    document = Document(library_id=uuid4(), name="Vector Overview")
    initial_updated = document.updated_at

    document.increment_chunk_count()
    assert document.chunk_count == 1
    assert document.updated_at > initial_updated

    document.decrement_chunk_count()
    assert document.chunk_count == 0

    document.decrement_chunk_count()
    assert document.chunk_count == 0


def test_library_validates_embedding_dimension_and_counters() -> None:
    library = Library(name="Docs", embedding_dimension=3)
    assert library.distance_metric == "cosine"
    assert library.document_count == 0
    assert library.chunk_count == 0

    library.validate_chunk_embedding([0.0, 0.1, 0.2])

    with pytest.raises(ValueError):
        library.validate_chunk_embedding([0.0, 0.1])

    library.increment_document_count()
    assert library.document_count == 1

    library.decrement_document_count()
    assert library.document_count == 0

    library.add_chunks(5)
    assert library.chunk_count == 5

    library.remove_chunks(10)
    assert library.chunk_count == 0

