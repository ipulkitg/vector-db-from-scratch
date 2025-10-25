from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from vector_db.indexes import FlatIndex, RandomProjectionIndex


def test_flat_index_validates_dimension() -> None:
    with pytest.raises(ValueError):
        FlatIndex(0)

    index = FlatIndex(3)
    chunk_id = uuid4()
    with pytest.raises(ValueError):
        index.add_vector(chunk_id, np.array([1.0, 0.0], dtype=np.float32))


def test_flat_index_search_orders_by_distance() -> None:
    index = FlatIndex(3)
    chunk_a = uuid4()
    chunk_b = uuid4()
    index.add_vector(chunk_a, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    index.add_vector(chunk_b, np.array([0.0, 1.0, 0.0], dtype=np.float32))

    results = index.search(np.array([0.9, 0.1, 0.0], dtype=np.float32), k=2, metric="cosine")
    assert [chunk for chunk, _ in results] == [chunk_a, chunk_b]


def test_flat_index_supports_dot_and_euclidean_metrics() -> None:
    index = FlatIndex(2)
    chunk_a = uuid4()
    chunk_b = uuid4()
    index.add_vector(chunk_a, np.array([1.0, 0.0], dtype=np.float32))
    index.add_vector(chunk_b, np.array([0.0, 1.0], dtype=np.float32))

    dot_results = index.search(np.array([1.0, 0.0], dtype=np.float32), k=1, metric="dot_product")
    assert dot_results[0][0] == chunk_a

    euclidean_results = index.search(
        np.array([0.0, 1.0], dtype=np.float32), k=1, metric="euclidean"
    )
    assert euclidean_results[0][0] == chunk_b


def test_flat_index_respects_allowed_ids() -> None:
    index = FlatIndex(2)
    chunk_a = uuid4()
    chunk_b = uuid4()
    index.add_vector(chunk_a, np.array([1.0, 0.0], dtype=np.float32))
    index.add_vector(chunk_b, np.array([0.0, 1.0], dtype=np.float32))

    results = index.search(
        np.array([1.0, 0.0], dtype=np.float32),
        k=2,
        metric="cosine",
        allowed_ids={chunk_b},
    )
    assert [chunk for chunk, _ in results] == [chunk_b]


def test_random_projection_index_searches_primary_bucket() -> None:
    index = RandomProjectionIndex(dimension=3, num_projections=4, random_state=123)
    chunk_a = uuid4()
    chunk_b = uuid4()
    index.add_vector(chunk_a, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    index.add_vector(chunk_b, np.array([0.0, 1.0, 0.0], dtype=np.float32))

    results = index.search(np.array([0.9, 0.1, 0.0], dtype=np.float32), k=1, metric="cosine")
    assert results
    assert results[0][0] == chunk_a


def test_random_projection_index_respects_allowed_ids() -> None:
    index = RandomProjectionIndex(dimension=3, num_projections=4, random_state=456)
    chunk_a = uuid4()
    chunk_b = uuid4()
    index.add_vector(chunk_a, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    index.add_vector(chunk_b, np.array([0.0, 1.0, 0.0], dtype=np.float32))

    results = index.search(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        k=2,
        metric="cosine",
        allowed_ids={chunk_b},
    )
    assert results
    assert results[0][0] == chunk_b
