from __future__ import annotations

from fastapi.testclient import TestClient

from vector_db.api import create_app


def client() -> TestClient:
    return TestClient(create_app())


def test_full_library_document_chunk_flow() -> None:
    test_client = client()
    library_payload = {
        "name": "API Library",
        "embedding_dimension": 3,
        "metadata": {},
    }
    library_resp = test_client.post("/libraries", json=library_payload)
    assert library_resp.status_code == 201
    library_id = library_resp.json()["id"]

    document_resp = test_client.post(
        "/documents", json={"library_id": library_id, "name": "Doc via API"}
    )
    assert document_resp.status_code == 201
    document_id = document_resp.json()["id"]

    chunk_resp = test_client.post(
        "/chunks",
        json={
            "document_id": document_id,
            "text": "Chunk body",
            "embedding": [1.0, 0.0, 0.0],
            "metadata": {"category": "alpha"},
            "chunk_index": 0,
        },
    )
    assert chunk_resp.status_code == 201
    chunk_id = chunk_resp.json()["id"]

    list_chunks_resp = test_client.get(f"/documents/{document_id}/chunks")
    assert list_chunks_resp.status_code == 200
    paginated_data = list_chunks_resp.json()
    chunk_ids = {chunk["id"] for chunk in paginated_data["items"]}
    assert chunk_id in chunk_ids

    search_resp = test_client.post(
        f"/libraries/{library_id}/search",
        json={"query_vector": [1.0, 0.0, 0.0], "k": 1},
    )
    assert search_resp.status_code == 200
    assert search_resp.json()[0]["chunk_id"] == chunk_id

    filtered_resp = test_client.post(
        f"/libraries/{library_id}/search",
        json={
            "query_vector": [1.0, 0.0, 0.0],
            "k": 1,
            "metadata_filters": {"category": "alpha"},
        },
    )
    assert filtered_resp.status_code == 200
    assert filtered_resp.json()[0]["chunk_id"] == chunk_id

    empty_resp = test_client.post(
        f"/libraries/{library_id}/search",
        json={
            "query_vector": [1.0, 0.0, 0.0],
            "k": 1,
            "metadata_filters": {"category": "beta"},
        },
    )
    assert empty_resp.status_code == 200
    assert empty_resp.json() == []


def test_document_creation_requires_existing_library() -> None:
    test_client = client()
    response = test_client.post(
        "/documents", json={"library_id": "00000000-0000-0000-0000-000000000000", "name": "Doc"}
    )
    assert response.status_code == 404


def test_library_creation_accepts_index_kind() -> None:
    test_client = client()
    response = test_client.post(
        "/libraries",
        json={
            "name": "Approx",
            "embedding_dimension": 3,
            "metadata": {},
            "index_kind": "random_projection",
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["index_kind"] == "random_projection"


def test_pagination_on_libraries() -> None:
    """Test pagination parameters work on library list endpoint."""
    test_client = client()
    # Create 5 libraries
    for i in range(5):
        test_client.post(
            "/libraries",
            json={"name": f"Library {i}", "embedding_dimension": 3},
        )

    # Get first page (2 items)
    resp = test_client.get("/libraries?skip=0&limit=2")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 5
    assert len(data["items"]) == 2
    assert data["skip"] == 0
    assert data["limit"] == 2
    assert data["has_more"] is True

    # Get last page
    resp = test_client.get("/libraries?skip=4&limit=2")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 5
    assert len(data["items"]) == 1
    assert data["has_more"] is False


def test_batch_chunk_creation() -> None:
    """Test creating multiple chunks in a single batch request."""
    test_client = client()

    # Create library and document
    library_resp = test_client.post(
        "/libraries", json={"name": "Batch Test", "embedding_dimension": 3}
    )
    library_id = library_resp.json()["id"]

    document_resp = test_client.post(
        "/documents", json={"library_id": library_id, "name": "Batch Doc"}
    )
    document_id = document_resp.json()["id"]

    # Batch create 3 chunks
    batch_payload = {
        "document_id": document_id,
        "chunks": [
            {
                "document_id": document_id,
                "text": f"Chunk {i}",
                "embedding": [float(i), 0.0, 0.0],
                "metadata": {"index": i},
                "chunk_index": i,
            }
            for i in range(3)
        ],
    }

    batch_resp = test_client.post("/chunks/batch", json=batch_payload)
    assert batch_resp.status_code == 201
    data = batch_resp.json()
    assert data["created_count"] == 3
    assert len(data["chunk_ids"]) == 3

    # Verify chunks were created
    list_resp = test_client.get(f"/documents/{document_id}/chunks")
    assert list_resp.status_code == 200
    assert len(list_resp.json()["items"]) == 3
