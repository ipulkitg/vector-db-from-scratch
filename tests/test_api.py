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
    chunk_ids = {chunk["id"] for chunk in list_chunks_resp.json()}
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
