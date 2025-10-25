"""Tests for disk persistence functionality."""
from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest

from vector_db.config import Settings
from vector_db.disk_store import DiskVectorStore
from vector_db.entities import Chunk, Document, Library
from vector_db.indexes import FlatIndex, RandomProjectionIndex


@pytest.fixture
def temp_dir():
    """Create a temporary directory for disk storage tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def disk_store(temp_dir):
    """Create a DiskVectorStore instance."""
    return DiskVectorStore(temp_dir)


class TestFlatIndexPersistence:
    """Tests for FlatIndex save/load functionality."""

    def test_save_and_load_empty_index(self, temp_dir):
        """Test saving and loading an empty index."""
        index = FlatIndex(dimension=3)
        index_path = temp_dir / "test_index"

        # Save
        index.save(index_path)

        # Load
        loaded_index = FlatIndex.load(index_path)

        assert loaded_index.dimension == 3
        assert len(loaded_index._vectors) == 0

    def test_save_and_load_with_vectors(self, temp_dir):
        """Test saving and loading an index with vectors."""
        index = FlatIndex(dimension=3)
        vec1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        vec2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        id1 = UUID("00000000-0000-0000-0000-000000000001")
        id2 = UUID("00000000-0000-0000-0000-000000000002")

        index.add_vector(id1, vec1)
        index.add_vector(id2, vec2)

        index_path = temp_dir / "test_index"
        index.save(index_path)

        # Load
        loaded_index = FlatIndex.load(index_path)

        assert loaded_index.dimension == 3
        assert len(loaded_index._vectors) == 2
        assert id1 in loaded_index._vectors
        assert id2 in loaded_index._vectors
        np.testing.assert_array_almost_equal(loaded_index._vectors[id1], vec1)
        np.testing.assert_array_almost_equal(loaded_index._vectors[id2], vec2)

    def test_save_creates_json_and_npy_files(self, temp_dir):
        """Test that save creates both .json and .npy files."""
        index = FlatIndex(dimension=3)
        vec1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        id1 = UUID("00000000-0000-0000-0000-000000000001")
        index.add_vector(id1, vec1)

        index_path = temp_dir / "test_index"
        index.save(index_path)

        assert (temp_dir / "test_index.json").exists()
        assert (temp_dir / "test_index.npy").exists()


class TestRandomProjectionIndexPersistence:
    """Tests for RandomProjectionIndex save/load functionality."""

    def test_save_and_load_empty_index(self, temp_dir):
        """Test saving and loading an empty random projection index."""
        index = RandomProjectionIndex(dimension=3, num_projections=8, random_state=42)
        index_path = temp_dir / "test_rp_index"

        # Save
        index.save(index_path)

        # Load
        loaded_index = RandomProjectionIndex.load(index_path)

        assert loaded_index.dimension == 3
        assert loaded_index.num_projections == 8
        assert loaded_index.random_state == 42
        assert len(loaded_index._vectors) == 0
        assert len(loaded_index._buckets) == 0

        # Verify projection matrices match
        np.testing.assert_array_almost_equal(loaded_index._projections, index._projections)

    def test_save_and_load_with_vectors(self, temp_dir):
        """Test saving and loading random projection index with vectors."""
        index = RandomProjectionIndex(dimension=3, num_projections=4, random_state=42)
        vec1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        vec2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        id1 = UUID("00000000-0000-0000-0000-000000000001")
        id2 = UUID("00000000-0000-0000-0000-000000000002")

        index.add_vector(id1, vec1)
        index.add_vector(id2, vec2)

        index_path = temp_dir / "test_rp_index"
        index.save(index_path)

        # Load
        loaded_index = RandomProjectionIndex.load(index_path)

        assert loaded_index.dimension == 3
        assert len(loaded_index._vectors) == 2
        assert id1 in loaded_index._vectors
        assert id2 in loaded_index._vectors
        assert len(loaded_index._buckets) == len(index._buckets)


class TestDiskVectorStore:
    """Tests for DiskVectorStore CRUD operations with persistence."""

    def test_library_persistence(self, disk_store, temp_dir):
        """Test that libraries are persisted to disk."""
        library = Library(
            name="Test Library",
            description="Test Description",
            embedding_dimension=128,
            distance_metric="cosine",
        )

        # Add library
        added = disk_store.add_library(library)
        assert added.id == library.id

        # Verify file exists
        lib_file = temp_dir / "libraries" / f"{library.id}.json"
        assert lib_file.exists()

        # Create new store instance and verify persistence
        new_store = DiskVectorStore(temp_dir)
        loaded = new_store.get_library(library.id)

        assert loaded is not None
        assert loaded.id == library.id
        assert loaded.name == "Test Library"
        assert loaded.embedding_dimension == 128

    def test_document_persistence(self, disk_store, temp_dir):
        """Test that documents are persisted to disk."""
        library = Library(name="Library", embedding_dimension=128)
        disk_store.add_library(library)

        document = Document(library_id=library.id, name="Test Document")
        disk_store.add_document(document)

        # Verify file exists
        doc_file = temp_dir / "documents" / f"{document.id}.json"
        assert doc_file.exists()

        # Load from new store
        new_store = DiskVectorStore(temp_dir)
        loaded = new_store.get_document(document.id)

        assert loaded is not None
        assert loaded.id == document.id
        assert loaded.name == "Test Document"
        assert loaded.library_id == library.id

    def test_chunk_persistence_with_vectors(self, disk_store, temp_dir):
        """Test that chunks and their vectors are persisted to disk."""
        library = Library(name="Library", embedding_dimension=3)
        disk_store.add_library(library)

        document = Document(library_id=library.id, name="Document")
        disk_store.add_document(document)

        chunk = Chunk(
            document_id=document.id,
            text="Test chunk",
            embedding=[1.0, 2.0, 3.0],
            chunk_index=0,
        )
        disk_store.add_chunk(chunk)

        # Verify chunk file exists
        chunk_file = temp_dir / "chunks" / f"{chunk.id}.json"
        assert chunk_file.exists()

        # Verify index file exists
        index_file = temp_dir / "indexes" / f"{library.id}.json"
        assert index_file.exists()

        # Load from new store
        new_store = DiskVectorStore(temp_dir)
        loaded_chunk = new_store.get_chunk(chunk.id)

        assert loaded_chunk is not None
        assert loaded_chunk.id == chunk.id
        assert loaded_chunk.text == "Test chunk"
        assert loaded_chunk.embedding == [1.0, 2.0, 3.0]

    def test_cascade_delete_removes_files(self, disk_store, temp_dir):
        """Test that cascade deletion removes all related files."""
        library = Library(name="Library", embedding_dimension=3)
        disk_store.add_library(library)

        document = Document(library_id=library.id, name="Document")
        disk_store.add_document(document)

        chunk = Chunk(
            document_id=document.id,
            text="Test chunk",
            embedding=[1.0, 2.0, 3.0],
            chunk_index=0,
        )
        disk_store.add_chunk(chunk)

        # Delete library (should cascade)
        disk_store.delete_library(library.id)

        # Verify all files are deleted
        assert not (temp_dir / "libraries" / f"{library.id}.json").exists()
        assert not (temp_dir / "documents" / f"{document.id}.json").exists()
        assert not (temp_dir / "chunks" / f"{chunk.id}.json").exists()
        assert not (temp_dir / "indexes" / f"{library.id}.json").exists()

    def test_search_works_after_reload(self, disk_store, temp_dir):
        """Test that search works after reloading from disk."""
        library = Library(name="Library", embedding_dimension=3)
        disk_store.add_library(library)

        document = Document(library_id=library.id, name="Document")
        disk_store.add_document(document)

        # Add multiple chunks
        chunks = []
        for i in range(5):
            chunk = Chunk(
                document_id=document.id,
                text=f"Chunk {i}",
                embedding=[float(i), float(i + 1), float(i + 2)],
                chunk_index=i,
            )
            disk_store.add_chunk(chunk)
            chunks.append(chunk)

        # Search before reload
        query = [0.0, 1.0, 2.0]
        results_before = disk_store.search(library.id, query, k=3)
        assert len(results_before) == 3

        # Reload from disk
        new_store = DiskVectorStore(temp_dir)
        results_after = new_store.search(library.id, query, k=3)

        # Results should be the same
        assert len(results_after) == 3
        assert [r[0] for r in results_before] == [r[0] for r in results_after]

    def test_random_projection_index_persists(self, disk_store, temp_dir):
        """Test that RandomProjectionIndex is persisted correctly."""
        library = Library(
            name="Library",
            embedding_dimension=3,
            index_kind="random_projection",
        )
        disk_store.add_library(library)

        document = Document(library_id=library.id, name="Document")
        disk_store.add_document(document)

        chunk = Chunk(
            document_id=document.id,
            text="Test chunk",
            embedding=[1.0, 2.0, 3.0],
            chunk_index=0,
        )
        disk_store.add_chunk(chunk)

        # Verify projection matrix file exists
        proj_file = temp_dir / "indexes" / f"{library.id}.projections.npy"
        assert proj_file.exists()

        # Load from new store and verify it's RandomProjectionIndex
        new_store = DiskVectorStore(temp_dir)
        # Force load by accessing a library
        loaded_lib = new_store.get_library(library.id)
        assert loaded_lib is not None
        assert library.id in new_store._vector_index
        assert isinstance(new_store._vector_index[library.id], RandomProjectionIndex)


class TestSettings:
    """Tests for configuration settings."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = Settings()
        assert settings.storage_type == "memory"
        assert settings.data_dir == Path("./data")
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000

    def test_settings_from_env(self, monkeypatch):
        """Test loading settings from environment variables."""
        monkeypatch.setenv("STORAGE_TYPE", "disk")
        monkeypatch.setenv("DATA_DIR", "/tmp/test_data")
        monkeypatch.setenv("PORT", "9000")

        settings = Settings()
        assert settings.storage_type == "disk"
        assert settings.data_dir == Path("/tmp/test_data")
        assert settings.port == 9000

    def test_ensure_data_dir_creates_directories(self, temp_dir):
        """Test that ensure_data_dir creates all necessary directories."""
        settings = Settings(storage_type="disk", data_dir=temp_dir / "vector_data")
        settings.ensure_data_dir()

        assert (temp_dir / "vector_data").exists()
        assert (temp_dir / "vector_data" / "libraries").exists()
        assert (temp_dir / "vector_data" / "documents").exists()
        assert (temp_dir / "vector_data" / "chunks").exists()
        assert (temp_dir / "vector_data" / "indexes").exists()


class TestAPIIntegration:
    """Tests for API integration with disk storage."""

    def test_create_app_with_memory_storage(self):
        """Test creating app with default memory storage."""
        from vector_db.api import create_app

        settings = Settings(storage_type="memory")
        app = create_app(settings)

        assert app.state.settings.storage_type == "memory"
        assert app.state.container is not None

    def test_create_app_with_disk_storage(self, temp_dir):
        """Test creating app with disk storage."""
        from vector_db.api import create_app

        settings = Settings(storage_type="disk", data_dir=temp_dir)
        app = create_app(settings)

        assert app.state.settings.storage_type == "disk"
        assert app.state.container is not None

    def test_full_workflow_with_disk_persistence(self, temp_dir):
        """Test complete workflow with disk persistence."""
        from vector_db.api import create_app
        from fastapi.testclient import TestClient

        settings = Settings(storage_type="disk", data_dir=temp_dir)
        app = create_app(settings)
        client = TestClient(app)

        # Create library
        library_response = client.post(
            "/libraries",
            json={
                "name": "Test Library",
                "embedding_dimension": 3,
                "distance_metric": "cosine",
            },
        )
        assert library_response.status_code == 201
        library_id = library_response.json()["id"]

        # Create document
        doc_response = client.post(
            "/documents",
            json={"library_id": library_id, "name": "Test Document"},
        )
        assert doc_response.status_code == 201
        document_id = doc_response.json()["id"]

        # Create chunk
        chunk_response = client.post(
            "/chunks",
            json={
                "document_id": document_id,
                "text": "Test text",
                "embedding": [1.0, 2.0, 3.0],
                "chunk_index": 0,
            },
        )
        assert chunk_response.status_code == 201

        # Verify files exist
        assert (temp_dir / "libraries" / f"{library_id}.json").exists()
        assert (temp_dir / "documents" / f"{document_id}.json").exists()

        # Create new app instance (simulates server restart)
        new_app = create_app(settings)
        new_client = TestClient(new_app)

        # Verify data persisted
        lib_response = new_client.get(f"/libraries/{library_id}")
        assert lib_response.status_code == 200
        assert lib_response.json()["name"] == "Test Library"

        doc_response = new_client.get(f"/documents/{document_id}")
        assert doc_response.status_code == 200
        assert doc_response.json()["name"] == "Test Document"
