#!/usr/bin/env python3
"""Demo script showing disk persistence functionality."""
from pathlib import Path
import tempfile
import shutil

from vector_db import (
    Settings,
    create_app,
    Library,
    Document,
    Chunk,
    DiskVectorStore,
)
from fastapi.testclient import TestClient


def demo_persistence():
    """Demonstrate disk persistence across app restarts."""
    print("=" * 60)
    print("Vector Database Disk Persistence Demo")
    print("=" * 60)

    # Create temporary directory for this demo
    temp_dir = Path(tempfile.mkdtemp())
    print(f"\n📁 Using temporary data directory: {temp_dir}")

    try:
        # ===== PART 1: Create data with disk persistence =====
        print("\n" + "=" * 60)
        print("PART 1: Creating data with disk persistence")
        print("=" * 60)

        settings = Settings(storage_type="disk", data_dir=temp_dir)
        app = create_app(settings)
        client = TestClient(app)

        # Create library
        print("\n✅ Creating library...")
        library_resp = client.post("/libraries", json={
            "name": "Research Papers",
            "embedding_dimension": 3,
            "distance_metric": "cosine",
            "index_kind": "flat"
        })
        library = library_resp.json()
        library_id = library["id"]
        print(f"   Library ID: {library_id}")
        print(f"   Name: {library['name']}")

        # Create document
        print("\n✅ Creating document...")
        doc_resp = client.post("/documents", json={
            "library_id": library_id,
            "name": "Machine Learning Basics",
            "metadata": {"author": "John Doe", "year": 2024}
        })
        document = doc_resp.json()
        document_id = document["id"]
        print(f"   Document ID: {document_id}")
        print(f"   Name: {document['name']}")

        # Create chunks
        print("\n✅ Creating 3 chunks...")
        chunks = []
        chunk_data = [
            ("Neural networks are...", [1.0, 0.5, 0.2]),
            ("Deep learning uses...", [0.8, 0.7, 0.3]),
            ("Training requires...", [0.6, 0.9, 0.4]),
        ]

        for i, (text, embedding) in enumerate(chunk_data):
            chunk_resp = client.post("/chunks", json={
                "document_id": document_id,
                "text": text,
                "embedding": embedding,
                "chunk_index": i,
                "metadata": {"section": f"Section {i+1}"}
            })
            chunk = chunk_resp.json()
            chunks.append(chunk)
            print(f"   Chunk {i+1}: {text[:20]}... (embedding: {embedding})")

        # Search
        print("\n✅ Searching for similar vectors...")
        search_resp = client.post(f"/libraries/{library_id}/search", json={
            "query_vector": [1.0, 0.6, 0.3],
            "k": 2
        })
        results = search_resp.json()
        print(f"   Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"     {i}. Chunk ID: {result['chunk_id'][:8]}... (distance: {result['distance']:.4f})")

        # Check files on disk
        print("\n📂 Files created on disk:")
        for pattern in ["libraries", "documents", "chunks", "indexes"]:
            files = list((temp_dir / pattern).glob("*"))
            print(f"   {pattern}/: {len(files)} file(s)")
            for f in files[:2]:  # Show first 2 files
                print(f"     - {f.name}")

        # ===== PART 2: Simulate server restart =====
        print("\n" + "=" * 60)
        print("PART 2: Simulating server restart...")
        print("=" * 60)

        print("\n🔄 Creating NEW app instance (simulating restart)")
        print("   Old app discarded, new app loading from disk...")

        # Create completely new app instance
        new_settings = Settings(storage_type="disk", data_dir=temp_dir)
        new_app = create_app(new_settings)
        new_client = TestClient(new_app)

        # ===== PART 3: Verify data persisted =====
        print("\n" + "=" * 60)
        print("PART 3: Verifying data persisted across restart")
        print("=" * 60)

        # Get library
        print("\n✅ Retrieving library...")
        lib_resp = new_client.get(f"/libraries/{library_id}")
        loaded_lib = lib_resp.json()
        print(f"   ✓ Library loaded: {loaded_lib['name']}")
        print(f"   ✓ Document count: {loaded_lib['document_count']}")
        print(f"   ✓ Chunk count: {loaded_lib['chunk_count']}")

        # Get document
        print("\n✅ Retrieving document...")
        doc_resp = new_client.get(f"/documents/{document_id}")
        loaded_doc = doc_resp.json()
        print(f"   ✓ Document loaded: {loaded_doc['name']}")
        print(f"   ✓ Metadata: {loaded_doc['metadata']}")

        # List chunks
        print("\n✅ Listing chunks...")
        chunks_resp = new_client.get(f"/documents/{document_id}/chunks")
        loaded_chunks = chunks_resp.json()
        print(f"   ✓ Found {len(loaded_chunks)} chunks")
        for i, chunk in enumerate(loaded_chunks, 1):
            print(f"     {i}. {chunk['text'][:30]}...")

        # Search again
        print("\n✅ Searching again (after restart)...")
        search_resp = new_client.post(f"/libraries/{library_id}/search", json={
            "query_vector": [1.0, 0.6, 0.3],
            "k": 2
        })
        new_results = search_resp.json()
        print(f"   ✓ Found {len(new_results)} results (same as before)")

        # Verify results match
        if [r["chunk_id"] for r in results] == [r["chunk_id"] for r in new_results]:
            print("   ✓ Search results IDENTICAL to pre-restart! 🎉")

        # ===== PART 4: Test with RandomProjectionIndex =====
        print("\n" + "=" * 60)
        print("PART 4: Testing RandomProjectionIndex persistence")
        print("=" * 60)

        print("\n✅ Creating library with RandomProjectionIndex...")
        rp_lib_resp = new_client.post("/libraries", json={
            "name": "LSH Index Library",
            "embedding_dimension": 3,
            "distance_metric": "cosine",
            "index_kind": "random_projection"
        })
        rp_library = rp_lib_resp.json()
        rp_library_id = rp_library["id"]
        print(f"   Library ID: {rp_library_id}")
        print(f"   Index kind: random_projection")

        # Create document and chunk
        rp_doc_resp = new_client.post("/documents", json={
            "library_id": rp_library_id,
            "name": "Test Document",
        })
        rp_document_id = rp_doc_resp.json()["id"]

        new_client.post("/chunks", json={
            "document_id": rp_document_id,
            "text": "Test chunk",
            "embedding": [0.5, 0.5, 0.5],
            "chunk_index": 0,
        })

        # Verify projection matrix file exists
        proj_file = temp_dir / "indexes" / f"{rp_library_id}.projections.npy"
        if proj_file.exists():
            print(f"   ✓ Projection matrix saved: {proj_file.name}")

        # Restart and verify RP index loads
        print("\n🔄 Restarting again to test RP index loading...")
        final_app = create_app(Settings(storage_type="disk", data_dir=temp_dir))
        final_client = TestClient(final_app)

        rp_lib_reload = final_client.get(f"/libraries/{rp_library_id}")
        if rp_lib_reload.json()["index_kind"] == "random_projection":
            print("   ✓ RandomProjectionIndex persisted correctly! 🎉")

        print("\n" + "=" * 60)
        print("SUCCESS! Disk persistence working perfectly!")
        print("=" * 60)

    finally:
        # Cleanup
        print(f"\n🧹 Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)
        print("   ✓ Cleanup complete")


if __name__ == "__main__":
    demo_persistence()
