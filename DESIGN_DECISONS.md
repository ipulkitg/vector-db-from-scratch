# Design Decisions

## Data Models
- **Pydantic models** for Chunk, Document, Library (validation + FastAPI integration)
- **UUID4** for all IDs (thread-safe, globally unique)
- **Three-level hierarchy**: Library → Document → Chunk
- **Fixed embedding dimension** per library (mathematical requirement for search)
- **Flexible metadata**: `Dict[str, str | int | float | bool]` for filtering

## Storage Layer
- **Dual storage implementations**:
  - **In-memory (VectorStore)**: Dictionaries for O(1) lookups, volatile
  - **Disk-backed (DiskVectorStore)**: JSON + NumPy files, persistent across restarts
- **Secondary indexes** using `Dict[UUID, Set[UUID]]`:
  - `_docs_by_library`: Library → Document IDs
  - `_chunks_by_document`: Document → Chunk IDs
  - `_chunks_by_library`: Library → Chunk IDs
- **Sets over Lists** in indexes (O(1) remove/contains vs O(n))
- **Configuration-based switching**: Environment variable `STORAGE_TYPE` selects backend

## Concurrency
- **RLock (Reentrant Lock)** for thread safety (allows nested locking for cascade operations)
- **Single global lock** (simple, correct, sufficient for MVP)

## Data Integrity
- **Cascade deletes**: Parent deletion automatically deletes children
- **Synchronous count updates**: `document_count`, `chunk_count` updated immediately

## Indexing
- **Multiple index implementations**:
  - **FlatIndex** (brute-force): Time O(n*d), exact results
  - **RandomProjectionIndex** (LSH-based): Time O(sqrt(n)*d), approximate results
- **Distance metrics**: Cosine (default), Euclidean, Dot Product
- **Index persistence**: Save/load to disk with NumPy binary format
- **NumPy vectorization** for batch similarity computation
- **Metadata filtering support** (post-search filtering)

## Architecture
- **Layered design**: API → Service → Repository → VectorStore
- **Repository pattern**: Protocol-based abstract interface (persistence-agnostic)
- **Service layer**: Business logic, cross-repository validation
- **API layer**: Thin FastAPI endpoints with dependency injection

## Type Safety
- **Type hints** on all functions and attributes
- **Pydantic validators** for runtime validation

## Persistence Strategy
- **File format**: JSON for entities + NumPy binary for vectors
- **Directory structure**:
  ```
  data/
    libraries/{library_id}.json
    documents/{document_id}.json
    chunks/{chunk_id}.json
    indexes/{library_id}.{json,npy,projections.npy}
  ```
- **Lazy loading**: Data loaded on first access
- **Atomic writes**: Individual file writes ensure partial consistency
- **Index reconstruction**: Projection matrices saved for RandomProjectionIndex

## Future Enhancements
- Transaction support (multi-entity atomicity)
- Write-ahead logging for durability
- Advanced indexes (IVF, HNSW, Product Quantization)
- Read-Write locks for concurrent reads
- Compression for vector storage