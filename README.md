# Vector Database

A production-ready Python vector database with clean architecture, dual storage backends, and multiple indexing strategies.

## Features

- **Full CRUD operations** for libraries, documents, and chunks
- **Vector similarity search** with k-NN retrieval
- **Multiple distance metrics**: Cosine, Euclidean, Dot Product
- **Two indexing algorithms**:
  - FlatIndex (exact brute-force search)
  - RandomProjectionIndex (approximate LSH-based search)
- **Dual storage backends**:
  - In-memory (fast, volatile)
  - Disk-based (persistent, survives restarts)
- **Pre-search metadata filtering** for efficient queries
- **Pagination** on all list endpoints
- **Batch operations** (create up to 1000 chunks atomically)
- **Structured logging** with configurable levels
- **Thread-safe** with RLock-based concurrency
- **Type-safe** with full Pydantic validation
- **RESTful API** with FastAPI

## Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd stack-try2

# Install dependencies
uv sync
```

### Running the Server

**In-memory mode (default):**
```bash
uvicorn vector_db.api:create_app --factory --reload
```

**Disk persistence mode:**
```bash
echo "STORAGE_TYPE=disk" > .env
uvicorn vector_db.api:create_app --factory --reload
```

API available at `http://localhost:8000` | Docs at `http://localhost:8000/docs`

## Usage Example

```python
from fastapi.testclient import TestClient
from vector_db.api import create_app

client = TestClient(create_app())

# Create a library
library = client.post("/libraries", json={
    "name": "My Embeddings",
    "embedding_dimension": 384,
    "distance_metric": "cosine"
}).json()

# Create a document
document = client.post("/documents", json={
    "library_id": library["id"],
    "name": "Document 1"
}).json()

# Add chunks with embeddings
chunk = client.post("/chunks", json={
    "document_id": document["id"],
    "text": "Sample text",
    "embedding": [0.1, 0.2, ...],  # 384-dimensional
    "chunk_index": 0
}).json()

# Search for similar vectors
results = client.post(f"/libraries/{library['id']}/search", json={
    "query_vector": [0.15, 0.18, ...],
    "k": 10,
    "metadata_filters": {"category": "tech"}  # optional
}).json()
```

## Architecture

The project follows clean architecture principles with clear separation of concerns:

```
┌─────────────────────────────────────┐
│      API Layer (FastAPI)            │  ← HTTP endpoints, routing
├─────────────────────────────────────┤
│      Service Layer                  │  ← Business logic, validation
│  LibraryService | DocumentService   │
│       ChunkService                  │
├─────────────────────────────────────┤
│      Repository Layer               │  ← Storage abstraction
│  (Protocols for swappable backends) │
├─────────────────────────────────────┤
│   VectorStore    │  DiskVectorStore │  ← Storage implementations
│   (In-Memory)    │  (Persistent)    │
├─────────────────────────────────────┤
│      Vector Indexes                 │  ← Search algorithms
│  FlatIndex | RandomProjectionIndex  │
└─────────────────────────────────────┘
```

**Key Design Decisions:**

- **Repository Pattern**: Enables swapping storage backends (memory ↔ disk) without changing business logic
- **Service Layer**: Coordinates multiple repositories, enforces business rules (dimension validation, parent existence)
- **Protocol-based Interfaces**: Python Protocols for structural subtyping (no ABC inheritance)
- **Strategy Pattern**: Multiple index implementations selected at runtime
- **Dependency Injection**: FastAPI `Depends()` wires components cleanly

## Project Structure

```
vector_db/
├── entities.py          # Domain models (Library, Document, Chunk)
├── schemas.py           # API DTOs (Request/Response models)
├── indexes.py           # Vector search algorithms
├── vector_store.py      # In-memory storage implementation
├── disk_store.py        # Disk-backed storage implementation
├── repositories.py      # Repository interfaces & implementations
├── services.py          # Business logic layer
├── api.py               # FastAPI routes & dependency injection
├── exceptions.py        # Custom exception hierarchy
└── config.py            # Settings & logging configuration

tests/                   # 55 tests, ~85% coverage
├── test_entities.py
├── test_indexes.py
├── test_vector_store.py
├── test_disk_persistence.py
├── test_services.py
└── test_api.py
```

## Data Model

```
Library (Collection of documents)
  ├── embedding_dimension: int         # e.g., 384 for sentence-transformers
  ├── distance_metric: str             # "cosine", "euclidean", "dot_product"
  ├── index_kind: str                  # "flat", "random_projection"
  └── Documents[]
       ├── name: str
       ├── metadata: Dict[str, Any]
       └── Chunks[]
            ├── text: str              # Original text content
            ├── embedding: List[float] # Vector representation
            ├── metadata: Dict         # Filterable key-value pairs
            └── chunk_index: int       # Position in document
```

**Hierarchy:**
- **Library**: Enforces consistent embedding dimension and distance metric
- **Document**: Logical grouping of related chunks
- **Chunk**: Text piece with vector embedding (searchable unit)

## Indexing Algorithms

### FlatIndex (Exact Search)
- **Algorithm**: Brute-force comparison against all vectors
- **Complexity**: O(n × d) where n = vectors, d = dimensions
- **Accuracy**: 100% (exact results)
- **Best for**: <10K vectors, exact results required

### RandomProjectionIndex (LSH - Approximate Search)
- **Algorithm**: Locality-Sensitive Hashing with random hyperplanes
- **Complexity**: O(√n × d) average case
- **Accuracy**: ~95% (configurable via num_projections)
- **Best for**: >10K vectors, speed prioritized

**Note:** Index type selected at library creation via `index_kind` parameter.

## Configuration

Create a `.env` file:

```env
# Storage backend
STORAGE_TYPE=disk                    # "memory" or "disk"
DATA_DIR=./data                      # Directory for disk storage

# Server settings
HOST=0.0.0.0
PORT=8000

# Logging
LOG_LEVEL=INFO                       # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

## API Reference

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| **Libraries** |
| POST | `/libraries` | Create library with embedding dimension & distance metric |
| GET | `/libraries?skip=0&limit=100` | List libraries (paginated) |
| GET | `/libraries/{id}` | Get library details |
| PATCH | `/libraries/{id}` | Update library metadata |
| DELETE | `/libraries/{id}` | Delete library (cascades to documents/chunks) |
| POST | `/libraries/{id}/search` | Search vectors with optional metadata filters |
| **Documents** |
| POST | `/documents` | Create document in a library |
| GET | `/documents/{id}` | Get document details |
| GET | `/documents/{id}/chunks?skip=0&limit=100` | List chunks (paginated) |
| **Chunks** |
| POST | `/chunks` | Create single chunk |
| POST | `/chunks/batch` | Batch create up to 1000 chunks |
| GET | `/chunks/{id}` | Get chunk (excludes embedding for bandwidth) |
| PATCH | `/chunks/{id}` | Update chunk (text, embedding, metadata) |

**Search Request:**
```json
{
  "query_vector": [0.1, 0.2, ...],
  "k": 10,
  "metadata_filters": {"category": "tech"}  // optional pre-filter
}
```

**Search Response:**
```json
[
  {"chunk_id": "uuid", "distance": 0.23},
  {"chunk_id": "uuid", "distance": 0.45}
]
```

## Disk Persistence

When `STORAGE_TYPE=disk`, data is stored in:

```
data/
├── libraries/{library_id}.json          # Library metadata
├── documents/{document_id}.json         # Document metadata
├── chunks/{chunk_id}.json               # Chunk text + metadata (no embedding)
└── indexes/
    ├── {library_id}.json                # Index metadata
    ├── {library_id}.npy                 # Embeddings (NumPy binary)
    └── {library_id}.projections.npy     # Random projections (LSH only)
```

**Design rationale:**
- JSON for structured metadata (human-readable)
- NumPy binary (`.npy`) for vector embeddings (space-efficient)
- Embeddings stored in index, not chunk files (avoids duplication)
- Lazy loading: data loaded on first access

## Testing

```bash
# Run all tests
uv run pytest

# With coverage report
uv run pytest --cov=vector_db --cov-report=term-missing

# Specific test file
uv run pytest tests/test_disk_persistence.py -v
```

**Test Coverage:** ~85% across all layers (entities, indexes, stores, services, API)

## Advanced Features

### Pagination
All list endpoints return paginated responses:
```python
response = client.get("/libraries?skip=10&limit=20")
# {"items": [...], "total": 100, "skip": 10, "limit": 20, "has_more": true}
```

### Batch Operations
Create multiple chunks atomically:
```python
client.post("/chunks/batch", json={
    "document_id": "uuid",
    "chunks": [
        {"document_id": "uuid", "text": "Chunk 1", "embedding": [...], "chunk_index": 0},
        {"document_id": "uuid", "text": "Chunk 2", "embedding": [...], "chunk_index": 1}
    ]
})
# Response: {"created_count": 2, "chunk_ids": ["uuid1", "uuid2"]}
```

### Metadata Filtering
Pre-search filtering (more efficient than post-filtering):
```python
# Only searches chunks matching metadata filters
results = client.post(f"/libraries/{id}/search", json={
    "query_vector": [...],
    "k": 10,
    "metadata_filters": {"source": "pdf", "page": 5}
})
```

### Structured Logging
```bash
export LOG_LEVEL=DEBUG
uvicorn vector_db.api:create_app --factory

# Logs: startup, CRUD operations, search queries, errors
```

## Dependencies

**Core:**
- FastAPI (>=0.110) - Web framework
- Pydantic (>=2.11) - Data validation
- NumPy (>=1.26) - Vector operations
- Pydantic-Settings (>=2.0) - Configuration

**Dev:**
- pytest (>=8.2) - Testing framework
- httpx (>=0.27) - HTTP client for tests

## Performance Characteristics

| Operation | In-Memory | Disk |
|-----------|-----------|------|
| Add chunk | O(d) | O(d) + disk write |
| Get by ID | O(1) | O(1) + lazy load |
| Search (Flat) | O(n×d) | O(n×d) + lazy load |
| Search (LSH) | O(√n×d) | O(√n×d) + lazy load |

**Concurrency:** Thread-safe via RLock (single process). For distributed deployment, use leader-follower pattern.

## Design Highlights

**Why this architecture?**
- **Pythonic**: Idiomatic Python (walrus operator, comprehensions, protocols)
- **Testable**: Dependency injection enables easy mocking
- **Extensible**: Add new indexes by implementing `VectorIndex` protocol
- **Type-safe**: Full type hints + Pydantic runtime validation
- **Production-ready**: Error handling, logging, persistence, testing

**Key Tradeoffs:**
- **RLock vs RWLock**: Chose RLock for simplicity; writes are fast enough
- **JSON + NumPy vs SQLite**: Easier to debug, simpler implementation
- **Pre-filtering vs Post-filtering**: Pre-filtering is more efficient
- **Repository pattern overhead**: Worth it for storage flexibility

## License

MIT

---

**Built to demonstrate:**
- Clean software architecture (Repository, Service, DI patterns)
- Type-safe Python with Pydantic
- Production features (persistence, multiple indexes, logging)
- Comprehensive testing (~85% coverage)
- Clear, concise documentation
