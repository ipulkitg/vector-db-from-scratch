# Vector Database

A production-ready Python vector database with clean architecture, multiple indexing strategies, and configurable persistence.

## Features

- ✅ **Full CRUD operations** for libraries, documents, and chunks
- ✅ **Vector similarity search** with k-NN retrieval
- ✅ **Multiple distance metrics**: Cosine, Euclidean, Dot Product
- ✅ **Two indexing algorithms**:
  - FlatIndex (exact brute-force search)
  - RandomProjectionIndex (approximate LSH-based search)
- ✅ **Dual storage backends**:
  - In-memory (fast, volatile)
  - Disk-based (persistent, survives restarts)
- ✅ **Metadata filtering** on search results
- ✅ **Thread-safe** with RLock-based concurrency
- ✅ **Type-safe** with full Pydantic validation
- ✅ **RESTful API** with FastAPI
- ✅ **Clean architecture** with Repository and Service patterns

## Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd stack-try2

# Install dependencies
uv sync

# Or with dev dependencies
uv sync --all-extras
```

### Running the Server

**In-memory mode (default):**
```bash
uvicorn vector_db.api:create_app --factory --reload
```

**Disk persistence mode:**
```bash
# Create .env file
echo "STORAGE_TYPE=disk" > .env
echo "DATA_DIR=./data" >> .env

# Start server
uvicorn vector_db.api:create_app --factory --reload
```

The API will be available at `http://localhost:8000`.

**Interactive API docs:** http://localhost:8000/docs

## Usage Examples

### Python SDK

```python
from vector_db.config import Settings
from vector_db.api import create_app
from fastapi.testclient import TestClient

# Create app with disk persistence
settings = Settings(storage_type="disk", data_dir="./my_data")
app = create_app(settings)
client = TestClient(app)

# Create a library
library = client.post("/libraries", json={
    "name": "My Embeddings",
    "embedding_dimension": 384,
    "distance_metric": "cosine",
    "index_kind": "flat"  # or "random_projection"
}).json()

library_id = library["id"]

# Create a document
document = client.post("/documents", json={
    "library_id": library_id,
    "name": "My Document",
    "metadata": {"source": "pdf", "page": 1}
}).json()

document_id = document["id"]

# Add chunks with embeddings
chunk = client.post("/chunks", json={
    "document_id": document_id,
    "text": "This is a sample text chunk",
    "embedding": [0.1, 0.2, 0.3, ...],  # 384-dimensional vector
    "chunk_index": 0,
    "metadata": {"type": "paragraph"}
}).json()

# Search for similar vectors
results = client.post(f"/libraries/{library_id}/search", json={
    "query_vector": [0.15, 0.18, 0.35, ...],
    "k": 10
}).json()

for result in results:
    print(f"Chunk {result['chunk_id']}: distance = {result['distance']}")
```

### cURL Examples

```bash
# Create a library
curl -X POST http://localhost:8000/libraries \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Text Embeddings",
    "embedding_dimension": 128,
    "distance_metric": "cosine"
  }'

# List all libraries
curl http://localhost:8000/libraries

# Search vectors
curl -X POST http://localhost:8000/libraries/{library_id}/search \
  -H "Content-Type: application/json" \
  -d '{
    "query_vector": [0.1, 0.2, 0.3, ...],
    "k": 5
  }'
```

## Configuration

Create a `.env` file in the project root:

```env
# Storage backend: "memory" or "disk"
STORAGE_TYPE=disk

# Data directory for disk storage
DATA_DIR=./data

# Server settings
HOST=0.0.0.0
PORT=8000
```

Or use environment variables:
```bash
export STORAGE_TYPE=disk
export DATA_DIR=/var/lib/vectordb
uvicorn vector_db.api:create_app --factory
```

## Architecture

```
┌─────────────────────────────────────────────┐
│           FastAPI Application               │
│  (API Layer - Endpoints & Routing)          │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│          Service Layer                      │
│  (Business Logic & Validation)              │
│  - LibraryService                           │
│  - DocumentService                          │
│  - ChunkService                             │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│       Repository Layer                      │
│  (Storage Abstraction via Protocols)        │
│  - LibraryRepository                        │
│  - DocumentRepository                       │
│  - ChunkRepository                          │
└─────────────┬───────────────────────────────┘
              │
        ┌─────┴──────┐
        ▼            ▼
┌──────────────┐  ┌──────────────┐
│ VectorStore  │  │ DiskVector   │
│ (In-Memory)  │  │ Store (Disk) │
└──────┬───────┘  └──────┬───────┘
       │                 │
       ▼                 ▼
┌─────────────────────────────────┐
│      Vector Indexes             │
│  - FlatIndex (exact)            │
│  - RandomProjectionIndex (LSH)  │
└─────────────────────────────────┘
```

### Key Design Patterns

- **Repository Pattern**: Abstracts storage details, allows swapping backends
- **Service Layer**: Encapsulates business logic, coordinates repositories
- **Dependency Injection**: FastAPI `Depends()` wires components cleanly
- **Protocol-based Interfaces**: Python Protocols for structural subtyping
- **Strategy Pattern**: Multiple index implementations (Flat, RandomProjection)

## Data Model

```
Library (Collection)
  ├── embedding_dimension: int
  ├── distance_metric: "cosine" | "euclidean" | "dot_product"
  ├── index_kind: "flat" | "random_projection"
  └── Documents[]
       └── Chunks[]
            ├── text: str
            ├── embedding: List[float]
            ├── metadata: Dict[str, str|int|float|bool]
            └── chunk_index: int
```

## Indexing Algorithms

### FlatIndex (Exact Search)
- **Algorithm**: Brute-force comparison against all vectors
- **Time Complexity**: O(n × d) where n = vectors, d = dimensions
- **Space Complexity**: O(n × d)
- **Accuracy**: 100% (exact results)
- **Best for**: <10K vectors, exact results required

### RandomProjectionIndex (Approximate Search)
- **Algorithm**: Locality-Sensitive Hashing (LSH) with random hyperplanes
- **Time Complexity**: O(√n × d) average case
- **Space Complexity**: O(n × d + p × d) where p = projections
- **Accuracy**: ~95% (configurable via num_projections)
- **Best for**: >10K vectors, speed prioritized

## Disk Persistence Format

When using `STORAGE_TYPE=disk`, data is stored in:

```
data/
├── libraries/
│   └── {library_id}.json          # Library metadata
├── documents/
│   └── {document_id}.json         # Document metadata
├── chunks/
│   └── {chunk_id}.json            # Chunk text + metadata
└── indexes/
    ├── {library_id}.json          # Index metadata
    ├── {library_id}.npy           # Vector embeddings (NumPy)
    └── {library_id}.projections.npy  # Random projections (if LSH)
```

### Persistence Features

- **Lazy Loading**: Data loaded on first access
- **Atomic Writes**: Each entity written to separate file
- **Index Serialization**: NumPy binary format for vectors
- **Survives Restarts**: Full state restored from disk

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=vector_db --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_disk_persistence.py -v

# Run tests matching pattern
uv run pytest -k "test_search"
```

Test coverage: **~85%** across all layers.

## API Reference

### Libraries

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/libraries` | Create a new library |
| GET | `/libraries` | List all libraries |
| GET | `/libraries/{id}` | Get library by ID |
| PATCH | `/libraries/{id}` | Update library |
| DELETE | `/libraries/{id}` | Delete library (cascades) |
| GET | `/libraries/{id}/documents` | List library's documents |
| POST | `/libraries/{id}/search` | Search vectors in library |

### Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/documents` | Create a new document |
| GET | `/documents/{id}` | Get document by ID |
| PATCH | `/documents/{id}` | Update document |
| DELETE | `/documents/{id}` | Delete document (cascades) |
| GET | `/documents/{id}/chunks` | List document's chunks |

### Chunks

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chunks` | Create a new chunk |
| GET | `/chunks/{id}` | Get chunk by ID |
| PATCH | `/chunks/{id}` | Update chunk (including embedding) |
| DELETE | `/chunks/{id}` | Delete chunk |

## Performance Characteristics

| Operation | VectorStore | DiskVectorStore |
|-----------|-------------|-----------------|
| Add chunk | O(d) | O(d) + disk write |
| Get by ID | O(1) | O(1) + lazy load |
| Search (Flat) | O(n×d) | O(n×d) + lazy load |
| Search (LSH) | O(√n×d) | O(√n×d) + lazy load |
| Cascade delete | O(n×m) | O(n×m) + disk deletes |

## Project Structure

```
vector_db/
├── entities.py          # Domain models (Pydantic)
├── schemas.py           # API DTOs (Request/Response)
├── indexes.py           # Vector search algorithms
├── vector_store.py      # In-memory storage
├── disk_store.py        # Disk-backed storage
├── repositories.py      # Repository implementations
├── services.py          # Business logic layer
├── api.py               # FastAPI routes
└── config.py            # Settings management

tests/
├── test_entities.py
├── test_indexes.py
├── test_vector_store.py
├── test_disk_persistence.py
├── test_services.py
└── test_api.py
```

## Dependencies

- **FastAPI** (>=0.110): Web framework
- **Pydantic** (>=2.11): Data validation
- **Pydantic-Settings** (>=2.0): Configuration management
- **NumPy** (>=1.26): Vector operations

**Dev Dependencies:**
- **pytest** (>=8.2): Testing framework
- **httpx** (>=0.27): HTTP client for tests

## Design Decisions

See [DESIGN_DECISONS.md](DESIGN_DECISONS.md) for detailed rationale on:
- Storage layer architecture
- Concurrency strategy (RLock vs RWLock)
- Indexing algorithms (FlatIndex vs RandomProjection)
- Persistence format (JSON + NumPy binary)
- Repository pattern implementation

## Development

```bash
# Install with dev dependencies
uv sync --all-extras

# Run tests in watch mode
uv run pytest-watch

# Format code
uv run black vector_db tests

# Type checking
uv run mypy vector_db

# Start development server
uvicorn vector_db.api:create_app --factory --reload
```

## Roadmap

- [ ] Batch operations API (add multiple chunks atomically)
- [ ] Advanced indexes (IVF, HNSW, Product Quantization)
- [ ] Query-time metadata filtering (pre-search)
- [ ] Pagination for list endpoints
- [ ] Compression for vector storage
- [ ] Write-ahead logging for disk store
- [ ] Distributed deployment with leader-follower
- [ ] Temporal workflows for background tasks

## License

MIT

## Author

Built as a take-home assignment demonstrating:
- Clean software architecture
- Type-safe Python code
- Comprehensive testing
- Production-ready features (persistence, multiple indexes)
- Clear documentation

For design decisions and implementation details, see:
- [DESIGN_DECISONS.md](DESIGN_DECISONS.md) - Architecture rationale
- [DEVELOPMENT.md](DEVELOPMENT.md) - Development setup
- [goals.md](goals.md) - Project requirements
