from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List
from uuid import UUID

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError

from .config import Settings, get_logger
from .disk_store import DiskVectorStore
from .entities import Chunk, Document, Library
from .exceptions import (
    ChunkNotFoundError,
    ConflictError,
    DocumentNotFoundError,
    LibraryNotFoundError,
    ResourceNotFoundError,
    SearchError,
    StorageError,
    ValidationError,
    VectorDBError,
)
from .repositories import (
    DiskChunkRepository,
    DiskDocumentRepository,
    DiskLibraryRepository,
    VectorStoreChunkRepository,
    VectorStoreDocumentRepository,
    VectorStoreLibraryRepository,
)
from .schemas import (
    ChunkBatchCreate,
    ChunkBatchCreateResult,
    ChunkCreate,
    ChunkResponse,
    ChunkSearchRequest,
    ChunkSearchResult,
    ChunkUpdate,
    DocumentCreate,
    DocumentResponse,
    DocumentUpdate,
    LibraryCreate,
    LibraryResponse,
    LibraryUpdate,
    PaginatedResponse,
    chunk_to_response,
    document_to_response,
    library_to_response,
)
from .services import ChunkService, DocumentService, LibraryService
from .vector_store import VectorStore

logger = get_logger("api")


@dataclass(slots=True)
class ServiceContainer:
    library_service: LibraryService
    document_service: DocumentService
    chunk_service: ChunkService

    @classmethod
    def build(cls, settings: Settings) -> "ServiceContainer":
        """Build the service container with storage based on settings."""
        logger.info(f"Building service container with storage_type={settings.storage_type}")

        if settings.storage_type == "disk":
            settings.ensure_data_dir()
            store = DiskVectorStore(settings.data_dir)
            library_repo = DiskLibraryRepository(store)
            document_repo = DiskDocumentRepository(store)
            chunk_repo = DiskChunkRepository(store)
            logger.info(f"Initialized disk storage at {settings.data_dir}")
        else:  # memory
            store = VectorStore()
            library_repo = VectorStoreLibraryRepository(store)
            document_repo = VectorStoreDocumentRepository(store)
            chunk_repo = VectorStoreChunkRepository(store)
            logger.info("Initialized in-memory storage")

        library_service = LibraryService(library_repo)
        document_service = DocumentService(document_repo, library_repo)
        chunk_service = ChunkService(chunk_repo, document_repo, library_repo)

        return cls(
            library_service=library_service,
            document_service=document_service,
            chunk_service=chunk_service,
        )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create FastAPI application with configurable storage backend."""
    if settings is None:
        settings = Settings()

    # Configure logging
    settings.configure_logging()
    logger.info("Starting Vector DB application")

    app = FastAPI(
        title="Vector DB",
        description=f"Vector database with {settings.storage_type} storage",
        version="0.1.0",
    )
    app.state.container = ServiceContainer.build(settings)
    app.state.settings = settings

    # Register exception handlers
    _register_exception_handlers(app)

    app.include_router(_library_router())
    app.include_router(_document_router())
    app.include_router(_chunk_router())

    logger.info("Vector DB application ready")
    return app


def _register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers for custom exceptions."""

    @app.exception_handler(ResourceNotFoundError)
    async def handle_not_found(request: Request, exc: ResourceNotFoundError) -> JSONResponse:
        logger.warning(f"Resource not found: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "error": "ResourceNotFoundError",
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(ValidationError)
    async def handle_validation_error(request: Request, exc: ValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": type(exc).__name__,
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(SearchError)
    async def handle_search_error(request: Request, exc: SearchError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": type(exc).__name__,
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(ConflictError)
    async def handle_conflict(request: Request, exc: ConflictError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "error": type(exc).__name__,
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(StorageError)
    async def handle_storage_error(request: Request, exc: StorageError) -> JSONResponse:
        logger.error(f"Storage error: {exc.message}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": type(exc).__name__,
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(VectorDBError)
    async def handle_vector_db_error(request: Request, exc: VectorDBError) -> JSONResponse:
        """Catch-all for any VectorDBError not handled above."""
        logger.error(f"Vector DB error: {exc.message}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": type(exc).__name__,
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(PydanticValidationError)
    async def handle_pydantic_validation_error(
        request: Request, exc: PydanticValidationError
    ) -> JSONResponse:
        """Handle Pydantic validation errors from request body parsing."""
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "ValidationError",
                "message": "Request validation failed",
                "details": {"errors": exc.errors()},
            },
        )


def _get_container(request: Request) -> ServiceContainer:
    return request.app.state.container


def _library_service(container: ServiceContainer = Depends(_get_container)) -> LibraryService:
    return container.library_service


def _document_service(container: ServiceContainer = Depends(_get_container)) -> DocumentService:
    return container.document_service


def _chunk_service(container: ServiceContainer = Depends(_get_container)) -> ChunkService:
    return container.chunk_service


def _library_router() -> APIRouter:
    router = APIRouter(prefix="/libraries", tags=["libraries"])

    @router.post("", response_model=LibraryResponse, status_code=status.HTTP_201_CREATED)
    def create_library(
        payload: LibraryCreate, service: LibraryService = Depends(_library_service)
    ) -> LibraryResponse:
        logger.info(f"Creating library: {payload.name}")
        library = service.create_library(payload)
        logger.info(f"Created library {library.id} with dimension {library.embedding_dimension}")
        return library_to_response(library)

    @router.get("", response_model=PaginatedResponse[LibraryResponse])
    def list_libraries(
        skip: int = 0, limit: int = 100, service: LibraryService = Depends(_library_service)
    ) -> PaginatedResponse[LibraryResponse]:
        libraries = service.list_libraries()
        responses = [library_to_response(library) for library in libraries]
        return PaginatedResponse.paginate(responses, skip, limit)

    @router.get("/{library_id}", response_model=LibraryResponse)
    def get_library(
        library_id: UUID, service: LibraryService = Depends(_library_service)
    ) -> LibraryResponse:
        if not (library := service.get_library(library_id)):
            raise LibraryNotFoundError(library_id)
        return library_to_response(library)

    @router.patch("/{library_id}", response_model=LibraryResponse)
    def update_library(
        library_id: UUID,
        payload: LibraryUpdate,
        service: LibraryService = Depends(_library_service),
    ) -> LibraryResponse:
        updated = service.update_library(library_id, payload)
        return library_to_response(updated)

    @router.delete("/{library_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_library(library_id: UUID, service: LibraryService = Depends(_library_service)) -> None:
        if not service.delete_library(library_id):
            raise LibraryNotFoundError(library_id)

    @router.get("/{library_id}/documents", response_model=PaginatedResponse[DocumentResponse])
    def list_library_documents(
        library_id: UUID,
        skip: int = 0,
        limit: int = 100,
        service: DocumentService = Depends(_document_service),
    ) -> PaginatedResponse[DocumentResponse]:
        documents = service.list_documents(library_id=library_id)
        responses = [document_to_response(doc) for doc in documents]
        return PaginatedResponse.paginate(responses, skip, limit)

    @router.post("/{library_id}/search", response_model=List[ChunkSearchResult])
    def search_chunks(
        library_id: UUID,
        payload: ChunkSearchRequest,
        service: ChunkService = Depends(_chunk_service),
    ) -> List[ChunkSearchResult]:
        logger.info(f"Searching library {library_id} with k={payload.k}, filters={payload.metadata_filters}")
        results = service.search_chunks(
            library_id,
            payload.query_vector,
            payload.k,
            metadata_filters=payload.metadata_filters,
        )
        logger.info(f"Search returned {len(results)} results")
        return [ChunkSearchResult(chunk_id=chunk_id, distance=distance) for chunk_id, distance in results]

    return router


def _document_router() -> APIRouter:
    router = APIRouter(prefix="/documents", tags=["documents"])

    @router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
    def create_document(
        payload: DocumentCreate,
        service: DocumentService = Depends(_document_service),
    ) -> DocumentResponse:
        document = service.create_document(payload)
        return document_to_response(document)

    @router.get("/{document_id}", response_model=DocumentResponse)
    def get_document(
        document_id: UUID, service: DocumentService = Depends(_document_service)
    ) -> DocumentResponse:
        if not (document := service.get_document(document_id)):
            raise DocumentNotFoundError(document_id)
        return document_to_response(document)

    @router.patch("/{document_id}", response_model=DocumentResponse)
    def update_document(
        document_id: UUID,
        payload: DocumentUpdate,
        service: DocumentService = Depends(_document_service),
    ) -> DocumentResponse:
        return document_to_response(service.update_document(document_id, payload))

    @router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_document(document_id: UUID, service: DocumentService = Depends(_document_service)) -> None:
        if not service.delete_document(document_id):
            raise DocumentNotFoundError(document_id)

    @router.get("/{document_id}/chunks", response_model=PaginatedResponse[ChunkResponse])
    def list_document_chunks(
        document_id: UUID,
        skip: int = 0,
        limit: int = 100,
        chunk_service: ChunkService = Depends(_chunk_service),
    ) -> PaginatedResponse[ChunkResponse]:
        chunks = chunk_service.list_chunks(document_id=document_id)
        responses = [chunk_to_response(chunk) for chunk in chunks]
        return PaginatedResponse.paginate(responses, skip, limit)

    return router


def _chunk_router() -> APIRouter:
    router = APIRouter(prefix="/chunks", tags=["chunks"])

    @router.post("", response_model=ChunkResponse, status_code=status.HTTP_201_CREATED)
    def create_chunk(
        payload: ChunkCreate, service: ChunkService = Depends(_chunk_service)
    ) -> ChunkResponse:
        chunk = service.create_chunk(payload)
        return chunk_to_response(chunk)

    @router.post("/batch", response_model=ChunkBatchCreateResult, status_code=status.HTTP_201_CREATED)
    def create_chunks_batch(
        payload: ChunkBatchCreate, service: ChunkService = Depends(_chunk_service)
    ) -> ChunkBatchCreateResult:
        logger.info(f"Batch creating {len(payload.chunks)} chunks for document {payload.document_id}")
        chunks = service.create_chunks_batch(payload.chunks)
        logger.info(f"Batch created {len(chunks)} chunks successfully")
        return ChunkBatchCreateResult(created_count=len(chunks), chunk_ids=[c.id for c in chunks])

    @router.get("/{chunk_id}", response_model=ChunkResponse)
    def get_chunk(
        chunk_id: UUID, service: ChunkService = Depends(_chunk_service)
    ) -> ChunkResponse:
        if not (chunk := service.get_chunk(chunk_id)):
            raise ChunkNotFoundError(chunk_id)
        return chunk_to_response(chunk)

    @router.patch("/{chunk_id}", response_model=ChunkResponse)
    def update_chunk(
        chunk_id: UUID,
        payload: ChunkUpdate,
        service: ChunkService = Depends(_chunk_service),
    ) -> ChunkResponse:
        return chunk_to_response(service.update_chunk(chunk_id, payload))

    @router.delete("/{chunk_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_chunk(chunk_id: UUID, service: ChunkService = Depends(_chunk_service)) -> None:
        if not service.delete_chunk(chunk_id):
            raise ChunkNotFoundError(chunk_id)

    return router


__all__ = ["create_app", "ServiceContainer"]
