from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List
from uuid import UUID

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status

from .config import Settings
from .disk_store import DiskVectorStore
from .entities import Chunk, Document, Library
from .repositories import (
    DiskChunkRepository,
    DiskDocumentRepository,
    DiskLibraryRepository,
    VectorStoreChunkRepository,
    VectorStoreDocumentRepository,
    VectorStoreLibraryRepository,
)
from .schemas import (
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
    chunk_to_response,
    document_to_response,
    library_to_response,
)
from .services import ChunkService, DocumentService, LibraryService
from .vector_store import VectorStore


@dataclass(slots=True)
class ServiceContainer:
    library_service: LibraryService
    document_service: DocumentService
    chunk_service: ChunkService

    @classmethod
    def build(cls, settings: Settings) -> "ServiceContainer":
        """Build the service container with storage based on settings."""
        if settings.storage_type == "disk":
            settings.ensure_data_dir()
            store = DiskVectorStore(settings.data_dir)
            library_repo = DiskLibraryRepository(store)
            document_repo = DiskDocumentRepository(store)
            chunk_repo = DiskChunkRepository(store)
        else:  # memory
            store = VectorStore()
            library_repo = VectorStoreLibraryRepository(store)
            document_repo = VectorStoreDocumentRepository(store)
            chunk_repo = VectorStoreChunkRepository(store)

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

    app = FastAPI(
        title="Vector DB",
        description=f"Vector database with {settings.storage_type} storage",
        version="0.1.0",
    )
    app.state.container = ServiceContainer.build(settings)
    app.state.settings = settings

    app.include_router(_library_router())
    app.include_router(_document_router())
    app.include_router(_chunk_router())
    return app


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
        library = service.create_library(payload)
        return library_to_response(library)

    @router.get("", response_model=List[LibraryResponse])
    def list_libraries(service: LibraryService = Depends(_library_service)) -> List[LibraryResponse]:
        return [library_to_response(library) for library in service.list_libraries()]

    @router.get("/{library_id}", response_model=LibraryResponse)
    def get_library(
        library_id: UUID, service: LibraryService = Depends(_library_service)
    ) -> LibraryResponse:
        library = service.get_library(library_id)
        if not library:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Library not found")
        return library_to_response(library)

    @router.patch("/{library_id}", response_model=LibraryResponse)
    def update_library(
        library_id: UUID,
        payload: LibraryUpdate,
        service: LibraryService = Depends(_library_service),
    ) -> LibraryResponse:
        try:
            updated = service.update_library(library_id, payload)
        except ValueError as exc:
            raise _http_error(exc)
        return library_to_response(updated)

    @router.delete("/{library_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_library(library_id: UUID, service: LibraryService = Depends(_library_service)) -> None:
        if not service.delete_library(library_id):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Library not found")

    @router.get("/{library_id}/documents", response_model=List[DocumentResponse])
    def list_library_documents(
        library_id: UUID,
        service: DocumentService = Depends(_document_service),
    ) -> List[DocumentResponse]:
        documents = service.list_documents(library_id=library_id)
        return [document_to_response(doc) for doc in documents]

    @router.post("/{library_id}/search", response_model=List[ChunkSearchResult])
    def search_chunks(
        library_id: UUID,
        payload: ChunkSearchRequest,
        service: ChunkService = Depends(_chunk_service),
    ) -> List[ChunkSearchResult]:
        try:
            results = service.search_chunks(
                library_id,
                payload.query_vector,
                payload.k,
                metadata_filters=payload.metadata_filters,
            )
        except ValueError as exc:
            raise _http_error(exc)
        return [ChunkSearchResult(chunk_id=chunk_id, distance=distance) for chunk_id, distance in results]

    return router


def _document_router() -> APIRouter:
    router = APIRouter(prefix="/documents", tags=["documents"])

    @router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
    def create_document(
        payload: DocumentCreate,
        service: DocumentService = Depends(_document_service),
    ) -> DocumentResponse:
        try:
            document = service.create_document(payload)
        except ValueError as exc:
            raise _http_error(exc)
        return document_to_response(document)

    @router.get("/{document_id}", response_model=DocumentResponse)
    def get_document(
        document_id: UUID, service: DocumentService = Depends(_document_service)
    ) -> DocumentResponse:
        document = service.get_document(document_id)
        if not document:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
        return document_to_response(document)

    @router.patch("/{document_id}", response_model=DocumentResponse)
    def update_document(
        document_id: UUID,
        payload: DocumentUpdate,
        service: DocumentService = Depends(_document_service),
    ) -> DocumentResponse:
        try:
            document = service.update_document(document_id, payload)
        except ValueError as exc:
            raise _http_error(exc)
        return document_to_response(document)

    @router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_document(document_id: UUID, service: DocumentService = Depends(_document_service)) -> None:
        if not service.delete_document(document_id):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    @router.get("/{document_id}/chunks", response_model=List[ChunkResponse])
    def list_document_chunks(
        document_id: UUID,
        chunk_service: ChunkService = Depends(_chunk_service),
    ) -> List[ChunkResponse]:
        chunks = chunk_service.list_chunks(document_id=document_id)
        return [chunk_to_response(chunk) for chunk in chunks]

    return router


def _chunk_router() -> APIRouter:
    router = APIRouter(prefix="/chunks", tags=["chunks"])

    @router.post("", response_model=ChunkResponse, status_code=status.HTTP_201_CREATED)
    def create_chunk(
        payload: ChunkCreate, service: ChunkService = Depends(_chunk_service)
    ) -> ChunkResponse:
        try:
            chunk = service.create_chunk(payload)
        except ValueError as exc:
            raise _http_error(exc)
        return chunk_to_response(chunk)

    @router.get("/{chunk_id}", response_model=ChunkResponse)
    def get_chunk(
        chunk_id: UUID, service: ChunkService = Depends(_chunk_service)
    ) -> ChunkResponse:
        chunk = service.get_chunk(chunk_id)
        if not chunk:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chunk not found")
        return chunk_to_response(chunk)

    @router.patch("/{chunk_id}", response_model=ChunkResponse)
    def update_chunk(
        chunk_id: UUID,
        payload: ChunkUpdate,
        service: ChunkService = Depends(_chunk_service),
    ) -> ChunkResponse:
        try:
            chunk = service.update_chunk(chunk_id, payload)
        except ValueError as exc:
            raise _http_error(exc)
        return chunk_to_response(chunk)

    @router.delete("/{chunk_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_chunk(chunk_id: UUID, service: ChunkService = Depends(_chunk_service)) -> None:
        if not service.delete_chunk(chunk_id):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chunk not found")

    return router


def _http_error(exc: ValueError) -> HTTPException:
    detail = str(exc)
    status_code = status.HTTP_404_NOT_FOUND if "not found" in detail.lower() else status.HTTP_400_BAD_REQUEST
    return HTTPException(status_code=status_code, detail=detail)


__all__ = ["create_app", "ServiceContainer"]
