from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Set, Tuple
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float32]
DistanceResult = Tuple[UUID, float]


class VectorIndex(Protocol):
    """Protocol describing the minimal operations of a vector index."""

    @property
    def dimension(self) -> int: ...

    def add_vector(self, chunk_id: UUID, vector: Vector) -> None: ...

    def update_vector(self, chunk_id: UUID, vector: Vector) -> None: ...

    def remove_vector(self, chunk_id: UUID) -> None: ...

    def search(
        self,
        query_vector: Vector,
        k: int,
        metric: str,
        allowed_ids: Optional[Set[UUID]] = None,
    ) -> List[DistanceResult]: ...


class FlatIndex:
    """Brute-force index that compares a query vector with every stored vector."""

    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError("Embedding dimension must be positive")
        self._dimension = dimension
        self._vectors: Dict[UUID, Vector] = {}

    @property
    def dimension(self) -> int:
        return self._dimension

    def add_vector(self, chunk_id: UUID, vector: Vector) -> None:
        self._vectors[chunk_id] = self._normalize_vector(vector)

    def update_vector(self, chunk_id: UUID, vector: Vector) -> None:
        if chunk_id not in self._vectors:
            raise ValueError(f"Chunk {chunk_id} not found in index")
        self._vectors[chunk_id] = self._normalize_vector(vector)

    def remove_vector(self, chunk_id: UUID) -> None:
        self._vectors.pop(chunk_id, None)

    def search(
        self,
        query_vector: Vector,
        k: int,
        metric: str,
        allowed_ids: Optional[Set[UUID]] = None,
    ) -> List[DistanceResult]:
        if k <= 0 or not self._vectors:
            return []

        query = self._normalize_vector(query_vector)
        candidates: Iterable[Tuple[UUID, Vector]]
        if allowed_ids is not None:
            candidates = (
                (chunk_id, vector)
                for chunk_id, vector in self._vectors.items()
                if chunk_id in allowed_ids
            )
        else:
            candidates = self._vectors.items()

        distances = [
            (chunk_id, self._distance(metric, vector, query))
            for chunk_id, vector in candidates
        ]
        if not distances:
            return []
        distances.sort(key=lambda result: result[1])
        return distances[: min(k, len(distances))]

    def _normalize_vector(self, vector: Vector) -> Vector:
        array = np.asarray(vector, dtype=np.float32)
        if array.shape != (self._dimension,):
            raise ValueError(
                f"Vector dimension mismatch: expected {self._dimension}, got {array.shape[0]}"
            )
        return array

    def _distance(self, metric: str, vector: Vector, query: Vector) -> float:
        if metric == "cosine":
            return self._cosine_distance(vector, query)
        if metric == "euclidean":
            return self._euclidean_distance(vector, query)
        if metric == "dot_product":
            return self._dot_product_distance(vector, query)
        raise ValueError(f"Unsupported metric: {metric}")

    @staticmethod
    def _cosine_distance(vector: Vector, query: Vector) -> float:
        vector_norm = np.linalg.norm(vector)
        query_norm = np.linalg.norm(query)
        if vector_norm == 0.0 or query_norm == 0.0:
            return float("inf")
        similarity = float(np.dot(vector, query) / (vector_norm * query_norm))
        return 1.0 - similarity

    @staticmethod
    def _euclidean_distance(vector: Vector, query: Vector) -> float:
        return float(np.linalg.norm(vector - query))

    @staticmethod
    def _dot_product_distance(vector: Vector, query: Vector) -> float:
        return -float(np.dot(vector, query))

    def save(self, path: Path) -> None:
        """Save the index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "dimension": self._dimension,
            "vector_ids": [str(vid) for vid in self._vectors.keys()],
        }
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(metadata, f)

        # Save vectors as numpy array
        if self._vectors:
            vector_array = np.stack(list(self._vectors.values()))
            np.save(path.with_suffix(".npy"), vector_array)

    @classmethod
    def load(cls, path: Path) -> "FlatIndex":
        """Load the index from disk."""
        path = Path(path)

        # Load metadata
        with open(path.with_suffix(".json"), "r") as f:
            metadata = json.load(f)

        # Create index
        index = cls(dimension=metadata["dimension"])

        # Load vectors if they exist
        npy_path = path.with_suffix(".npy")
        if npy_path.exists() and metadata["vector_ids"]:
            vector_array = np.load(npy_path)
            for vid_str, vector in zip(metadata["vector_ids"], vector_array):
                index._vectors[UUID(vid_str)] = vector

        return index


@dataclass(slots=True)
class RandomProjectionIndex:
    """
    Approximate index that buckets vectors using random hyperplanes (LSH style).
    Falls back to exhaustive search when few candidates are found.
    """

    dimension: int
    num_projections: int = 8
    random_state: Optional[int] = None
    _projections: NDArray[np.float32] = field(init=False, repr=False)
    _vectors: Dict[UUID, Vector] = field(init=False, default_factory=dict, repr=False)
    _buckets: Dict[int, Set[UUID]] = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("Embedding dimension must be positive")
        if self.num_projections <= 0:
            raise ValueError("num_projections must be positive")
        rng = np.random.default_rng(self.random_state)
        self._projections = rng.normal(
            size=(self.num_projections, self.dimension)
        ).astype(np.float32)

    def add_vector(self, chunk_id: UUID, vector: Vector) -> None:
        normalized = self._normalize_vector(vector)
        self._vectors[chunk_id] = normalized
        bucket = self._bucket_for(normalized)
        self._buckets.setdefault(bucket, set()).add(chunk_id)

    def update_vector(self, chunk_id: UUID, vector: Vector) -> None:
        if chunk_id not in self._vectors:
            raise ValueError(f"Chunk {chunk_id} not found in index")
        self.remove_vector(chunk_id)
        self.add_vector(chunk_id, vector)

    def remove_vector(self, chunk_id: UUID) -> None:
        self._vectors.pop(chunk_id, None)
        for ids in self._buckets.values():
            ids.discard(chunk_id)

    def search(
        self,
        query_vector: Vector,
        k: int,
        metric: str,
        allowed_ids: Optional[Set[UUID]] = None,
    ) -> List[DistanceResult]:
        if k <= 0 or not self._vectors:
            return []

        query = self._normalize_vector(query_vector)
        bucket = self._bucket_for(query)
        candidates = self._buckets.get(bucket, set()).copy()

        if allowed_ids is not None:
            candidates &= allowed_ids

        if not candidates or len(candidates) < k:
            candidates = set(self._vectors.keys()) if allowed_ids is None else allowed_ids.copy()

        if not candidates:
            return []

        distances = [
            (chunk_id, self._distance(metric, self._vectors[chunk_id], query))
            for chunk_id in candidates
        ]
        distances.sort(key=lambda result: result[1])
        return distances[: min(k, len(distances))]

    def _bucket_for(self, vector: Vector) -> int:
        signs = np.dot(self._projections, vector) >= 0
        hash_value = 0
        for bit, sign in enumerate(signs):
            if sign:
                hash_value |= 1 << bit
        return hash_value

    def _normalize_vector(self, vector: Vector) -> Vector:
        array = np.asarray(vector, dtype=np.float32)
        if array.shape != (self.dimension,):
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, got {array.shape[0]}"
            )
        return array

    @staticmethod
    def _distance(metric: str, vector: Vector, query: Vector) -> float:
        if metric == "cosine":
            return FlatIndex._cosine_distance(vector, query)
        if metric == "euclidean":
            return FlatIndex._euclidean_distance(vector, query)
        if metric == "dot_product":
            return FlatIndex._dot_product_distance(vector, query)
        raise ValueError(f"Unsupported metric: {metric}")

    def save(self, path: Path) -> None:
        """Save the index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save metadata and buckets
        metadata = {
            "dimension": self.dimension,
            "num_projections": self.num_projections,
            "random_state": self.random_state,
            "vector_ids": [str(vid) for vid in self._vectors.keys()],
            "buckets": {
                str(bucket_id): [str(vid) for vid in ids]
                for bucket_id, ids in self._buckets.items()
            },
        }
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(metadata, f)

        # Save projection matrix
        np.save(path.with_suffix(".projections.npy"), self._projections)

        # Save vectors
        if self._vectors:
            vector_array = np.stack(list(self._vectors.values()))
            np.save(path.with_suffix(".npy"), vector_array)

    @classmethod
    def load(cls, path: Path) -> "RandomProjectionIndex":
        """Load the index from disk."""
        path = Path(path)

        # Load metadata
        with open(path.with_suffix(".json"), "r") as f:
            metadata = json.load(f)

        # Create index (this will regenerate projections, which we'll overwrite)
        index = cls(
            dimension=metadata["dimension"],
            num_projections=metadata["num_projections"],
            random_state=metadata["random_state"],
        )

        # Load the actual projection matrix
        index._projections = np.load(path.with_suffix(".projections.npy"))

        # Load vectors if they exist
        npy_path = path.with_suffix(".npy")
        if npy_path.exists() and metadata["vector_ids"]:
            vector_array = np.load(npy_path)
            for vid_str, vector in zip(metadata["vector_ids"], vector_array):
                index._vectors[UUID(vid_str)] = vector

        # Restore buckets
        index._buckets = {
            int(bucket_id): {UUID(vid) for vid in ids}
            for bucket_id, ids in metadata["buckets"].items()
        }

        return index


__all__ = ["VectorIndex", "FlatIndex", "RandomProjectionIndex", "Vector", "DistanceResult"]
