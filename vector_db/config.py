"""Configuration management for the vector database."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Storage configuration
    storage_type: Literal["memory", "disk"] = "memory"
    data_dir: Path = Path("./data")

    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def ensure_data_dir(self) -> None:
        """Create data directory if it doesn't exist."""
        if self.storage_type == "disk":
            self.data_dir.mkdir(parents=True, exist_ok=True)
            (self.data_dir / "libraries").mkdir(exist_ok=True)
            (self.data_dir / "documents").mkdir(exist_ok=True)
            (self.data_dir / "chunks").mkdir(exist_ok=True)
            (self.data_dir / "indexes").mkdir(exist_ok=True)


__all__ = ["Settings"]
