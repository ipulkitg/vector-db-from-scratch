"""Configuration management for the vector database."""
from __future__ import annotations

import logging
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

    # Logging configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

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

    def configure_logging(self) -> None:
        """Configure application logging based on settings."""
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format=self.log_format,
            force=True,  # Override any existing configuration
        )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module."""
    return logging.getLogger(f"vector_db.{name}")


__all__ = ["Settings", "get_logger"]
