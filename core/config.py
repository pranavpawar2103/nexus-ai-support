"""Configuration management for NexusAI Support.

Loads and validates all environment variables using Pydantic BaseSettings.
Provides factory functions for LLM and embedding models.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Attributes:
        openai_api_key: OpenAI API key for LLM and embeddings.
        openai_model: OpenAI model name (default: gpt-4o).
        embedding_model: Embedding model name.
        db_path: Path to SQLite database file.
        chroma_path: Path to ChromaDB persistence directory.
        mcp_host: MCP server host.
        mcp_port: MCP server port.
    """

    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o", description="OpenAI model name")
    embedding_model: str = Field(
        default="text-embedding-3-small", description="Embedding model"
    )
    db_path: str = Field(default="./data/nexus_support.db", description="SQLite DB path")
    chroma_path: str = Field(
        default="./data/chroma_db", description="ChromaDB storage path"
    )
    mcp_host: str = Field(default="localhost", description="MCP server host")
    mcp_port: int = Field(default=8765, description="MCP server port")
    log_level: str = Field(default="INFO", description="Logging level")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @field_validator("openai_api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate that the API key is not the placeholder value."""
        if v == "your_key_here" or not v.startswith("sk-"):
            raise ValueError(
                "OPENAI_API_KEY must be a valid OpenAI API key starting with 'sk-'"
            )
        return v

    @field_validator("mcp_port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port is in valid range."""
        if not (1024 <= v <= 65535):
            raise ValueError("MCP_PORT must be between 1024 and 65535")
        return v

    def get_db_abs_path(self) -> Path:
        """Return absolute path to the database file."""
        return Path(self.db_path).resolve()

    def get_chroma_abs_path(self) -> Path:
        """Return absolute path to ChromaDB directory."""
        return Path(self.chroma_path).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached application settings.

    Returns:
        Settings: Validated application settings instance.

    Raises:
        ValidationError: If required environment variables are missing or invalid.
    """
    settings = Settings()
    logger.info(
        "Settings loaded: model=%s, db=%s", settings.openai_model, settings.db_path
    )
    return settings


def get_llm(temperature: float = 0.1, streaming: bool = False):
    """Factory function returning configured ChatOpenAI instance.

    Args:
        temperature: Sampling temperature (0.0 for deterministic, 1.0 for creative).
        streaming: Enable token streaming.

    Returns:
        ChatOpenAI: Configured language model instance.
    """
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=temperature,
        streaming=streaming,
        api_key=settings.openai_api_key,
    )


def get_embeddings():
    """Factory function returning configured OpenAIEmbeddings instance.

    Returns:
        OpenAIEmbeddings: Configured embeddings model.
    """
    from langchain_openai import OpenAIEmbeddings

    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )


def setup_logging() -> None:
    """Configure application-wide logging with consistent formatting."""
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
    logging.getLogger("posthog").setLevel(logging.CRITICAL)