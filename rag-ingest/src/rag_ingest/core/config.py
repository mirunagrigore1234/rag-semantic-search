"""
Configuration settings module for rag_ingest.core.

Defines application settings using Pydantic and environment variables.
"""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables or defaults."""

    storage_dir: str = Field(default="./storage", env="STORAGE_DIR")
    meta_path: str = Field(default="./storage/meta.json", env="META_PATH")

    chroma_persist_directory: str = Field(
        default="./vector_db", env="CHROMA_PERSIST_DIRECTORY"
    )
    chroma_collection_name: str = Field(
        default="documents", env="CHROMA_COLLECTION_NAME"
    )

    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_project_id: str = Field(..., env="OPENAI_PROJECT_ID") 
    openai_model_name: str = Field(default="gpt-4.1", env="OPENAI_MODEL_NAME")

    chunk_size: int = Field(default=3072, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")

    class Config:
        """Pydantic configuration for environment file settings."""
        env_file = "../../.env"
        env_file_encoding = "utf-8"
        extra = "ignore"


def load_config() -> Settings:
    """Load and return application settings."""
    return Settings()
