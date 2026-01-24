"""Configuration settings for rag-query application."""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    # Vector DB connection (shared with rag-ingest)
    chroma_persist_directory: str = Field(
        default="C:\\Miruna\\RAG\\rag-ingest\\src\\vector_db", env="CHROMA_PERSIST_DIRECTORY"
    )
    chroma_collection_name: str = Field(
        default="documents", env="CHROMA_COLLECTION_NAME"
    )

    # OpenAI/LLM configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_project_id: str = Field(..., env="OPENAI_PROJECT_ID") 
    openai_model_name: str = Field(default="gpt-4.1", env="OPENAI_MODEL_NAME")

    # Query configuration
    max_retrieved_docs: int = Field(default=5, env="MAX_RETRIEVED_DOCS")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")

    class Config:
        """Pydantic configuration for environment variable loading."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
