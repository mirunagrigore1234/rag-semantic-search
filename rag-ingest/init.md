# RAG Ingester Module Initialization

## Directory Structure
```
rag-ingest/
├── pyproject.toml
├── Pipfile
├── .env
├── .env.example
├── src/
│   └── rag_ingest/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py
│       │   ├── content_extractor.py
│       │   ├── source_manager.py
│       │   ├── vector_db.py
│       │   └── ingestion_pipeline.py
│       └── processors/
│           ├── __init__.py
│           ├── text_extractor.py
│           ├── vision_extractor.py
│           └── pdf_extractor.py
└── tests/
    └── __init__.py
```

## Class Specifications

### Core Classes

#### `ContentExtractor` (`src/rag_ingest/core/content_extractor.py`)
- **Purpose**: Routes documents to appropriate processors based on file type
- **Methods**:
  - `extract_text(file_path: str) -> str`: Main extraction method
  - `get_processor(file_type: str) -> BaseExtractor`: Returns appropriate processor
  - `supported_formats() -> List[str]`: Lists supported file formats

#### `SourceManager` (`src/rag_ingest/core/source_manager.py`)
- **Purpose**: Manages storage and retrieval of original source documents
- **Methods**:
  - `store_document(file_path: str, document_id: str) -> bool`: Store original document
  - `retrieve_document(document_id: str) -> bytes`: Get original document
  - `delete_document(document_id: str) -> bool`: Remove document from storage
  - `list_documents() -> List[dict]`: List all stored documents with metadata

#### `VectorDB` (`src/rag_ingest/core/vector_db.py`)
- **Purpose**: Manages vector database operations and embeddings
- **Methods**:
  - `add_document(document_id: str, text: str, metadata: dict) -> bool`: Add document to vector DB
  - `delete_document(document_id: str) -> bool`: Remove document from vector DB
  - `get_document_count() -> int`: Get total document count

#### `IngestionPipeline` (`src/rag_ingest/core/ingestion_pipeline.py`)
- **Purpose**: Main orchestrator for the entire ingestion process
- **Methods**:
  - `ingest_document(file_path: str) -> str`: Full ingestion pipeline, returns document ID
  - `delete_document(document_id: str) -> bool`: Remove document and cleanup
  - `delete_document_by_path(file_path: str) -> bool`: Remove document by computing checksum
  - `batch_ingest(file_paths: List[str]) -> List[str]`: Batch processing, returns document IDs
  - `get_pipeline_status() -> dict`: Pipeline health and stats
  - `generate_document_id(file_path: str) -> str`: Generate SHA-256 checksum from file content

### Processor Classes

#### `TextExtractor` (`src/rag_ingest/processors/text_extractor.py`)
- **Purpose**: Process plain text files
- **Methods**:
  - `extract(file_path: str) -> str`: Extract text from file
  - `validate_file(file_path: str) -> bool`: Check if file is valid text

#### `VisionExtractor` (`src/rag_ingest/processors/vision_extractor.py`)
- **Purpose**: Extract text from images using LLM
- **Methods**:
  - `extract(file_path: str) -> str`: Send image to LLM for text extraction
  - `validate_file(file_path: str) -> bool`: Check if file is valid image
  - `supported_formats() -> List[str]`: List supported image formats

#### `PDFExtractor` (`src/rag_ingest/processors/pdf_extractor.py`)
- **Purpose**: Convert PDF to images then extract text
- **Methods**:
  - `extract(file_path: str) -> str`: Convert PDF to images, then extract text
  - `pdf_to_images(pdf_path: str) -> List[str]`: Convert PDF pages to image files
  - `validate_file(file_path: str) -> bool`: Check if file is valid PDF

### CLI Interface

#### `CLI` (`src/rag_ingest/cli.py`)
- **Purpose**: Command-line interface for ingestion operations
- **Commands**:
  - `ingest <file_path>`: Ingest single document, returns document ID
  - `ingest-batch <directory>`: Ingest all documents in directory
  - `delete <file_path>`: Delete document by computing checksum from file path
  - `delete-id <document_id>`: Delete document by checksum ID directly
  - `list`: List all ingested documents with their checksums and metadata
  - `status`: Show pipeline status and stats

#### `__main__.py` updates
- CLI entry point using `click` for command parsing
- Routes commands to appropriate `IngestionPipeline` methods
- Handles file path validation and error reporting

## Document ID Strategy
- **Document IDs**: SHA-256 checksum of entire file content
- **Benefits**: Automatic deduplication, content-based identity, deterministic
- **Implementation**: Read file in chunks, compute hash over entire content
- **CLI workflow**: Users work with file paths, system uses checksums internally

## Setup Steps
1. Create project structure
2. Configure `pyproject.toml` with dependencies
3. Set up `Pipfile` for pipenv
4. Create `.env` file for environment variables
5. Implement configuration loading
6. Add core modules and processors
7. Install in development mode: `pipenv install --dev && pip install -e .`

## Key Dependencies
- `python-dotenv` - environment variables
- `pydantic` - configuration and validation
- `llamaindex` - document processing and embeddings
- `chromadb` - vector storage
- `pymupdf` - PDF to image conversion
- `pillow` - image processing
- `openai` - LLM for vision extraction
- `click` - CLI interface

## Development Dependencies
- `ruff` - fast linting and formatting
- `mypy` - type checking
- `pytest` - testing framework
- `pytest-asyncio` - async testing support
- `pytest-cov` - test coverage

## Configuration
Environment variables loaded from `.env` file using pydantic BaseSettings:
- Vector DB settings (ChromaDB path, collection name)
- LLM settings (OpenAI API key, model names)
- Processing settings (chunk size, batch size)
- Storage settings (document storage path)