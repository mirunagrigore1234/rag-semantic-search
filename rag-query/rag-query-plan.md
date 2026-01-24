# RAG Query Project Structure Plan

## Overview
The `rag-query` module will handle querying the vector database populated by `rag-ingest`. It will provide RAG (Retrieval-Augmented Generation) capabilities by retrieving relevant documents and generating responses using LLMs.

## Project Structure

```
rag-query/
├── LICENSE                    # Already exists
├── pyproject.toml            # Python project configuration
├── Pipfile                   # Pipenv dependencies (to match rag-ingest)
├── Pipfile.lock             # Locked dependencies
├── .env                     # Environment variables
├── src/
│   └── rag_query/
│       ├── __init__.py       # Package initializer
│       ├── __main__.py       # CLI entry point
│       ├── cli.py            # Click-based CLI interface
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py     # Configuration management (Pydantic settings)
│       │   ├── query_engine.py    # Main query processing logic
│       │   ├── retriever.py       # Document retrieval from vector DB
│       │   ├── generator.py       # LLM response generation
│       │   └── vector_db.py       # Vector database interface (shared with ingest)
│       └── utils/
│           ├── __init__.py
│           └── formatting.py      # Simple response formatting utilities
└── tests/
    ├── __init__.py
    └── test_query_engine.py
```

## Core Components

### 1. CLI Interface (`cli.py`)
Simple commands to implement:
- `query "question"` - Basic RAG query
- `search "query"` - Similarity search only (no generation)
- `status` - Show query engine status

### 2. Configuration (`config.py`)
Settings loaded from `.env` file:
```python
class Settings(BaseSettings):
    # Vector DB connection (shared with rag-ingest)
    chroma_persist_directory: str = Field(default="./vector_db", env="CHROMA_PERSIST_DIRECTORY")
    chroma_collection_name: str = Field(default="documents", env="CHROMA_COLLECTION_NAME")
    
    # OpenAI/LLM configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model_name: str = Field(default="gpt-4", env="OPENAI_MODEL_NAME")
    
    # Query configuration
    max_retrieved_docs: int = Field(default=5, env="MAX_RETRIEVED_DOCS")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
```

### 3. Query Engine (`query_engine.py`)
Simple orchestrator that:
- Validates incoming queries
- Retrieves relevant documents using the retriever
- Generates responses using the generator
- Returns formatted responses

### 4. Retriever (`retriever.py`)
Basic document retrieval:
- Vector similarity search
- Simple result filtering based on similarity threshold
- Returns top-k documents

### 5. Generator (`generator.py`)
Simple LLM interactions:
- Basic prompt construction
- OpenAI API calls
- Simple error handling

## Dependencies (`pyproject.toml`)

Core dependencies (minimal set):
- `python-dotenv` - Environment configuration
- `pydantic` & `pydantic-settings` - Configuration management
- `chromadb` - Vector database (shared with rag-ingest)
- `openai` - LLM API client
- `click` - CLI framework

Development dependencies:
- `ruff` - Linting and formatting
- `mypy` - Type checking
- `pytest` - Testing framework
- `pytest-asyncio` - Async testing support

## Environment Configuration (`.env`)

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL_NAME=gpt-4

# Vector Database
CHROMA_PERSIST_DIRECTORY=./vector_db
CHROMA_COLLECTION_NAME=documents

# Query Settings
MAX_RETRIEVED_DOCS=5
SIMILARITY_THRESHOLD=0.7
```

## CLI Usage Examples

```bash
# Basic query
rag-query query "What is the main topic of the documents?"

# Search without generation
rag-query search "machine learning"

# Status check
rag-query status
```

## Integration with rag-ingest

The `rag-query` module will:
- Share the same vector database location and collection
- Use compatible configuration for ChromaDB settings
- Reference the same document IDs and metadata structure

## Implementation Guidelines for Interns

### Phase 1: Basic Setup
1. Create `pyproject.toml` with minimal dependencies
2. Set up basic CLI structure with Click
3. Implement configuration loading from `.env`
4. Create basic package structure

### Phase 2: Core Functionality
1. Implement vector database connection
2. Create simple retriever for similarity search
3. Add basic LLM integration
4. Build simple query engine pipeline

### Phase 3: CLI Commands
1. Implement `query` command with basic RAG
2. Add `search` command for retrieval-only
3. Create `status` command for diagnostics

### Phase 4: Testing & Polish
1. Add basic unit tests
2. Error handling and validation
3. Documentation and usage examples

## Shared Components with rag-ingest

To avoid duplication, consider sharing:
- Vector database interface (`vector_db.py`)
- Basic configuration patterns
- Common data models

The interns can copy and adapt the `vector_db.py` from `rag-ingest` as a starting point.