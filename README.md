# RAG Query

A **Retrieval-Augmented Generation (RAG)** module for **semantic search and question answering** over vectorized documents.

This project queries a vector database populated by a companion ingestion pipeline, retrieves the most relevant document chunks using semantic similarity, and generates answers using Large Language Models (LLMs).


## Features
- Semantic search over documents using vector embeddings  
- Retrieval-Augmented Generation (RAG) pipeline  
- Command-line interface (CLI) for querying and diagnostics  
- Modular, extensible, and production-oriented architecture  


## How It Works
1. A user submits a query via the CLI  
2. Relevant documents are retrieved from a vector database using similarity search  
3. Retrieved context is injected into a prompt  
4. An LLM generates a final response based on the retrieved information  


## Tech Stack
- **Python**
- **OpenAI API**
- **ChromaDB** (vector database)
- **Pydantic / Pydantic-Settings**
- **Click** (CLI framework)

---

## Project Structure
rag-query/
├── src/rag_query/
│ ├── core/
│ │ ├── query_engine.py
│ │ ├── retriever.py
│ │ ├── generator.py
│ │ └── vector_db.py
│ └── cli.py
├── tests/
└── pyproject.toml

## Installation

```bash
git clone https://github.com/your-username/rag-query.git
cd rag-query
pipenv install

# Run a full RAG query (retrieval + generation)
pipenv run python -m rag_query.cli query "What is the main topic of the documents?"

# Perform semantic search only (no generation)
pipenv run python -m rag_query.cli search "machine learning"
Environment Configuration

Create a .env file in the project root:

OPENAI_PROJECT_ID=project_id
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL_NAME=gpt-4

CHROMA_PERSIST_DIRECTORY=./vector_db
CHROMA_COLLECTION_NAME=documents

MAX_RETRIEVED_DOCS=5
SIMILARITY_THRESHOLD=0.7

Usage

# Run a full RAG query
rag-query query "What is the main topic of the documents?"

# Perform similarity search only
rag-query search "machine learning"

# Check system status
rag-query status

Integration

This module is designed to work with a separate RAG ingestion pipeline, sharing the same vector database persistence directory and collection.

Notes

This project focuses on clean architecture, separation of concerns, and best practices for building RAG systems suitable for real-world applications.
