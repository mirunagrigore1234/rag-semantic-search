"""Module for managing and querying the vector database."""

import copy
import re
from collections import Counter

from chromadb import PersistentClient
from chromadb.errors import ChromaError
from openai import OpenAI
from openai import (
APIError,
APIConnectionError,
RateLimitError,
AuthenticationError,
BadRequestError)
from rag_ingest.core.config import Settings
from rag_ingest.core.Chunkers.chunks_ai import ChunksAI
from rag_ingest.core.Chunkers.llm_chunker import LLMChunker
from rag_ingest.core.Chunkers.semantic_chunker import SemanticChunker
from rag_ingest.core.Chunkers.markdown_chunker import MarkdownChunker
from rag_ingest.core.Chunkers.recursive_chunker import RecursiveChunker


class VectorDB:
    """Class for storing and managing vectors in a database."""

    def add_document(self, document_id: str, text: str, metadata: dict, method) -> bool:
        """Adds a new document's chunks and embeddings to the vector database."""
        try:
            settings = Settings()
            print(f"ChromaDB location: {settings.chroma_persist_directory}")
            chroma_client = PersistentClient(path=settings.chroma_persist_directory)
            chroma_collection = chroma_client.get_or_create_collection("documents")

            # Chunk the text using the chosen method
            if int(method) == 1:
                texts = ChunksAI().chunk(text)
            elif int(method) == 2:
                texts = LLMChunker(settings).chunk(text)
            elif int(method) == 3:
                texts = SemanticChunker(settings).chunk(text)
            elif int(method) == 4:
                texts = MarkdownChunker(settings).chunk(text)
            else:
                texts = RecursiveChunker(settings).chunk(text)

            # Generate metadata with top keywords
            stopwords = {
                "the", "and", "is", "in", "to", "of", "a", "for", "on", "with", "as", "by", "an",
                "at", "from", "that", "this", "it", "be", "are", "or", "was", "but", "not", "have",
                "has", "had", "were", "which", "can", "will", "would", "should", "could", "may",
                "might", "do", "does", "did", "so", "if", "then", "than", "about", "into", "out",
                "up", "down", "over", "under", "again", "more", "most", "some", "such", "no", "nor",
                "only", "own", "same", "too", "very", "s", "t", "just", "don", "now"
            }

            metadatas = []
            for chunk in texts:
                words = re.findall(r'\b\w+\b', chunk.lower())
                words = [w for w in words if w not in stopwords and len(w) > 2]
                common = [w for w, _ in Counter(words).most_common(5)]
                meta = copy.deepcopy(metadata)
                meta["keywords"] = ", ".join(common)
                metadatas.append(meta)
                print("[DEBUG] Metadata for chunk:", meta)

            ids = [f"{document_id}_chunk_{i}" for i in range(len(texts))]
            text_strings = list(texts)

            # Generate embeddings
            client = OpenAI(api_key=settings.openai_api_key)
            response = client.embeddings.create(
                input=text_strings,
                model="text-embedding-ada-002",
            )
            embeddings = [item.embedding for item in response.data]

            # Store in ChromaDB
            chroma_collection.add(
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
                documents=text_strings
            )

            return True

        except AuthenticationError as e:
            print(f"[ERROR] OpenAI authentication failed: {e}")
        except BadRequestError as e:
            print(f"[ERROR] OpenAI invalid request: {e}")
        except (APIConnectionError, RateLimitError) as e:
            print(f"[ERROR] OpenAI API connection or rate limit issue: {e}")
        except APIError as e:
            print(f"[ERROR] General OpenAI API error: {e}")
        return False

    def delete_document(self, document_id: str) -> bool:
        """Deletes all chunks for a document from the vector database."""
        try:
            settings = Settings()
            chroma_client = PersistentClient(path=settings.chroma_persist_directory)
            chroma_collection = chroma_client.get_or_create_collection("documents")

            all_ids = chroma_collection.get()["ids"]
            matching_ids = [
                doc_id for doc_id in all_ids if f"{document_id}_chunk_" in doc_id
            ]

            chroma_collection.delete(ids=matching_ids)
            return True

        except ValueError as e:
            print(f"[ERROR] Invalid value: {e}")
        except ChromaError as e:
            print(f"[ERROR] ChromaDB operation failed: {e}")
        return False

    @staticmethod
    def get_document_count() -> int:
        """Returns the number of unique documents stored in the vector database."""
        try:
            settings = Settings()
            chroma_client = PersistentClient(path=settings.chroma_persist_directory)
            chroma_collection = chroma_client.get_collection("documents")

            result = chroma_collection.get()
            ids = result.get("ids", [])

            document_ids = {doc_id.split("_chunk_")[0] for doc_id in ids if "_chunk_" in doc_id}
            return len(document_ids)

        except ChromaError as e:
            print(f"[ERROR] ChromaDB operation failed: {e}")
        except ValueError as e:
            print(f"[ERROR] Invalid value: {e}")
        return 0
