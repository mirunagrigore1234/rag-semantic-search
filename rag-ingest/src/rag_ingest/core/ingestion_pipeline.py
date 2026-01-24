"""
Module implementing the ingestion pipeline for documents,
including storing, vectorizing, and deleting documents.
"""

import hashlib
import os
from typing import List

from rag_ingest.core.config import load_config
from rag_ingest.core.content_extractor import ContentExtractor
from rag_ingest.core.source_manager import SourceManager
from rag_ingest.core.vector_db import VectorDB


class IngestionPipeline:
    """
    Pipeline class to manage ingestion and deletion of documents,
    handling source storage and vector database operations.
    """

    def __init__(self):
        config = load_config()
        self.source_manager = SourceManager(config.storage_dir)

    def generate_document_id(self, file_path: str) -> str:
        """
        Generate a SHA-256 hash based document ID for the file contents.

        Args:
            file_path (str): Path to the file.

        Returns:
            str: SHA-256 hash hex digest as document ID.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    def ingest_document(self, file_path: str, method: int) -> str:
        """
        Ingest a document: store it, extract text, generate embeddings and store vectors.

        Args:
            file_path (str): Path to the document file.
            method (int): Chunking method selector.

        Raises:
            ValueError: If the document already exists in storage.

        Returns:
            str: Document ID generated for the ingested document.
        """
        document_id = self.generate_document_id(file_path)
        if not self.source_manager.store_document(file_path, document_id):
            raise ValueError(f"Document with ID {document_id} already exists.")

        text = ContentExtractor().extract(file_path)
        metadata = {"original_filename": os.path.basename(file_path)}
        vector_db = VectorDB()
        vector_db.add_document(document_id, text, metadata, method)

        return document_id

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from both source storage and vector database.

        Args:
            document_id (str): The document ID to delete.

        Returns:
            bool: True if deletion succeeded in both places, False otherwise.
        """
        source_deleted = self.source_manager.delete_document(document_id)
        vector_db_deleted = VectorDB().delete_document(document_id)
        return source_deleted and vector_db_deleted

    def delete_document_by_path(self, file_path: str) -> bool:
        """
        Delete a document given its file path by generating its ID.

        Args:
            file_path (str): Path to the document file.

        Returns:
            bool: True if deletion succeeded in both places, False otherwise.
        """
        document_id = self.generate_document_id(file_path)
        return self.delete_document(document_id)

    def batch_ingest(self, file_paths: List[str], method: int) -> List[str]:
        """
        Batch ingest multiple documents.

        Args:
            file_paths (List[str]): List of file paths to ingest.
            method (int): Chunking method selector.

        Returns:
            List[str]: List of document IDs successfully ingested.
        """
        document_ids = []
        for file_path in file_paths:
            try:
                document_id = self.ingest_document(file_path, method)
                document_ids.append(document_id)
            except ValueError as e:
                print(f"File {file_path} could not be ingested: {e}")
            except OSError as e:
                print(f"OS error with file {file_path}: {e}")

        return document_ids
