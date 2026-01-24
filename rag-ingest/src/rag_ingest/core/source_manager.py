"""
Module for managing storage and indexing of documents.
"""

import json
import os
from typing import List


class SourceManager:
    """
    Manages storage and indexing of documents in a specified directory.

    Attributes:
        storage_dir (str): Directory where documents and metadata are stored.
        meta_path (str): Path to the JSON file storing document indices.
    """

    def __init__(self, storage_dir: str):
        """
        Initializes SourceManager.

        Args:
            storage_dir (str): Directory to store documents and metadata.
        """
        self.storage_dir = storage_dir
        self.meta_path = os.path.join(storage_dir, "document_index.json")
        os.makedirs(self.storage_dir, exist_ok=True)

        if not os.path.exists(self.meta_path):
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def store_document(self, file_path: str, document_id: str) -> bool:
        """
        Stores a document's path in the index if it exists and is not already indexed.

        Args:
            file_path (str): Path to the document file.
            document_id (str): Unique ID for the document.

        Returns:
            bool: True if document stored successfully, False otherwise.
        """
        if not os.path.exists(file_path):
            return False

        with open(self.meta_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        if document_id in index:
            return False  # Already indexed

        index[document_id] = {"path": file_path}

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

        return True

    def delete_document(self, document_id: str) -> bool:
        """
        Deletes a document entry from the index.

        Args:
            document_id (str): Unique ID of the document to delete.

        Returns:
            bool: True if deletion successful, False otherwise.
        """
        if not os.path.exists(self.meta_path):
            return False

        with open(self.meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if document_id not in data:
            return False

        del data[document_id]

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return True

    def list_documents(self) -> List[dict]:
        """
        Lists all indexed documents sorted by document_id.

        Returns:
            List[dict]: List of dictionaries with 'document_id' and file info.
        """
        if not os.path.exists(self.meta_path):
            print("No documents found")
            return []

        with open(self.meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        document_list = [{"document_id": doc_id, **info} for doc_id, info in data.items()]
        document_list.sort(key=lambda x: x["document_id"])

        return document_list
