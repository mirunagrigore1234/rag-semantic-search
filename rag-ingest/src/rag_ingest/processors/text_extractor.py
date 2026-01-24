"""
Module for extracting plain text files in rag_ingest.processors.
"""

from .base_extractor import BaseExtractor


class TextExtractor(BaseExtractor):
    """
    Extractor for plain text files.
    """

    def extract(self, file_path: str) -> str:
        """
        Extract text from a plain text file.

        Args:
            file_path (str): Path to the text file.

        Returns:
            str: Extracted text content.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
