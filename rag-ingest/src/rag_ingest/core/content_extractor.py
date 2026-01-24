"""
Module for extracting content from different file types in rag_ingest.core.
"""

from typing import List

from rag_ingest.processors.base_extractor import BaseExtractor
from rag_ingest.processors.pdf_extractor import PDFExtractor
from rag_ingest.processors.text_extractor import TextExtractor
from rag_ingest.processors.vision_extractor import VisionExtractor


class ContentExtractor(BaseExtractor):
    """Extracts textual content from supported file formats."""

    def extract(self, file_path: str) -> str:
        if file_path.endswith(".txt"):
            return TextExtractor().extract(file_path)
        if file_path.endswith(".pdf"):
            return PDFExtractor().extract(file_path)
        if any(file_path.endswith(fmt) for fmt in VisionExtractor.supported_formats()):
            return VisionExtractor.extract(file_path)
        raise ValueError("Unsupported file format")

    @staticmethod
    def supported_formats() -> List[str]:
        """Returns list of supported file formats."""
        return ["txt", "pdf"] + VisionExtractor.supported_formats()
