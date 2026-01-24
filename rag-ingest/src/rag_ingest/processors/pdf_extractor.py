"""
Module to extract text from PDF files by converting pages to images
and extracting markdown via VisionExtractor.
"""

import os
from typing import List

import fitz  # PyMuPDF

from .base_extractor import BaseExtractor
from .vision_extractor import VisionExtractor


class PDFExtractor(BaseExtractor):
    """
    Extractor class to process PDFs by rendering pages as images
    and converting them into markdown text.
    """

    def extract(self, file_path: str, batch_size: int = 4) -> str:
        """
        Extract text from a PDF file.

        Args:
            pdf_path (str): Path to the PDF file.
            batch_size (int): Number of images to process per batch.

        Returns:
            str: Extracted markdown text.
        """
        output_path = r"C:\Miruna\RAG\rag-data\llm_markdown_output.txt"
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                return f.read()

        # Convert PDF pages to images
        image_paths: List[str] = []
        doc = fitz.open(file_path)
        zoom_x = 2.0
        zoom_y = 2.0
        mat = fitz.Matrix(zoom_x, zoom_y)
        for page_number, page in enumerate(doc):
            pix = page.get_pixmap(matrix=mat)
            image_path = f"page-{page_number + 1}.png"
            pix.save(image_path)
            image_paths.append(image_path)

        markdown = VisionExtractor.extract_images_to_markdown(image_paths, batch_size=batch_size)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        return markdown
