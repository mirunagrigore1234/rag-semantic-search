"""
Module providing a RecursiveChunker for splitting text into chunks
using a recursive character-based splitter.
"""

from rag_ingest.core.Chunkers.base_chunker import BaseChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter


class RecursiveChunker(BaseChunker):
    """
    Chunker that splits text recursively using character-based separators
    and manages chunk sizes with overlaps.
    """

    def __init__(self, settings):
        self.settings = settings
        self.splitter = RecursiveCharacterTextSplitter(
            separators=[r"\n#\s", r"\n##\s", r"\n###\s", r"\n####\s", r"\n#####\s"],
            is_separator_regex=True,
            chunk_size=4000,
            chunk_overlap=200,
        )

    def chunk(self, text):
        """
        Split the given text into chunks based on recursive separators.

        Args:
            text (str): The input text to split.

        Returns:
            List[str]: List of text chunks.
        """
        chunks = self.splitter.split_text(text)

        for i, chunk in enumerate(chunks, 1):
            print(f"🔹 Chunk {i}:\n{chunk}\n{'-'*40}")
        return chunks
