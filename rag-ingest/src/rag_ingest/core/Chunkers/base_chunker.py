"""
BaseChunker module.

Defines the abstract base class for chunking text into smaller pieces.
"""

from abc import ABC, abstractmethod


class BaseChunker(ABC):
    """Abstract base class for all chunkers."""

    @abstractmethod
    def chunk(self, text: str):
        """
        Abstract method to split text into chunks.

        Args:
            text (str): The input text to chunk.

        Returns:
            list[str]: A list of text chunks.
        """
        pass
