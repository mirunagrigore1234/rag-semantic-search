"""
Base extractor module defining the interface for file extractors.
"""

import os


class BaseExtractor:
    """
    Abstract base class for extractors that process files.
    """

    def extract(self, file_path: str) -> str:
        """
        Extract content from the given file path.

        Args:
            file_path (str): Path to the file to extract.

        Returns:
            str: Extracted content as a string.
        """

    def validate_file(self, file_path: str) -> bool:
        """
        Validate that the file exists and is readable.

        Args:
            file_path (str): Path to the file to validate.

        Returns:
            bool: True if file exists and is readable, False otherwise.
        """
        if os.path.exists(file_path) and os.path.isfile(file_path) and os.access(file_path, os.R_OK):
            return True
        print(f"File {file_path} does not exist or is not readable.")
        return False
