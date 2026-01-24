"""
ChunksAI module.

Implements an AI-based chunker that partitions text semantically using OpenAI models.
"""

import time
from openai import OpenAIError
from llama_index.llms.openai import OpenAI
from rag_ingest.core.Chunkers.base_chunker import BaseChunker


class ChunksAI(BaseChunker):
    """AI-powered chunker implementing semantic paragraph chunking."""

    @staticmethod
    def paragraph_chunking(text: str, chunk_size: int = 1024) -> list[str]:
        """
        Split text into semantic chunks using an LLM.

        Args:
            text (str): The input text to chunk.
            chunk_size (int): Max size of each chunk in characters.

        Returns:
            List of chunked text pieces.
        """
        chunk_separator = "~" * 25
        questions_separator = "^" * 25
        max_tokens = 32000
        system_prompt = (
            "Process the whole text, regardless if you think it is similar, "
            "go through the whole text, don't be lazy."
        )

        llm = OpenAI(
            system_prompt=system_prompt,
            model="o4-mini",
            max_tokens=max_tokens,
            request_timeout=300,
        )

        prompt = (
            f"Partition semantically the following text while respecting these rules:\n"
            f"- preserve the original text for each chunk\n"
            f"- each chunk should have a chunk size of max {chunk_size} characters\n"
            f"- chunks should contain complete phrases (don't split in the middle)\n"
            f"- if a chunk would be too large, break it into smaller chunks\n"
            f"- separate chunks with {chunk_separator}\n"
            f"- for each chunk, figure out and prepend the questions it answers (max 5); "
            f"separate the questions from the rest of the chunk text with {questions_separator}\n\n"
            f"Text:\n{text}"
        )

        try:
            response = llm.complete(prompt)
            chunks = response.text.split(chunk_separator)
        except Exception as exc:
            print(f"Error during chunking: {exc}")
            time.sleep(2)
            chunks = []

        for i, chunk in enumerate(chunks, start=1):
            print(f"🔹 Chunk {i}:\n{chunk}\n{'-'*40}")

        return chunks

    @staticmethod
    def paragraph_chunking_old(text: str, chunk_size: int = 512) -> list[str]:
        """
        Older paragraph chunking method (deprecated).

        Args:
            text (str): The input text.
            chunk_size (int): Max chunk size.

        Returns:
            List of chunked text pieces.
        """
        llm = OpenAI(model="gpt-4.1", request_timeout=300)

        chunks = []
        raw_chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

        for raw in raw_chunks:
            prompt = (
                f"Partition semantically the following text while respecting these rules:\n"
                f"- each chunk should have max {chunk_size} characters\n"
                f"- chunks should contain complete phrases (don't split in the middle)\n"
                f"- if a chunk would be too large, break it into smaller chunks\n"
                f"- separate chunks with \n---\n\n"
                f"- for each chunk, figure out and prepend the questions it answers (max 5)\n"
                f"Text:\n{raw}"
            )

            try:
                response = llm.complete(prompt)
                sub_chunks = response.text.split("\n---\n")
                chunks.extend([c.strip() for c in sub_chunks if c.strip()])
            except OpenAIError as e:
                print(f"OpenAI API error: {e}")
            except TimeoutError as e:
                print(f"Timeout error: {e}")

        for i, chunk in enumerate(chunks, start=1):
            print(f"🔹 Chunk {i}:\n{chunk}\n{'-'*40}")

        return chunks

    def chunk(self, text: str) -> list[str]:
        """
        Implements the abstract method to chunk text.

        Args:
            text (str): The input text.

        Returns:
            List of chunked text pieces.
        """
        return self.paragraph_chunking(text)
