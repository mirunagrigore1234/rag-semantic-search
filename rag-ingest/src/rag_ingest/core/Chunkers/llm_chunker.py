"""
Module for chunking text with LLM assistance.
"""

import json
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.llms.openai import OpenAI
from rag_ingest.core.Chunkers.base_chunker import BaseChunker


class LLMChunker(BaseChunker):
    """
    Chunker that splits text into chunks and uses an LLM to semantically arrange them.
    """

    def __init__(self, settings):
        self.settings = settings

    def _raw_chunk(self, text, chunk_size=500, chunk_overlap=0):
        splitter = SentenceSplitter.from_defaults(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return splitter.split_text(text)

    def _arrange_chunks_with_llm(self, chunks, batch_size=20):
        client = OpenAI(api_key=self.settings.openai_api_key)
        all_arranged = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            prompt = (
                "Given the following list of text chunks (each up to 500 characters), "
                "rearrange, merge, or improve them for optimal semantic flow. "
                "Return the improved list as a JSON array of strings, "
                "each no longer than 500 characters. "
                "Do not add or remove information, just improve the order and flow."
                "\n\nChunks:\n"
                + str(batch)
            )
            try:
                response = client.complete(prompt)
                content = response.text
                if content.strip().startswith("{"):
                    obj = json.loads(content)
                    for v in obj.values():
                        if isinstance(v, list):
                            all_arranged.extend(v)
                            break
                    else:
                        all_arranged.extend(obj)
                else:
                    all_arranged.extend(json.loads(content))
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"Error parsing LLM response: {e}")
                all_arranged.extend(batch)
        return all_arranged

    def chunk(self, text):
        """
        Split and arrange text into semantically meaningful chunks using LLM.
        """
        print("sunt la chunking")
        chunks = self._raw_chunk(text)
        arranged_chunks = self._arrange_chunks_with_llm(chunks)
        return arranged_chunks
