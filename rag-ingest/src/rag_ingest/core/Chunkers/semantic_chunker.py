"""
Module for semantic chunking using embedding-based splitting.
"""

from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from rag_ingest.core.Chunkers.base_chunker import BaseChunker


class SemanticChunker(BaseChunker):
    """
    Chunker that uses semantic splitting based on embeddings
    to divide text into meaningful chunks.
    """

    def __init__(self, settings):
        self.settings = settings
        self.splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=97,
            embed_model=OpenAIEmbedding(),
        )

    def chunk(self, text: str):
        """
        Split text into semantically meaningful chunks.

        Args:
            text (str): The input text to chunk.

        Returns:
            List[str]: List of chunked text strings.
        """
        document = Document(text=text)

        nodes = self.splitter.get_nodes_from_documents([document])
        print(type(nodes[0]))

        for i, chunk in enumerate(nodes, 1):
            print(f"🔹 Chunk {i}:\n{chunk.get_content(text)}\n{'-'*40}")

        chunks = [str(node.get_content(text)) for node in nodes]
        print(type(chunks[0]))

        return chunks
