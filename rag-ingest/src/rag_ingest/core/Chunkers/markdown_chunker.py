"""
Module for chunking and converting text to markdown using LLM assistance.
"""

import re

from rag_ingest.core.Chunkers.base_chunker import BaseChunker
from llama_index.llms.openai import OpenAI


class MarkdownChunker(BaseChunker):
    """
    Chunker that converts text to markdown and splits it into chunks based on headers and size.
    """
    def __init__(self, settings, max_chunk_chars=4000, min_chunk_chars=200):
        self.settings = settings
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars
        self.llm = OpenAI(api_key=settings.openai_api_key, model="gpt-4.1", request_timeout=120)

    def chunk(self, text):
        markdown = self._to_markdown(text)
        chunks = self._recursive_chunk(markdown)
        for i, chunk in enumerate(chunks):
            print(f"------------------------------------Chunk {i+1}: ", chunk)
        return chunks

    def _to_markdown(self, text):
        # Split text into manageable pieces for LLM
        max_chars = 2000
        pieces = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
        markdown_pieces = []
        for i, piece in enumerate(pieces):
            if i == 0:
                continuation_note = ""
            else:
                continuation_note = (
                    f"This is part {i+1} of a longer document. "
                    "Do not close any sections, lists, or tables if they are not finished. "
                    "Continue naturally from the previous part.\n"
                )
            prompt = (
                f"{continuation_note}"
                "Convert the following text to markdown format. "
                "Use appropriate headers, lists, and formatting where possible. "
                "IMPORTANT: Separate each paragraph with two newlines "
                "(\\n\\n) in the markdown output.\n\n"
                f"Text:\n\"\"\"\n{piece}\n\"\"\""
            )
            response = self.llm.complete(prompt)
            markdown_piece = response.text.strip()
            print(f"[DEBUG] Markdown piece {i+1} length: {len(markdown_piece)}")
            markdown_pieces.append(markdown_piece)
        markdown_full = "\n\n".join(markdown_pieces)
        with open("llm_markdown_output.txt", "w", encoding="utf-8") as f:
            f.write(markdown_full)
        print("[INFO] Markdown output saved to llm_markdown_output.txt")
        return markdown_full

    def _recursive_chunk(self, markdown, level=1):
        header_pattern = re.compile(rf"^{'#'*level} (.+)", re.MULTILINE)
        sections = []
        last_pos = 0
        for match in header_pattern.finditer(markdown):
            if match.start() > last_pos:
                sections.append((last_pos, match.start()))
            last_pos = match.start()
        sections.append((last_pos, len(markdown)))

        chunks = []
        buffer = ""
        for start, end in sections:
            section = markdown[start:end].strip()
            if not section:
                continue
            # If section is just a header, skip it
            if re.match(r"^#{1,6} .*$", section) and "\n" not in section:
                continue
            if len(section) > self.max_chunk_chars and level < 6:
                sub_chunks = self._recursive_chunk(section, level=level+1)
                for sub in sub_chunks:
                    if len(sub) < self.min_chunk_chars:
                        buffer += "\n" + sub
                    else:
                        if buffer:
                            chunks.append(buffer.strip())
                            buffer = ""
                        chunks.append(sub)
            elif len(section) > self.max_chunk_chars:
                paragraphs = [p for p in section.split("\n\n") if p.strip()]
                for p in paragraphs:
                    if len(p) > self.max_chunk_chars:
                        for i in range(0, len(p), self.max_chunk_chars):
                            part = p[i:i+self.max_chunk_chars]
                            if len(part) < self.min_chunk_chars:
                                buffer += "\n" + part
                            else:
                                if buffer:
                                    chunks.append(buffer.strip())
                                    buffer = ""
                                chunks.append(part)
                    else:
                        if len(p) < self.min_chunk_chars:
                            buffer += "\n" + p
                        else:
                            if buffer:
                                chunks.append(buffer.strip())
                                buffer = ""
                            chunks.append(p)
            elif len(section) < self.min_chunk_chars:
                buffer += "\n" + section
            else:
                if buffer:
                    chunks.append(buffer.strip())
                    buffer = ""
                chunks.append(section)
        if buffer:
            chunks.append(buffer.strip())
        return chunks
