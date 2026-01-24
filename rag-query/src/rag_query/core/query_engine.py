"""Query engine for interacting with ChromaDB and OpenAI LLM."""

import re
from openai import OpenAI
from chromadb import PersistentClient

from rag_query.core.config import Settings


class Query:
    """Provides methods to query a ChromaDB collection and generate answers using OpenAI models."""

    def __init__(self):
        """Initialize the ChromaDB client and load configuration settings."""
        settings = Settings()
        self.chroma_client = PersistentClient(path=settings.chroma_persist_directory)
        self.collection = self.chroma_client.get_or_create_collection("documents")
        self.openai_api_key = settings.openai_api_key
        self.openai_model_name = settings.openai_model_name
        print("Using Chroma DB at:", settings.chroma_persist_directory)

    def embed_question(self, question: str):
        """Return the vector embedding of a given question string."""
        client = OpenAI(api_key=self.openai_api_key)
        response = client.embeddings.create(
            input=question,
            model="text-embedding-ada-002",
        )
        return response.data[0].embedding

    def get_relevant_chunks(self, question: str, top_k: int = 5):
        """
        Retrieve the most relevant chunks for a given question.

        Falls back to semantic search if no keyword matches are found.
        """
        stopwords = {
            "the", "and", "is", "in", "to", "of", "a", "for", "on", "with",
            "as", "by", "an", "at", "from", "that", "this", "it", "be", "are",
            "or", "was", "but", "not", "have", "has", "had", "were", "which",
            "can", "will", "would", "should", "could", "may", "might", "do",
            "does", "did", "so", "if", "then", "than", "about", "into", "out",
            "up", "down", "over", "under", "again", "more", "most", "some",
            "such", "no", "nor", "only", "own", "same", "too", "very", "s",
            "t", "just", "don", "now"
        }
        qwords = re.findall(r'\b\w+\b', question.lower())
        qwords = [w for w in qwords if w not in stopwords and len(w) > 2]

        all_data = self.collection.get()
        all_docs = all_data.get("documents", [])
        all_metas = all_data.get("metadatas", [])
        all_ids = all_data.get("ids", [])
        all_distances = [None] * len(all_docs)

        matching_docs, matching_metas, matching_ids = [], [], []

        for doc, meta, cid in zip(all_docs, all_metas, all_ids):
            if not doc or not meta or "keywords" not in meta:
                continue
            if any(qk in meta["keywords"] for qk in qwords):
                matching_docs.append(doc)
                matching_metas.append(meta)
                matching_ids.append(cid)

        if not matching_docs:
            embedding = self.embed_question(question)
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=top_k
            )
            return {
                "documents": results.get("documents", [[]])[0],
                "metadatas": results.get("metadatas", [[]])[0],
                "ids": results.get("ids", [[]])[0],
                "distances": results.get("distances", [[]])[0],
            }

        return {
            "documents": matching_docs,
            "metadatas": matching_metas,
            "ids": matching_ids,
            "distances": all_distances[:len(matching_docs)],
        }

    def get_answer(self, question: str, top_k: int = 5):
        """Generate an answer to the question using relevant chunks and the OpenAI model."""
        chunks = self.get_relevant_chunks(question, top_k=top_k)
        context = "\n\n".join(doc for doc in chunks["documents"] if doc)
        prompt = (
            "Given the following context, answer the question as accurately "
            "and concisely as possible.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )
        client = OpenAI(api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model=self.openai_model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that answers questions "
                        "and gives context based on provided information."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
