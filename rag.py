from __future__ import annotations
import re
import numpy as np
from openai import OpenAI


class DocumentStore:
    """
    Lightweight in-memory RAG store.

    Documents are chunked by paragraph, embedded with OpenAI's
    text-embedding-3-small, and retrieved via cosine similarity.
    """

    EMBED_MODEL = "text-embedding-3-small"

    def __init__(self, client: OpenAI) -> None:
        self.client = client
        self.chunks: list[str] = []
        self.metadatas: list[dict] = []
        self._embeddings: np.ndarray | None = None  # shape (n_chunks, dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_text(self, text: str, metadata: dict | None = None) -> int:
        """Chunk *text*, embed it, and add to the store. Returns chunks added."""
        new_chunks = self._chunk(text)
        if not new_chunks:
            return 0

        new_embeddings = self._embed(new_chunks)
        meta = metadata or {}

        self.chunks.extend(new_chunks)
        self.metadatas.extend([meta] * len(new_chunks))

        if self._embeddings is None:
            self._embeddings = new_embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, new_embeddings])

        return len(new_chunks)

    def search(self, query: str, k: int = 3) -> list[dict]:
        """Return the top-k most relevant chunks for *query*."""
        if self._embeddings is None or not self.chunks:
            return []

        q_vec = self._embed([query])[0]
        norms = np.linalg.norm(self._embeddings, axis=1)
        q_norm = np.linalg.norm(q_vec)
        scores = self._embeddings @ q_vec / (norms * q_norm + 1e-10)

        top_k = min(k, len(self.chunks))
        indices = np.argsort(scores)[-top_k:][::-1]

        return [
            {
                "chunk": self.chunks[i],
                "score": float(scores[i]),
                "metadata": self.metadatas[i],
            }
            for i in indices
        ]

    def format_results(self, query: str, k: int = 3) -> str:
        """Convenience method — returns a formatted string ready for tool output."""
        results = self.search(query, k=k)
        if not results:
            return "No relevant documents found in the local corpus."
        parts = [
            f"[Relevance {r['score']:.2f}] {r['chunk']}"
            for r in results
        ]
        return "\n\n---\n\n".join(parts)

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    def __len__(self) -> int:
        return len(self.chunks)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _chunk(self, text: str, max_words: int = 300, overlap_words: int = 40) -> list[str]:
        """Split *text* into overlapping paragraph-aware chunks."""
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text.strip()) if p.strip()]
        if not paragraphs:
            return [text.strip()] if text.strip() else []

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for para in paragraphs:
            para_len = len(para.split())
            if current_len + para_len > max_words and current:
                chunks.append(" ".join(current))
                # overlap: keep last paragraph for context continuity
                overlap = current[-1:]
                current = overlap
                current_len = len(overlap[0].split()) if overlap else 0
            current.append(para)
            current_len += para_len

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _embed(self, texts: list[str]) -> np.ndarray:
        response = self.client.embeddings.create(model=self.EMBED_MODEL, input=texts)
        return np.array([e.embedding for e in response.data], dtype=np.float32)
