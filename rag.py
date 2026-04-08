from __future__ import annotations
import json
import re
import uuid
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI


class DocumentStore:
    """
    RAG store backed by a FAISS index.

    Documents are chunked by paragraph, embedded with OpenAI's
    text-embedding-3-small, and retrieved via cosine similarity using
    a FAISS IndexFlatIP index (inner-product on L2-normalised vectors).

    Parameters
    ----------
    client:
        An OpenAI client instance used for embedding calls.
    persist_dir:
        Optional directory for on-disk persistence.  When supplied the
        FAISS index is written to ``<persist_dir>/index.faiss`` and
        metadata to ``<persist_dir>/meta.json`` after every ``add_text``
        call.  Pass None (default) for a purely in-memory store.
    """

    EMBED_MODEL = "text-embedding-3-small"

    def __init__(self, client: OpenAI, persist_dir: str | None = None) -> None:
        self.client = client
        self._persist_dir = Path(persist_dir) if persist_dir else None
        self._dim: int | None = None
        self._index: faiss.IndexFlatIP | None = None
        self.chunks: list[str] = []
        self.metadatas: list[dict] = []

        if self._persist_dir and self._persist_dir.exists():
            self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_text(self, text: str, metadata: dict | None = None) -> int:
        """Chunk *text*, embed it, and add to the store. Returns chunks added."""
        new_chunks = self._chunk(text)
        if not new_chunks:
            return 0

        vecs = self._embed(new_chunks)          # (n, dim), float32, L2-normalised
        self._ensure_index(vecs.shape[1])
        self._index.add(vecs)

        meta = metadata or {}
        self.chunks.extend(new_chunks)
        self.metadatas.extend([meta] * len(new_chunks))

        if self._persist_dir:
            self._save()

        return len(new_chunks)

    def add_file(self, path: str | Path) -> int:
        """Read a .txt file and ingest it. Returns chunks added."""
        p = Path(path)
        text = p.read_text(encoding="utf-8", errors="replace")
        return self.add_text(text, metadata={"filename": p.name, "filepath": str(p)})

    def search(self, query: str, k: int = 3) -> list[dict]:
        """Return the top-k most relevant chunks for *query*."""
        if self.is_empty:
            return []

        q_vec = self._embed([query])            # (1, dim), already normalised
        top_k = min(k, len(self.chunks))
        scores, indices = self._index.search(q_vec, top_k)

        return [
            {
                "chunk": self.chunks[idx],
                "score": float(scores[0][rank]),
                "metadata": self.metadatas[idx],
            }
            for rank, idx in enumerate(indices[0])
            if idx != -1
        ]

    def format_results(self, query: str, k: int = 3) -> str:
        """Return a formatted string of top-k results, ready for tool output."""
        results = self.search(query, k=k)
        if not results:
            return "No relevant documents found in the local corpus."
        parts = [f"[Relevance {r['score']:.2f}] {r['chunk']}" for r in results]
        return "\n\n---\n\n".join(parts)

    def clear(self) -> None:
        """Remove all documents and reset the index."""
        self._index = None
        self._dim = None
        self.chunks = []
        self.metadatas = []

    @property
    def is_empty(self) -> bool:
        return self._index is None or self._index.ntotal == 0

    def __len__(self) -> int:
        return 0 if self._index is None else self._index.ntotal

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._persist_dir / "index.faiss"))
        (self._persist_dir / "meta.json").write_text(
            json.dumps({"chunks": self.chunks, "metadatas": self.metadatas}),
            encoding="utf-8",
        )

    def _load(self) -> None:
        index_path = self._persist_dir / "index.faiss"
        meta_path = self._persist_dir / "meta.json"
        if not index_path.exists() or not meta_path.exists():
            return
        self._index = faiss.read_index(str(index_path))
        self._dim = self._index.d
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        self.chunks = data["chunks"]
        self.metadatas = data["metadatas"]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_index(self, dim: int) -> None:
        if self._index is None:
            self._dim = dim
            self._index = faiss.IndexFlatIP(dim)

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised embeddings, shape (len(texts), dim), float32."""
        response = self.client.embeddings.create(model=self.EMBED_MODEL, input=texts)
        vecs = np.array([e.embedding for e in response.data], dtype=np.float32)
        faiss.normalize_L2(vecs)
        return vecs

    def _chunk(self, text: str, max_words: int = 300) -> list[str]:
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
                overlap = current[-1:]
                current = overlap
                current_len = len(overlap[0].split()) if overlap else 0
            current.append(para)
            current_len += para_len

        if current:
            chunks.append(" ".join(current))

        return chunks
