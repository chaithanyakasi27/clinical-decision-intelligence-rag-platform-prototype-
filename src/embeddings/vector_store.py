# src/embeddings/vector_store.py
# ============================================================
# FAISS Vector Store
#
# What this does:
#   Stores embeddings in a FAISS index on disk.
#   Given a query embedding, finds the most similar chunks.
#
# What is FAISS:
#   Facebook AI Similarity Search — a library that does
#   extremely fast nearest-neighbour search in high dimensions.
#   We use it locally instead of Pinecone (which costs money).
#
# Files saved to disk:
#   data/faiss_index/index.faiss  — the FAISS binary index
#   data/faiss_index/chunks.pkl   — the chunk metadata
#
# How search works:
#   1. Query comes in: "what ICD-10 codes are present?"
#   2. Query gets embedded: [0.023, -0.412, ...] (384 numbers)
#   3. FAISS finds the top-K most similar stored vectors
#   4. Returns the corresponding DocumentChunks
# ============================================================

from __future__ import annotations

import pickle
from pathlib import Path
from dataclasses import dataclass

import faiss
import numpy as np
from loguru import logger

from src.config import settings
from src.ingestion.chunker import DocumentChunk


# ── Search result ─────────────────────────────────────────────
# This is what the retriever gets back from vector_store.search()

@dataclass
class SearchResult:
    """
    One result from a vector similarity search.
    
    Attributes:
        chunk : the DocumentChunk that matched the query
        score : cosine similarity score (0.0 to 1.0, higher = better)
        rank  : position in results (0 = best match)
    """
    chunk : DocumentChunk
    score : float
    rank  : int

    def __repr__(self) -> str:
        return (
            f"SearchResult("
            f"rank={self.rank}, "
            f"score={self.score:.4f}, "
            f"section={self.chunk.section!r}, "
            f"text={self.chunk.text[:60]!r}...)"
        )


class FAISSVectorStore:
    """
    Local FAISS vector store for clinical document embeddings.
    
    Workflow:
        # During ingestion (run once):
        store = FAISSVectorStore()
        store.add_chunks(chunks, embeddings)
        store.save()
        
        # During API requests (load once at startup):
        store = FAISSVectorStore()
        store.load()
        results = store.search(query_embedding, top_k=5)
    """

    def __init__(self, index_path: Path = settings.faiss_index_path):
        """
        Args:
            index_path: directory where index.faiss and chunks.pkl are saved
        """
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        # These are None until add_chunks() or load() is called
        self._index : faiss.Index | None      = None
        self._chunks: list[DocumentChunk]     = []

    def add_chunks(
        self,
        chunks    : list[DocumentChunk],
        embeddings: np.ndarray,
    ) -> None:
        """
        Add chunks and their embeddings to the FAISS index.
        
        Args:
            chunks     : list of DocumentChunk objects
            embeddings : numpy array shape (len(chunks), embedding_dim)
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks "
                f"but {len(embeddings)} embeddings"
            )

        # Normalize embeddings to unit length
        # This converts dot product to cosine similarity
        # (BGE already normalizes but we do it again to be safe)
        norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / np.maximum(norms, 1e-8)

        dim = normalized.shape[1]  # embedding dimension (384 for BGE)

        # Create the index if this is the first call
        if self._index is None:
            # IndexFlatIP = Inner Product index
            # On normalized vectors, inner product == cosine similarity
            # "Flat" means exact search (no approximation)
            logger.info(f"Creating FAISS IndexFlatIP (dim={dim})")
            self._index = faiss.IndexFlatIP(dim)

        # Add vectors to the index
        self._index.add(normalized.astype(np.float32))

        # Keep the chunks in memory (parallel to the index)
        self._chunks.extend(chunks)

        logger.info(
            f"Added {len(chunks)} chunks → "
            f"index total: {self._index.ntotal} vectors"
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k          : int = settings.retriever_top_k,
    ) -> list[SearchResult]:
        """
        Find the top_k most similar chunks to a query embedding.
        
        Args:
            query_embedding: 1D numpy array from embed_query()
            top_k          : how many results to return
            
        Returns:
            List of SearchResult sorted by similarity (best first)
        """
        # Guard: can't search an empty index
        if self._index is None or self._index.ntotal == 0:
            logger.warning("FAISS index is empty — run ingestion first")
            return []

        # Normalize query vector
        q    = query_embedding.reshape(1, -1).astype(np.float32)
        norm = np.linalg.norm(q)
        q    = q / max(norm, 1e-8)

        # Don't request more results than we have stored
        k = min(top_k, self._index.ntotal)

        # FAISS search returns:
        #   scores  : shape (1, k) — cosine similarity scores
        #   indices : shape (1, k) — positions in self._chunks
        scores, indices = self._index.search(q, k)

        results = []
        for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0])
        ):
            # idx == -1 means FAISS found fewer results than k
            if idx == -1:
                continue

            results.append(SearchResult(
                chunk = self._chunks[idx],
                score = float(score),
                rank  = rank,
            ))

        return results

    def save(self) -> None:
        """
        Save the FAISS index and chunk list to disk.
        
        Creates two files:
            data/faiss_index/index.faiss  — binary FAISS index
            data/faiss_index/chunks.pkl   — serialized chunk list
        """
        if self._index is None:
            logger.warning("Nothing to save — index is empty")
            return

        index_file  = self.index_path / "index.faiss"
        chunks_file = self.index_path / "chunks.pkl"

        # Save FAISS binary index
        faiss.write_index(self._index, str(index_file))

        # Save chunk metadata using pickle
        with open(chunks_file, "wb") as f:
            pickle.dump(self._chunks, f)

        logger.info(
            f"Saved FAISS index: "
            f"{self._index.ntotal} vectors → {self.index_path}"
        )

    def load(self) -> None:
        """
        Load the FAISS index and chunks from disk.
        
        Called once at API server startup so the index
        is ready in memory for all incoming requests.
        
        Raises:
            FileNotFoundError: if index files don't exist.
            Run 'python scripts/run_ingestion.py' first.
        """
        index_file  = self.index_path / "index.faiss"
        chunks_file = self.index_path / "chunks.pkl"

        if not index_file.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_file}\n"
                f"Run: python scripts/run_ingestion.py"
            )

        # Load FAISS index from binary file
        self._index = faiss.read_index(str(index_file))

        # Load chunk metadata from pickle
        with open(chunks_file, "rb") as f:
            self._chunks = pickle.load(f)

        logger.info(
            f"Loaded FAISS index: "
            f"{self._index.ntotal} vectors, "
            f"{len(self._chunks)} chunks"
        )

    @property
    def count(self) -> int:
        """How many vectors are currently in the index."""
        return self._index.ntotal if self._index else 0


def get_vector_store() -> FAISSVectorStore:
    """
    Factory function — returns the configured vector store.
    
    Currently always returns FAISSVectorStore (local).
    Later: check settings.vector_store_type and return
    Pinecone when VECTOR_STORE_TYPE=pinecone in .env
    
    Usage:
        store = get_vector_store()
        store.load()
        results = store.search(query_embedding)
    """
    logger.info("Using FAISS local vector store")
    return FAISSVectorStore()