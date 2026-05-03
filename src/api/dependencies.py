# src/api/dependencies.py
# ============================================================
# FastAPI dependency injection — singleton pipeline
#
# What this does:
#   Creates the RAG pipeline ONCE when the first request
#   comes in, then reuses it for all subsequent requests.
#
# Why singleton:
#   Loading the FAISS index and BGE model takes ~3 seconds.
#   We do not want to reload them on every API request.
#   The singleton pattern means load once, serve forever.
# ============================================================

from __future__ import annotations
from loguru import logger

# Global singleton — None until first request
_rag_pipeline = None


def get_rag_pipeline():
    """
    Return the singleton ClinicalRAGPipeline.

    Loads FAISS index + initializes retriever on first call.
    All subsequent calls return the already-loaded pipeline.
    """
    global _rag_pipeline

    # Return existing pipeline if already loaded
    if _rag_pipeline is not None:
        return _rag_pipeline

    logger.info("Initializing RAG pipeline (first request)...")

    # ── Step 1: Load vector store ─────────────────────────────
    from src.embeddings.vector_store import get_vector_store, FAISSVectorStore
    vector_store = get_vector_store()

    try:
        vector_store.load()
        logger.info(f"FAISS index loaded: {vector_store.count} vectors")
    except FileNotFoundError:
        logger.warning(
            "FAISS index not found — run scripts/run_ingestion.py first. "
            "Starting with empty index."
        )

    # ── Step 2: Get chunks from the loaded store ───────────────
    # The chunks list is stored alongside the FAISS index in memory.
    # The retriever needs them to return DocumentChunk objects.
    all_chunks = []
    if isinstance(vector_store, FAISSVectorStore):
        all_chunks = vector_store._chunks

    # ── Step 3: Build embedding generator ─────────────────────
    # Reuse the same BGE model for query embedding at search time
    from src.embeddings.embedding_generator import EmbeddingGenerator
    embedder = EmbeddingGenerator()

    # ── Step 4: Build hybrid retriever ────────────────────────
    from src.retrieval.retriever import HybridRetriever
    retriever = HybridRetriever(
        vector_store = vector_store,
        all_chunks   = all_chunks,
        embedder     = embedder,
    )

    # ── Step 5: Build RAG pipeline ────────────────────────────
    from src.rag.pipeline import ClinicalRAGPipeline
    _rag_pipeline = ClinicalRAGPipeline(retriever=retriever)

    logger.info(
        f"RAG pipeline ready — "
        f"{vector_store.count} vectors, "
        f"{len(all_chunks)} chunks"
    )

    return _rag_pipeline