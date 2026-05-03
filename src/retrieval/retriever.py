# ============================================================
# Hybrid Retriever — BM25 + Dense Vector + Reranking
#
# What this does:
#   Given a query, finds the most relevant clinical chunks
#   using TWO different search methods then combines them.
#
# Why hybrid (not just vector search):
#   Dense vector search (FAISS) is great for semantic similarity
#   "diabetes management" matches "glycemic control" correctly.
#
#   BM25 is great for exact keyword matches
#   "E11.65" will score high if that exact code is in the text.
#
#   Combining both catches what either alone would miss.
#
# Reciprocal Rank Fusion (RRF):
#   Merges the two ranked lists into one final ranking.
#   A chunk that ranks #2 in both lists beats one that
#   ranks #1 in one list and #20 in the other.
# ============================================================


from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

from src.config import settings
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.embeddings.vector_store import FAISSVectorStore, SearchResult
from src.ingestion.chunker import DocumentChunk


# ── Result type ───────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """
    Final retrieval result after fusion and reranking.

    Attributes:
        chunk       : the DocumentChunk that matched
        dense_score : cosine similarity from FAISS
        bm25_score  : BM25 keyword score
        rrf_score   : Reciprocal Rank Fusion combined score
        final_score : score used for final ranking
        rank        : final position (0 = best)
    """
    chunk      : DocumentChunk
    dense_score: float = 0.0
    bm25_score : float = 0.0
    rrf_score  : float = 0.0
    final_score: float = 0.0
    rank       : int   = 0

    @property
    def text(self) -> str:
        """Shortcut to chunk text."""
        return self.chunk.text

    @property
    def section(self) -> Optional[str]:
        """Shortcut to chunk section."""
        return self.chunk.section


class HybridRetriever:
    """
    Hybrid BM25 + Dense retrieval with Reciprocal Rank Fusion.

    Usage:
        retriever = HybridRetriever(vector_store, chunks, embedder)

        # Retrieve top 5 relevant chunks for a query
        results = retriever.retrieve(
            "Type 2 diabetes HCC coding evidence",
            top_k=5
        )

        for r in results:
            print(f"Rank {r.rank}: [{r.section}] {r.text[:80]}")
    """

    def __init__(
        self,
        vector_store : FAISSVectorStore,
        all_chunks   : list[DocumentChunk],
        embedder     : Optional[EmbeddingGenerator] = None,
        bm25_weight  : float = settings.bm25_weight,
        dense_weight : float = settings.dense_weight,
        use_reranker : bool  = False,
    ):
        """
        Args:
            vector_store : loaded FAISS store for dense retrieval
            all_chunks   : all chunks (needed for BM25 index)
            embedder     : embedding generator for query embedding
            bm25_weight  : how much weight to give BM25 scores (default 0.3)
            dense_weight : how much weight to give vector scores (default 0.7)
            use_reranker : whether to apply cross-encoder reranking
        """
        self.vector_store = vector_store
        self.all_chunks   = all_chunks
        self.embedder     = embedder or EmbeddingGenerator()
        self.bm25_weight  = bm25_weight
        self.dense_weight = dense_weight
        self.use_reranker = use_reranker
        self._bm25        = None

        # Build BM25 index immediately if we have chunks
        if all_chunks:
            self._build_bm25_index()

    def retrieve(
        self,
        query          : str,
        top_k          : int = settings.reranker_top_k,
        retrieval_k    : int = settings.retriever_top_k,
    ) -> list[RetrievalResult]:
        """
        Full retrieval pipeline for a clinical query.

        Steps:
        1. Embed the query using BGE model
        2. Dense search via FAISS (semantic similarity)
        3. BM25 search (keyword matching)
        4. Reciprocal Rank Fusion to merge results
        5. Return top_k results

        Args:
            query      : clinical question or HCC coding request
            top_k      : final number of results to return
            retrieval_k: candidates from each retriever before fusion

        Returns:
            List of RetrievalResult sorted by final_score (best first)
        """
        logger.debug(f"Retrieving for: {query!r} (top_k={top_k})")

        # Handle empty index gracefully
        if not self.all_chunks:
            logger.warning("No chunks available — run ingestion first")
            return []

        # ── Step 1: Dense retrieval ───────────────────────────
        query_embedding = self.embedder.embed_query(query)
        dense_results   = self.vector_store.search(
            query_embedding,
            top_k=retrieval_k,
        )

        # ── Step 2: BM25 retrieval ────────────────────────────
        bm25_results = self._bm25_search(query, top_k=retrieval_k)

        # ── Step 3: Reciprocal Rank Fusion ────────────────────
        fused = self._reciprocal_rank_fusion(dense_results, bm25_results)

        # ── Step 4: Trim to top_k ─────────────────────────────
        fused = fused[:top_k]

        # Assign final ranks
        for i, result in enumerate(fused):
            result.rank = i

        logger.debug(
            f"Retrieved {len(fused)} results. "
            f"Top: [{fused[0].section}] score={fused[0].final_score:.4f}"
            if fused else "No results"
        )

        return fused

    # ── BM25 methods ──────────────────────────────────────────

    def _build_bm25_index(self) -> None:
        """
        Build BM25 index from all chunks.

        BM25 works on tokenized text (simple word splitting).
        This runs once at startup — not on every query.
        """
        tokenized = [
            chunk.text.lower().split()
            for chunk in self.all_chunks
        ]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(
            f"BM25 index built over {len(self.all_chunks)} chunks"
        )

    def _bm25_search(
        self,
        query: str,
        top_k: int,
    ) -> list[SearchResult]:
        """
        BM25 keyword search over all chunks.

        Returns SearchResult objects (same type as dense search)
        so they can be merged in RRF.
        """
        if self._bm25 is None:
            return []

        # Tokenize query the same way we tokenized chunks
        tokenized_query = query.lower().split()
        scores          = self._bm25.get_scores(tokenized_query)

        # Get indices of top scoring chunks
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:  # only include chunks with non-zero score
                results.append(SearchResult(
                    chunk = self.all_chunks[idx],
                    score = float(scores[idx]),
                    rank  = rank,
                ))

        return results

    # ── Reciprocal Rank Fusion ────────────────────────────────

    def _reciprocal_rank_fusion(
        self,
        dense: list[SearchResult],
        bm25 : list[SearchResult],
        k    : int = 60,
    ) -> list[RetrievalResult]:
        """
        Combine dense and BM25 results using Reciprocal Rank Fusion.

        RRF formula for each chunk:
            score = dense_weight * 1/(k + rank_in_dense)
                  + bm25_weight  * 1/(k + rank_in_bm25)

        k=60 is the standard RRF constant that prevents
        top-ranked results from dominating too much.

        Args:
            dense : results from FAISS vector search
            bm25  : results from BM25 keyword search
            k     : RRF smoothing constant

        Returns:
            Merged and re-ranked list of RetrievalResult
        """
        # Use chunk_id as key to merge results from both systems
        chunk_scores: dict[str, RetrievalResult] = {}

        # Add dense retrieval scores
        for rank, result in enumerate(dense):
            cid = result.chunk.chunk_id
            if cid not in chunk_scores:
                chunk_scores[cid] = RetrievalResult(
                    chunk       = result.chunk,
                    dense_score = result.score,
                )
            chunk_scores[cid].rrf_score += (
                self.dense_weight * (1.0 / (k + rank))
            )

        # Add BM25 scores
        for rank, result in enumerate(bm25):
            cid = result.chunk.chunk_id
            if cid not in chunk_scores:
                chunk_scores[cid] = RetrievalResult(
                    chunk      = result.chunk,
                    bm25_score = result.score,
                )
            chunk_scores[cid].rrf_score += (
                self.bm25_weight * (1.0 / (k + rank))
            )
            chunk_scores[cid].bm25_score = result.score

        # Sort by RRF score descending
        fused = sorted(
            chunk_scores.values(),
            key=lambda x: x.rrf_score,
            reverse=True,
        )

        # Set final_score = rrf_score
        for r in fused:
            r.final_score = r.rrf_score

        return fused