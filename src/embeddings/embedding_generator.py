# src/embeddings/embedding_generator.py
# ============================================================
# Embedding Generator — GPU-accelerated with RTX 4070
#
# Model: BAAI/bge-m3
#   - 1024 dimensional embeddings (vs 384 for bge-small)
#   - Runs on CUDA GPU — ~25x faster than CPU
#   - 2.2GB download, uses ~5GB VRAM
#   - Best open-source embedding model available
#
# Hardware: NVIDIA RTX 4070 Laptop (8GB VRAM)
#   - CUDA 12.x required
#   - Leaves ~3GB VRAM free for other tasks
#
# Fallback: automatically uses CPU if GPU not available
# ============================================================

from __future__ import annotations

import numpy as np
from loguru import logger

from src.config import settings
from src.ingestion.chunker import DocumentChunk


class EmbeddingGenerator:
    """
    GPU-accelerated embedding generator using BAAI/bge-m3.

    Automatically detects and uses CUDA GPU if available.
    Falls back to CPU gracefully if GPU is not found.

    Usage:
        gen = EmbeddingGenerator()

        # Check what device is being used
        print(gen.device)   # "cuda" or "cpu"

        # Embed chunks for ingestion
        vectors = gen.embed_chunks(chunks)
        # vectors.shape == (len(chunks), 1024)

        # Embed a query for retrieval
        query_vec = gen.embed_query("diabetes HCC coding")
        # query_vec.shape == (1024,)
    """

    # Best model for RTX 4070 — 1024 dims, excellent quality
    MODEL_NAME = "BAAI/bge-m3"

    def __init__(self, batch_size: int = settings.embedding_batch_size):
        """
        Args:
            batch_size: chunks per GPU forward pass.
                        RTX 4070 handles 64 comfortably.
                        Increase to 128 if you have headroom.
        """
        self.batch_size = batch_size
        self._model     = None   # lazy load on first use
        self._device    = None   # detected at load time

    @property
    def device(self) -> str:
        """Returns 'cuda' if GPU available, 'cpu' otherwise."""
        if self._device is None:
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._device = "cpu"
        return self._device

    @property
    def model(self):
        """
        Lazy-load the BGE-M3 model onto GPU.

        First call downloads ~2.2GB and loads to GPU VRAM.
        All subsequent calls reuse the loaded model.
        Takes ~20 seconds on first run.
        """
        if self._model is None:
            logger.info(f"Loading model: {self.MODEL_NAME}")
            logger.info(f"Device: {self.device.upper()}")

            if self.device == "cuda":
                import torch
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"GPU: {gpu_name} ({vram_gb:.1f}GB VRAM)")

            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.MODEL_NAME,
                device=self.device,
            )
            logger.info(
                f"Model loaded on {self.device.upper()} — "
                f"embedding dim: {self.embedding_dim}"
            )

        return self._model

    @property
    def embedding_dim(self) -> int:
        """BGE-M3 produces 1024-dimensional vectors."""
        return 1024

    def embed_chunks(self, chunks: list[DocumentChunk]) -> np.ndarray:
        """
        Embed DocumentChunks for ingestion into FAISS.

        Args:
            chunks: list from chunker.py

        Returns:
            float32 array shape (len(chunks), 1024)
        """
        # to_embedding_text() prepends section name
        # e.g. "[Section: Assessment And Plan]\nE11.9 diabetes..."
        texts = [chunk.to_embedding_text() for chunk in chunks]
        return self.embed_texts(texts)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed raw text strings in GPU-accelerated batches.

        With RTX 4070:
            batch_size=32  → ~0.3s per batch
            200 chunks     → ~2s total (vs ~50s on CPU)

        Args:
            texts: list of strings

        Returns:
            float32 array shape (len(texts), 1024)
        """
        if not texts:
            logger.warning("embed_texts called with empty list")
            return np.array([])

        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        logger.info(
            f"Embedding {len(texts)} texts on {self.device.upper()} "
            f"— {total_batches} batches of {self.batch_size}"
        )

        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch     = texts[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1

            logger.debug(
                f"  Batch {batch_num}/{total_batches} "
                f"({len(batch)} texts)"
            )

            # encode() automatically runs on self.device (GPU/CPU)
            # normalize_embeddings=True → unit vectors → cosine via dot product
            embeddings = self.model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
                # GPU-specific: larger batch size fits more in VRAM
                batch_size=self.batch_size,
            )
            all_embeddings.extend(embeddings)

        result = np.array(all_embeddings, dtype=np.float32)
        logger.info(
            f"Embedding complete — shape: {result.shape}, "
            f"device: {self.device.upper()}"
        )
        return result

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single search query for real-time retrieval.

        BGE models use a special prefix for queries to improve
        retrieval quality (asymmetric embedding).

        Args:
            query: user question from API request

        Returns:
            float32 array shape (1024,)
        """
        # BGE-M3 query prefix improves retrieval accuracy
        prefixed = (
            f"Represent this sentence for searching "
            f"relevant passages: {query}"
        )

        result = self.model.encode(
            [prefixed],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return result[0].astype(np.float32)

    def get_gpu_memory_usage(self) -> str:
        """
        Helper to check VRAM usage during development.

        Usage:
            gen = EmbeddingGenerator()
            gen.embed_chunks(chunks)
            print(gen.get_gpu_memory_usage())
            # "VRAM: 4.8GB used / 8.0GB total (60%)"
        """
        if self.device != "cuda":
            return "Running on CPU — no GPU memory stats"

        import torch
        used  = torch.cuda.memory_allocated(0) / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        pct   = (used / total) * 100
        return f"VRAM: {used:.1f}GB used / {total:.1f}GB total ({pct:.0f}%)"