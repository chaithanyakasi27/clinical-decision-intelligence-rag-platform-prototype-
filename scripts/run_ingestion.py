#!/usr/bin/env python3
# scripts/run_ingestion.py
# ============================================================
# Full ingestion pipeline:
# PDFs → parse → chunk → embed → FAISS index
# Run with: make ingest  OR  python scripts/run_ingestion.py
# ============================================================

import sys
import argparse
from pathlib import Path
from collections import Counter

# Add project root to Python path so import from src/work correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from src.config import settings
from src.ingestion.pdf_parser import ClinicalPDFParser
from src.ingestion.chunker import ClinicalChunker
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.embeddings.vector_store import get_vector_store

def run_ingestion(
    notes_dir: Path = settings.clinical_notes_dir,
    force_rebuild: bool = False,
) -> None:
    """
    Main ingestion pipeline for Clinical Decision Intelligence system.

    Steps:
    1. Parse clinical PDF documents
    2. Split documents into chunks
    3. Generate vector embeddings
    4. Store embeddings in FAISS vector database
    """

    # Print Pipeline header for better readability
    logger.info("=" * 60)
    logger.info("Clinical Decision Intelligence - Ingestion Pipeline")
    logger.info("=" * 60)

    # ----------------------------------------------------------
    # Check if FAISS index already exists
    # Skip rebuild unless --force flag is passed
    # ----------------------------------------------------------
    index_file = settings.faiss_index_path / "index.faiss"

    if index_file.exists() and not force_rebuild:
        logger.info(f"FAISS index already exists at {settings.faiss_index_path}")
        logger.info("Pass --force to rebuild. Exiting.")
        return

    # ----------------------------------------------------------
    # Step 1: Parse PDFs
    # Read all clinical PDFs from input directory
    # ----------------------------------------------------------
    logger.info(f"\n[1/4] Parsing PDFs from {notes_dir}...")

    # Validate input directory exists
    if not notes_dir.exists():
        logger.error(f"Directory not found: {notes_dir}")
        logger.error("Run 'make data' first.")
        sys.exit(1)

    # Initialize PDF parser and parse all files
    parser = ClinicalPDFParser()
    documents = parser.parse_directory(notes_dir)
    
    # Exit if no documents were successfully parsed
    if not documents:
        logger.error("No documents parsed — check clinical_notes/ directory")
        sys.exit(1)

    logger.info(f"  ✓ Parsed {len(documents)} documents")

    # ----------------------------------------------------------
    # Step 2: Chunk documents
    # Split long text into smaller overlapping chunks
    # for better embedding quality and retrieval
    # ----------------------------------------------------------
    logger.info(
        f"\n[2/4] Chunking (size={settings.chunk_size}, overlap={settings.chunk_overlap})..."
    )

    chunker = ClinicalChunker()
    chunks = chunker.chunk_many(documents)
    
    logger.info(f"  ✓ Created {len(chunks)} chunks")
    
    # Count top sections for debugging and validation
    section_counts = Counter(c.section or "unknown" for c in chunks)

    for section, count in section_counts.most_common(5):
        logger.info(f"     {section:35s} {count:4d} chunks")
    
    # ----------------------------------------------------------
    # Step 3: Generate embeddings
    # Convert text chunks into dense vector representations
    # using OpenAI embedding model
    # ----------------------------------------------------------
    logger.info(
        f"\n[3/4] Generating embeddings ({settings.openai_embedding_model})..."
    )
    
    embedder = EmbeddingGenerator()
    embeddings = embedder.embed_chunks(chunks)

    logger.info(f"  ✓ Embeddings shape: {embeddings.shape}")
    # ----------------------------------------------------------
    # Step 4: Build FAISS index
    # Store chunk embeddings inside vector database
    # for fast semantic similarity search
    # ----------------------------------------------------------
    logger.info(
        f"\n[4/4] Building FAISS index → {settings.faiss_index_path}..."
    )

    # Initialize vector store and insert data
    store = get_vector_store()
    store.add_chunks(chunks, embeddings)

    # Persist FAISS index and metadata to disk
    store.save()

    logger.info(f"  ✓ Index saved — {store.count} vectors")

    # Final success message
    logger.info("\n" + "=" * 60)
    logger.info("  Ingestion complete. Run 'make run' to start the API.")
    logger.info("=" * 60)

# --------------------------------------------------------------
# Script entry point
# Allows execution from terminal with optional arguments
# Example:
# python ingest.py --force
# python ingest.py --dir ./custom_notes
# --------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run clinical ingestion pipeline"
    )

    # Force rebuild of existing FAISS index
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild FAISS index"
    )

    # Override default clinical notes directory
    parser.add_argument(
        "--dir",
        type=str,
        help="Override clinical notes directory"
    )

    args = parser.parse_args()

    # Use provided directory if passed, else use default config path
    notes_dir = Path(args.dir) if args.dir else settings.clinical_notes_dir

    # Execute ingestion pipeline
    run_ingestion(
        notes_dir=notes_dir,
        force_rebuild=args.force
    )