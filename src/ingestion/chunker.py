# ============================================================
# Clinical Document Chunker
#
# What this does:
#   Takes a ParsedDocument and splits it into smaller chunks
#   that can be embedded and stored in the vector database.
#
# Why chunking matters:
#   LLMs have context limits. A full clinical note might be
#   3000 words. We split it into ~512 token chunks so each
#   chunk fits in an embedding model and retrieval is precise.
#
# Strategy — Section-aware chunking:
#   If the PDF has detected sections (HPI, Assessment etc),
#   we chunk each section SEPARATELY. This means a chunk
#   from "Assessment and Plan" will NEVER mix with content
#   from "Review of Systems". Much better for retrieval.
#
#   If no sections detected → fall back to splitting full text.
# ============================================================
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from src.config import settings
from src.ingestion.pdf_parser import ParsedDocument

# ── Output dataclass ─────────────────────────────────────────
# Each chunk is one piece of text ready to be embedded.
# The vector store stores these alongside their embeddings.

@dataclass
class DocumentChunk:
    """
    One chunk of text from a clinical document.
    
    Attributes:
        chunk_id      : unique ID e.g. "abc123_chunk_0001"
        document_id   : ID of the parent ParsedDocument
        file_name     : source PDF filename
        text          : the actual text content of this chunk
        chunk_index   : position of this chunk (0, 1, 2...)
        total_chunks  : total chunks from this document
        section       : which clinical section this came from
                        e.g. "assessment and plan"
                        None if document had no sections
        metadata      : extra fields stored in vector DB
    """
    chunk_id    : str
    document_id : str
    file_name   : str
    text        : str
    chunk_index : int
    total_chunks: int
    section     : Optional[str] = None
    metadata    : dict = field(default_factory=dict)

    def to_embedding_text(self) -> str:
        """
        Build the text string that gets sent to the embedding model.
        
        We prepend the section name so the embedding captures
        BOTH the content AND its clinical context.
        
        Example output:
            "[Section: Assessment And Plan]
             1. Type 2 diabetes mellitus E11.65..."
             
        This means similar queries like "what conditions are coded"
        will match Assessment chunks more than HPI chunks.
        """
        if self.section:
            # Title-case the section name for readability
            return f"[Section: {self.section.title()}]\n{self.text}"
        return self.text

# ── Priority sections for HCC coding ─────────────────────────
# These sections contain the most HCC-relevant content.
# We process them first so they get lower chunk indices
# and appear earlier in retrieval results.

PRIORITY_SECTIONS = {
    "assessment and plan",
    "assessment",
    "plan",
    "diagnosis",
    "hcc risk",
    "history of present illness",
    "hpi",
    "past medical history",
    "pmh",
}

class ClinicalChunker:
    """
    Section-aware chunker for clinical PDF documents.
    
    Usage:
        chunker = ClinicalChunker()
        
        # Chunk a single document
        chunks = chunker.chunk(parsed_doc)
        
        # Chunk multiple documents at once
        all_chunks = chunker.chunk_many(list_of_parsed_docs)
        
        print(f"Created {len(all_chunks)} chunks")
        print(f"First chunk section: {all_chunks[0].section}")
        print(f"First chunk text: {all_chunks[0].text[:100]}")
    """
    def __init__(
        self,
        chunk_size   : int = settings.chunk_size,    # default 512 tokens
        chunk_overlap: int = settings.chunk_overlap,  # default 50 tokens
    ):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

        # LangChain's recursive splitter tries to split on
        # paragraph breaks first, then sentences, then words.
        # This keeps natural language boundaries intact.
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size     = chunk_size,
            chunk_overlap  = chunk_overlap,
            separators     = ["\n\n", "\n", ". ", " ", ""],
            length_function= len,
        )
    def chunk(self, doc: ParsedDocument) -> list[DocumentChunk]:
        """
        Chunk a single ParsedDocument into DocumentChunks.
        
        Uses section-aware chunking if sections were detected,
        falls back to full-text chunking otherwise.
        
        Args:
            doc: a ParsedDocument from pdf_parser.py
            
        Returns:
            List of DocumentChunk objects ready for embedding
        """
        if doc.sections:
            # Document has clinical sections — chunk by section
            chunks = self._chunk_by_sections(doc)
        else:
            # No sections detected — chunk the full text
            logger.debug(
                f"{doc.file_name}: no sections detected, "
                f"using full-text chunking"
            )
            chunks = self._chunk_full_text(doc)

        logger.debug(
            f"{doc.file_name}: "
            f"{len(chunks)} chunks created "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks
    
    def chunk_many(self, docs: list[ParsedDocument]) -> list[DocumentChunk]:
        """
        Chunk a list of ParsedDocuments.
        
        This is what run_ingestion.py calls with all 20 PDFs.
        
        Args:
            docs: list of ParsedDocument objects
            
        Returns:
            All chunks from all documents in one flat list
        """
        all_chunks: list[DocumentChunk] = []

        for doc in docs:
            doc_chunks = self.chunk(doc)
            all_chunks.extend(doc_chunks)

        logger.info(
            f"Chunked {len(docs)} documents "
            f"→ {len(all_chunks)} total chunks"
        )
        return all_chunks
    
    # ── Private helpers ───────────────────────────────────────

    def _chunk_by_sections(self, doc: ParsedDocument) -> list[DocumentChunk]:
        """
        Chunk each clinical section independently.
        
        Why independently?
          A chunk will NEVER cross a section boundary.
          "Assessment and Plan" content stays in its own chunks.
          This is critical for HCC coding accuracy.
        """
        all_chunks  : list[DocumentChunk] = []
        chunk_index : int = 0

        # Sort sections: priority sections first, then alphabetical
        # This ensures Assessment chunks have lower indices
        sorted_sections = sorted(
            doc.sections.items(),
            key=lambda x: (
                0 if x[0] in PRIORITY_SECTIONS else 1,
                x[0],  # alphabetical within each priority group
            ),
        )

        for section_name, section_text in sorted_sections:
            # Skip empty sections
            if not section_text.strip():
                continue

            # Split this section's text into chunks
            raw_chunks = self._splitter.split_text(section_text)

            for raw_text in raw_chunks:
                # Skip chunks that are only whitespace
                if not raw_text.strip():
                    continue

                chunk = DocumentChunk(
                    chunk_id    = f"{doc.document_id}_chunk_{chunk_index:04d}",
                    document_id = doc.document_id,
                    file_name   = doc.file_name,
                    text        = raw_text.strip(),
                    chunk_index = chunk_index,
                    total_chunks= 0,  # updated below after we know total
                    section     = section_name,
                    metadata    = {
                        **doc.metadata,
                        "section"  : section_name,
                        "file_path": doc.file_path,
                    },
                )
                all_chunks.append(chunk)
                chunk_index += 1

        # Now we know the total — update every chunk
        total = len(all_chunks)
        for chunk in all_chunks:
            chunk.total_chunks = total

        return all_chunks
    
    def _chunk_full_text(self, doc: ParsedDocument) -> list[DocumentChunk]:
        """
        Fallback: chunk the entire document text without section awareness.
        Used when pdf_parser could not detect any section headers.
        """
        raw_chunks = self._splitter.split_text(doc.full_text)
        chunks     = []

        for i, raw_text in enumerate(raw_chunks):
            if not raw_text.strip():
                continue

            chunk = DocumentChunk(
                chunk_id    = f"{doc.document_id}_chunk_{i:04d}",
                document_id = doc.document_id,
                file_name   = doc.file_name,
                text        = raw_text.strip(),
                chunk_index = i,
                total_chunks= len(raw_chunks),
                section     = None,  # no section info available
                metadata    = {
                    **doc.metadata,
                    "file_path": doc.file_path,
                },
            )
            chunks.append(chunk)

        return chunks
    




