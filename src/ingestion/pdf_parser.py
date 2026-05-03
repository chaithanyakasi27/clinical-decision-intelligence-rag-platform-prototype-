# ============================================================
# Clinical PDF Parser using PyMuPDF (fitz)
# 
# What this does:
#   1. Opens a PDF file from disk
#   2. Extracts all text from every page
#   3. Detects clinical note sections (HPI, Assessment, etc.)
#   4. Returns a ParsedDocument with structured text
#
# Why section detection matters:
#   The "Assessment and Plan" section contains ICD-10 codes
#   which are critical for HCC coding. We want the chunker
#   to keep this section's content together.
# ============================================================
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF — opens and reads PDF files
from loguru import logger

# ── Output dataclass ─────────────────────────────────────────
# This is what the parser returns for each PDF file.
# The chunker.py receives a list of these objects.

@dataclass
class ParsedDocument:
    """
    Represents one parsed clinical PDF document.
    
    Attributes:
        document_id   : unique SHA256 hash of the file content
        file_name     : original PDF filename  
        file_path     : full path to the PDF on disk
        full_text     : all text extracted from every page
        page_count    : how many pages the PDF has
        metadata      : dict of PDF properties (author, title etc)
        sections      : dict mapping section name → section text
                        e.g. {"assessment and plan": "1. E11.9 diabetes..."}
    """
    document_id : str
    file_name   : str
    file_path   : str
    full_text   : str
    page_count  : int
    metadata    : dict = field(default_factory=dict)
    sections    : dict[str, str] = field(default_factory=dict)

# ── Clinical section headers to detect ───────────────────────
# These are the standard SOAP note sections found in clinical PDFs.
# When the parser sees one of these lines it starts a new section.
# Order matters — more specific phrases should come first.

CLINICAL_SECTIONS = [
    "assessment and plan",       # most important for HCC coding
    "assessment",
    "plan",
    "history of present illness",
    "hpi",
    "past medical history",
    "pmh",
    "current medications",
    "medications",
    "allergies",
    "review of systems",
    "ros",
    "physical examination",
    "physical exam",
    "chief complaint",
    "follow-up plan",
    "follow up",
    "follow-up",
    "diagnosis",
    "icd",
    "hcc risk",                  # most important for HCC coding
    "discharge summary",
]

class ClinicalPDFParser:
    """
    Parses clinical PDF documents into structured text.

    Usage:
        parser = ClinicalPDFParser()
        
        # Parse a single file
        doc = parser.parse("data/clinical_notes/patient_001.pdf")
        print(doc.sections["assessment and plan"])
        
        # Parse all PDFs in a folder
        docs = parser.parse_directory("data/clinical_notes/")
        print(f"Parsed {len(docs)} documents")
    """

    def parse(self, file_path: str | Path) -> ParsedDocument:
        """
        Parse a single PDF file into a ParsedDocument.
        
        Args:
            file_path: path to the PDF file
            
        Returns:
            ParsedDocument with full text and detected sections
            
        Raises:
            FileNotFoundError: if the PDF does not exist
        """
        path = Path(file_path)

        # Check file exists before trying to open it
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        logger.debug(f"Parsing: {path.name}")

        # Generate a unique ID by hashing the file content
        # This means the same file always gets the same ID
        raw_bytes   = path.read_bytes()
        document_id = hashlib.sha256(raw_bytes).hexdigest()[:16]

        # Open the PDF with PyMuPDF and extract text page by page
        with fitz.open(str(path)) as pdf:
            pages_text = []
            for page_num, page in enumerate(pdf):
                # "text" mode extracts plain text (no formatting)
                text = page.get_text("text")
                if text.strip():  # skip blank pages
                    pages_text.append(text)

            # Join all pages into one string
            full_text  = "\n".join(pages_text).strip()
            page_count = len(pdf)

            # Extract PDF metadata (author, title, creator app etc)
            metadata = self._extract_metadata(pdf)

        # Detect and split clinical note sections
        sections = self._extract_sections(full_text)

        logger.debug(
            f"  {path.name}: {page_count} pages, "
            f"{len(full_text)} chars, "
            f"{len(sections)} sections detected"
        )

        return ParsedDocument(
            document_id = document_id,
            file_name   = path.name,
            file_path   = str(path),
            full_text   = full_text,
            page_count  = page_count,
            metadata    = metadata,
            sections    = sections,
        )

    def parse_directory(self, directory: str | Path) -> list[ParsedDocument]:
        """
        Parse all PDF files found in a directory.
        
        Args:
            directory: folder containing PDF files
            
        Returns:
            List of ParsedDocument objects (one per PDF)
        """
        directory = Path(directory)
        pdf_files = list(directory.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in {directory}")
            return []

        logger.info(f"Parsing {len(pdf_files)} PDFs from {directory}")

        docs   = []
        failed = 0

        for pdf_path in sorted(pdf_files):  # sort for reproducible order
            try:
                doc = self.parse(pdf_path)
                docs.append(doc)
            except Exception as e:
                # Log the error but continue processing other files
                logger.error(f"Failed to parse {pdf_path.name}: {e}")
                failed += 1

        logger.info(
            f"Parsed {len(docs)}/{len(pdf_files)} PDFs "
            f"({failed} failed)"
        )
        return docs

    # ── Private helper methods ────────────────────────────────

    def _extract_metadata(self, pdf: fitz.Document) -> dict:
        """
        Extract standard metadata from the PDF file properties.
        This is the metadata set by the app that created the PDF
        (e.g. ReportLab sets 'creator' to 'ReportLab PDF Library').
        """
        meta = pdf.metadata or {}
        return {
            "title"      : meta.get("title", ""),
            "author"     : meta.get("author", ""),
            "subject"    : meta.get("subject", ""),
            "creator"    : meta.get("creator", ""),
            "page_count" : len(pdf),
        }

    def _extract_sections(self, text: str) -> dict[str, str]:
        """
        Split clinical note text into sections by detecting headers.
        
        Strategy:
          - Split text into lines
          - For each line, check if it matches a known section header
          - If yes: save the previous section, start collecting new one
          - At end: save the last section
          
        Returns:
            dict mapping lowercase section name → section content text
            Example: {"assessment and plan": "1. Type 2 diabetes E11.9..."}
        """
        sections: dict[str, str]  = {}
        current_section: Optional[str] = None
        current_lines  : list[str]     = []

        for line in text.split("\n"):
            # Normalize line for comparison:
            # strip whitespace, lowercase, remove trailing colon
            line_normalized = line.strip().lower().rstrip(":")

            # Try to match this line against known section headers
            matched_section = self._match_section(line_normalized)

            if matched_section:
                # Save the section we were collecting before
                if current_section and current_lines:
                    sections[current_section] = "\n".join(current_lines).strip()

                # Start collecting the new section
                current_section = matched_section
                current_lines   = []
            else:
                # This line belongs to the current section
                if current_section:
                    current_lines.append(line)

        # Don't forget to save the very last section
        if current_section and current_lines:
            sections[current_section] = "\n".join(current_lines).strip()

        return sections

    def _match_section(self, line: str) -> Optional[str]:
        """
        Check if a line is a clinical section header.
        
        Args:
            line: normalized (lowercase, stripped) line of text
            
        Returns:
            The matched section name, or None if no match
        """
        for section in CLINICAL_SECTIONS:
            # Exact match OR line starts with the section name
            # e.g. "assessment and plan (continued)" still matches
            if line == section or line.startswith(section):
                return section
        return None