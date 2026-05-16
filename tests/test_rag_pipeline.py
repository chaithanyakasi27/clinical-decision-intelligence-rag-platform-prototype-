# tests/test_rag_pipeline.py
# Unit tests for ClinicalPDFParser, ClinicalChunker, DocumentChunk
# Run with: python -m pytest tests/test_rag_pipeline.py -v
#
# Design: no real PDF files required.
# - Parser internals (_extract_sections, _match_section) are tested
#   directly with in-memory text strings.
# - DocumentChunk is constructed directly (it is a plain dataclass).
# - ClinicalChunker is tested with a hand-crafted ParsedDocument.

import pytest
from src.ingestion.pdf_parser import ClinicalPDFParser, ParsedDocument, CLINICAL_SECTIONS
from src.ingestion.chunker import ClinicalChunker, DocumentChunk, PRIORITY_SECTIONS


# ── Fixtures ──────────────────────────────────────────────────

SAMPLE_CLINICAL_TEXT = """\
Chief Complaint
Patient presents for follow-up of diabetes and hypertension.

History of Present Illness
Patient is a 67-year-old male with Type 2 diabetes mellitus (E11.9)
and essential hypertension (I10). Reports blood sugar readings
consistently above 180 mg/dL over the past two weeks.

Past Medical History
1. Type 2 diabetes mellitus without complications - diagnosed 2018
2. Essential hypertension - diagnosed 2015
3. Chronic kidney disease stage 3 (N18.3) - diagnosed 2021

Current Medications
Metformin 1000mg twice daily
Lisinopril 10mg once daily
Insulin glargine 20 units nightly (Z79.4)

Assessment and Plan
1. Type 2 diabetes mellitus with hyperglycemia (E11.65)
   - HCC Category 37 - Diabetes without Complication
   - Increase insulin glargine to 24 units nightly
   - Recheck HbA1c in 3 months

2. Chronic kidney disease stage 3 (N18.3)
   - HCC Category 138 - CKD Stage 3-4
   - Monitor creatinine and eGFR quarterly
   - Refer to nephrology for further evaluation

HCC Risk
Total RAF Score estimate: 0.187
Active HCC conditions: HCC 37, HCC 138
"""

@pytest.fixture
def parser():
    return ClinicalPDFParser()

@pytest.fixture
def sample_parsed_doc():
    """A ParsedDocument built directly — no PDF file needed."""
    return ParsedDocument(
        document_id="test_doc_abc123",
        file_name="test_patient.pdf",
        file_path="/data/clinical_notes/test_patient.pdf",
        full_text=SAMPLE_CLINICAL_TEXT,
        page_count=1,
        metadata={"title": "Progress Note", "author": "Dr. Smith"},
        sections={
            "chief complaint": "Patient presents for follow-up of diabetes and hypertension.",
            "history of present illness": (
                "Patient is a 67-year-old male with Type 2 diabetes mellitus (E11.9) "
                "and essential hypertension (I10)."
            ),
            "past medical history": (
                "1. Type 2 diabetes mellitus without complications\n"
                "2. Essential hypertension\n"
                "3. Chronic kidney disease stage 3 (N18.3)"
            ),
            "assessment and plan": (
                "1. Type 2 diabetes mellitus with hyperglycemia (E11.65)\n"
                "   HCC Category 37 - Diabetes without Complication\n"
                "2. Chronic kidney disease stage 3 (N18.3)\n"
                "   HCC Category 138 - CKD Stage 3-4"
            ),
            "hcc risk": (
                "Total RAF Score estimate: 0.187\n"
                "Active HCC conditions: HCC 37, HCC 138"
            ),
        },
    )

@pytest.fixture
def chunker():
    return ClinicalChunker(chunk_size=512, chunk_overlap=50)


# ── ClinicalPDFParser tests ───────────────────────────────────

class TestClinicalPDFParser:

    def test_parse_raises_on_missing_file(self, parser):
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/path/patient.pdf")

    def test_extract_sections_detects_known_headers(self, parser):
        sections = parser._extract_sections(SAMPLE_CLINICAL_TEXT)
        assert "assessment and plan" in sections
        assert "past medical history" in sections
        assert "history of present illness" in sections

    def test_extract_sections_assessment_contains_hcc_codes(self, parser):
        sections = parser._extract_sections(SAMPLE_CLINICAL_TEXT)
        assessment = sections.get("assessment and plan", "")
        assert "E11.65" in assessment
        assert "N18.3" in assessment

    def test_extract_sections_returns_non_empty_content(self, parser):
        sections = parser._extract_sections(SAMPLE_CLINICAL_TEXT)
        for name, text in sections.items():
            assert text.strip(), f"Section '{name}' should not be empty"

    def test_match_section_exact_match(self, parser):
        assert parser._match_section("assessment and plan") == "assessment and plan"
        assert parser._match_section("hpi") == "hpi"
        assert parser._match_section("medications") == "medications"

    def test_match_section_startswith(self, parser):
        # Headers like "Assessment and Plan (continued)" still match
        result = parser._match_section("assessment and plan (continued)")
        assert result == "assessment and plan"

    def test_match_section_returns_none_for_body_text(self, parser):
        assert parser._match_section("patient is a 67-year-old male") is None
        assert parser._match_section("increase insulin to 24 units") is None
        assert parser._match_section("") is None

    def test_match_section_case_insensitive_input(self, parser):
        # _extract_sections normalises to lowercase before calling _match_section
        assert parser._match_section("assessment and plan") is not None

    def test_clinical_sections_list_contains_required_headers(self):
        required = {
            "assessment and plan", "history of present illness",
            "past medical history", "medications", "hcc risk",
        }
        for header in required:
            assert header in CLINICAL_SECTIONS, f"Missing required section: {header}"


# ── DocumentChunk tests ───────────────────────────────────────

class TestDocumentChunk:

    def test_to_embedding_text_with_section(self):
        chunk = DocumentChunk(
            chunk_id="abc123_chunk_0000",
            document_id="abc123",
            file_name="patient.pdf",
            text="Type 2 diabetes mellitus with hyperglycemia (E11.65). HCC Category 37.",
            chunk_index=0,
            total_chunks=5,
            section="assessment and plan",
        )
        result = chunk.to_embedding_text()
        assert result.startswith("[Section: Assessment And Plan]")
        assert "E11.65" in result

    def test_to_embedding_text_without_section(self):
        chunk = DocumentChunk(
            chunk_id="abc123_chunk_0001",
            document_id="abc123",
            file_name="patient.pdf",
            text="Patient presents with chronic conditions.",
            chunk_index=1,
            total_chunks=5,
            section=None,
        )
        result = chunk.to_embedding_text()
        # No section prefix — just the raw text
        assert result == "Patient presents with chronic conditions."
        assert "[Section:" not in result

    def test_to_embedding_text_section_title_cased(self):
        chunk = DocumentChunk(
            chunk_id="abc123_chunk_0002",
            document_id="abc123",
            file_name="patient.pdf",
            text="HbA1c 8.2%. Adjust insulin.",
            chunk_index=2,
            total_chunks=5,
            section="history of present illness",
        )
        result = chunk.to_embedding_text()
        assert "[Section: History Of Present Illness]" in result

    def test_chunk_id_format(self):
        chunk = DocumentChunk(
            chunk_id="deadbeef_chunk_0042",
            document_id="deadbeef",
            file_name="patient.pdf",
            text="Some clinical text.",
            chunk_index=42,
            total_chunks=100,
        )
        assert chunk.chunk_id == "deadbeef_chunk_0042"
        assert chunk.chunk_index == 42


# ── ClinicalChunker tests ─────────────────────────────────────

class TestClinicalChunker:

    def test_chunk_with_sections_returns_document_chunks(self, chunker, sample_parsed_doc):
        chunks = chunker.chunk(sample_parsed_doc)
        assert len(chunks) > 0
        assert all(isinstance(c, DocumentChunk) for c in chunks)

    def test_chunk_with_sections_preserves_section_names(self, chunker, sample_parsed_doc):
        chunks = chunker.chunk(sample_parsed_doc)
        sections_found = {c.section for c in chunks}
        assert "assessment and plan" in sections_found

    def test_chunk_priority_sections_come_first(self, chunker, sample_parsed_doc):
        chunks = chunker.chunk(sample_parsed_doc)
        # First chunk must come from a priority section
        assert chunks[0].section in PRIORITY_SECTIONS, (
            f"First chunk should be from a priority section, got: {chunks[0].section}"
        )

    def test_chunk_all_have_document_id(self, chunker, sample_parsed_doc):
        chunks = chunker.chunk(sample_parsed_doc)
        for chunk in chunks:
            assert chunk.document_id == sample_parsed_doc.document_id

    def test_chunk_total_chunks_updated(self, chunker, sample_parsed_doc):
        chunks = chunker.chunk(sample_parsed_doc)
        total = len(chunks)
        for chunk in chunks:
            assert chunk.total_chunks == total

    def test_chunk_full_text_fallback_when_no_sections(self, chunker):
        doc_no_sections = ParsedDocument(
            document_id="nosec_abc",
            file_name="no_sections.pdf",
            file_path="/data/no_sections.pdf",
            full_text=(
                "Patient is a 72-year-old female with congestive heart failure (I50.9) "
                "and chronic obstructive pulmonary disease (J44.1). She presents today "
                "for routine follow-up. Her symptoms include dyspnea on exertion and "
                "pedal edema. Medications: furosemide 40mg, carvedilol 6.25mg, "
                "tiotropium inhaler. Plan: continue current medications, recheck BNP "
                "in 4 weeks, pulmonology referral for COPD management."
            ),
            page_count=1,
            sections={},  # no sections
        )
        chunks = chunker.chunk(doc_no_sections)
        assert len(chunks) > 0
        # All chunks should have section=None
        assert all(c.section is None for c in chunks)

    def test_chunk_many_aggregates_all_docs(self, chunker, sample_parsed_doc):
        doc2 = ParsedDocument(
            document_id="second_doc_xyz",
            file_name="patient2.pdf",
            file_path="/data/patient2.pdf",
            full_text="Heart failure follow-up. I50.32 chronic diastolic heart failure.",
            page_count=1,
            sections={
                "assessment and plan": (
                    "Chronic diastolic heart failure (I50.32). HCC Category 85."
                )
            },
        )
        all_chunks = chunker.chunk_many([sample_parsed_doc, doc2])
        doc_ids = {c.document_id for c in all_chunks}
        assert "test_doc_abc123" in doc_ids
        assert "second_doc_xyz" in doc_ids

    def test_chunk_ids_are_unique(self, chunker, sample_parsed_doc):
        chunks = chunker.chunk(sample_parsed_doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs detected"

    def test_chunk_text_is_not_empty(self, chunker, sample_parsed_doc):
        chunks = chunker.chunk(sample_parsed_doc)
        for chunk in chunks:
            assert chunk.text.strip(), f"Empty chunk found: {chunk.chunk_id}"