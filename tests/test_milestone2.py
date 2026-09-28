"""
Milestone 2 Tests: Section Detection, Section-Aware Chunking, Section Aggregation
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.segmentation.section_detector import SectionDetector
from src.segmentation.text_chunker import TextChunker
from src.aggregation.sentiment_aggregator import SentimentAggregator


def test_section_detector_with_headings():
    """Test section detection on text with clear financial headings."""
    sample_text = """ACME Corp Annual Report 2025
Page 1

Executive Summary
Revenue grew by 18% year-over-year driven by strong performance in cloud services.
Operating margins expanded to 22% reflecting improved cost discipline.
The company returned $2.1 billion to shareholders through dividends and buybacks.

Risk Factors
The company faces exposure to foreign currency fluctuations in international markets.
Supply chain disruptions could materially impact manufacturing operations.
Regulatory changes in key markets may require significant compliance investments.
Cybersecurity threats continue to evolve and pose operational risks.

Legal Proceedings
The company is a defendant in a patent infringement lawsuit filed by TechCo Inc.
Management believes the claims are without merit and intends to vigorously defend.

Outlook
Management expects revenue growth of 12-15% in fiscal year 2026.
Capital expenditure is projected at $800 million to support expansion.
"""
    detector = SectionDetector()
    sections = detector.detect_sections(sample_text)

    assert len(sections) >= 3, f"Expected at least 3 sections, got {len(sections)}"

    section_names = [s["section_name"] for s in sections]
    print(f"[PASSED] Section detection: found {len(sections)} sections: {section_names}")

    # Check that key sections were detected
    assert any("Risk" in name for name in section_names), "Risk Factors section not found"
    assert any("Legal" in name for name in section_names), "Legal Proceedings section not found"
    print("[PASSED] Key sections (Risk Factors, Legal Proceedings) correctly identified")


def test_section_detector_fallback():
    """Test fallback when no headings are present."""
    plain_text = (
        "The company reported strong earnings this quarter with revenue "
        "increasing by 15 percent compared to the same period last year. "
        "Operating expenses remained flat while gross margins improved by 200 basis points."
    )

    detector = SectionDetector()
    sections = detector.detect_sections(plain_text)

    assert len(sections) == 1, f"Expected 1 fallback section, got {len(sections)}: {[s['section_name'] for s in sections]}"
    assert sections[0]["section_name"] == "Full Document"
    print("[PASSED] Fallback: unstructured text wrapped as 'Full Document'")


def test_section_aware_chunking():
    """Test that chunk_sections stamps section_name and uses global IDs."""
    sections = [
        {
            "section_name": "Executive Summary",
            "text": "Revenue grew strongly. Margins expanded. Dividends increased. Cash flow was robust. Outlook is positive.",
            "word_count": 15,
        },
        {
            "section_name": "Risk Factors",
            "text": "Currency risk is significant. Supply chain issues persist. Regulatory changes loom. Cyber threats evolve.",
            "word_count": 14,
        },
    ]

    chunker = TextChunker(target_words_per_chunk=10, overlap_words=2)
    chunks = chunker.chunk_sections(sections)

    assert len(chunks) > 0, "chunk_sections produced 0 chunks"

    # Verify every chunk has section_name
    for chunk in chunks:
        assert "section_name" in chunk, f"Chunk {chunk['chunk_id']} missing section_name"
        assert chunk["section_name"] in ("Executive Summary", "Risk Factors")

    # Verify global IDs are sequential and unique
    ids = [c["chunk_id"] for c in chunks]
    assert ids == list(range(1, len(ids) + 1)), f"Chunk IDs not sequential: {ids}"

    print(f"[PASSED] Section-aware chunking: {len(chunks)} chunks with section_name and sequential global IDs")


def test_section_aggregation():
    """Test aggregate_by_section groups and classifies priority correctly."""
    annotated_chunks = [
        # Executive Summary chunks (positive)
        {"chunk_id": 1, "section_name": "Executive Summary", "positive_score": 0.85, "negative_score": 0.05, "neutral_score": 0.10, "predicted_label": "Positive", "confidence_score": 0.85},
        {"chunk_id": 2, "section_name": "Executive Summary", "positive_score": 0.78, "negative_score": 0.08, "neutral_score": 0.14, "predicted_label": "Positive", "confidence_score": 0.78},
        # Risk Factors chunks (negative)
        {"chunk_id": 3, "section_name": "Risk Factors", "positive_score": 0.05, "negative_score": 0.82, "neutral_score": 0.13, "predicted_label": "Negative", "confidence_score": 0.82},
        {"chunk_id": 4, "section_name": "Risk Factors", "positive_score": 0.03, "negative_score": 0.88, "neutral_score": 0.09, "predicted_label": "Negative", "confidence_score": 0.88},
        {"chunk_id": 5, "section_name": "Risk Factors", "positive_score": 0.07, "negative_score": 0.75, "neutral_score": 0.18, "predicted_label": "Negative", "confidence_score": 0.75},
        # Outlook chunks (mildly negative / watch)
        {"chunk_id": 6, "section_name": "Outlook", "positive_score": 0.25, "negative_score": 0.40, "neutral_score": 0.35, "predicted_label": "Negative", "confidence_score": 0.40},
    ]

    section_summaries = SentimentAggregator.aggregate_by_section(annotated_chunks)

    assert len(section_summaries) == 3, f"Expected 3 sections, got {len(section_summaries)}"

    # Build lookup
    by_name = {s["section_name"]: s for s in section_summaries}

    # Risk Factors should be High Risk (mean_neg > 0.5)
    assert by_name["Risk Factors"]["priority"] == "High Risk", \
        f"Risk Factors priority: {by_name['Risk Factors']['priority']}"

    # Executive Summary should be Normal
    assert by_name["Executive Summary"]["priority"] == "Normal", \
        f"Executive Summary priority: {by_name['Executive Summary']['priority']}"

    # Outlook should be Watch (mean_neg = 0.40 > 0.3)
    assert by_name["Outlook"]["priority"] == "Watch", \
        f"Outlook priority: {by_name['Outlook']['priority']}"

    # Verify sort order: High Risk first
    assert section_summaries[0]["priority"] == "High Risk"

    print(f"[PASSED] Section aggregation: priorities = {[s['priority'] for s in section_summaries]}")
    print(f"         Sort order correct: {[s['section_name'] for s in section_summaries]}")


if __name__ == "__main__":
    test_section_detector_with_headings()
    test_section_detector_fallback()
    test_section_aware_chunking()
    test_section_aggregation()
    print("\n=== ALL MILESTONE 2 TESTS PASSED ===")
