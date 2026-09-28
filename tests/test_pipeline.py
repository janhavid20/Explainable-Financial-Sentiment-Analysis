"""
Milestone 1 Core Pipeline Integration Test
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.segmentation import TextChunker
from src.aggregation import SentimentAggregator

def test_pipeline_components():
    sample_financial_text = (
        "Revenue increased by 15% year-over-year to $4.2 billion, driven by strong growth in cloud services. "
        "Operating expenses rose moderately due to strategic investments in artificial intelligence research. "
        "However, macroeconomic uncertainties and supply chain bottlenecks present potential headwinds for the upcoming quarter. "
        "The board declared a quarterly dividend of $0.50 per share, reflecting strong cash flow generation and balance sheet resilience. "
        "Litigation costs decreased significantly following the settlement of outstanding patent disputes."
    )
    
    # 1. Test Chunker
    chunker = TextChunker(target_words_per_chunk=30, overlap_words=5)
    chunks = chunker.chunk_text(sample_financial_text)
    assert len(chunks) > 0, "Chunker returned 0 chunks"
    print(f"[PASSED] TextChunker test passed! Produced {len(chunks)} chunks.")
    
    # 2. Test Aggregator
    sample_annotated = [
        {
            "chunk_id": 1,
            "text": "Revenue increased by 15%...",
            "word_count": 12,
            "positive_score": 0.92,
            "negative_score": 0.02,
            "neutral_score": 0.06,
            "predicted_label": "Positive",
            "confidence_score": 0.92
        },
        {
            "chunk_id": 2,
            "text": "However macroeconomic uncertainties...",
            "word_count": 14,
            "positive_score": 0.05,
            "negative_score": 0.88,
            "neutral_score": 0.07,
            "predicted_label": "Negative",
            "confidence_score": 0.88
        }
    ]
    agg = SentimentAggregator.aggregate_document_sentiment(sample_annotated)
    assert "overall_sentiment" in agg
    assert len(agg["top_positive_chunks"]) == 2
    assert len(agg["top_negative_chunks"]) == 2
    print(f"[PASSED] SentimentAggregator test passed! Overall Sentiment: {agg['overall_sentiment']}")

if __name__ == "__main__":
    test_pipeline_components()
