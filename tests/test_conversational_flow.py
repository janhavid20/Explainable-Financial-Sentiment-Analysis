"""
Unit Tests for Conversational Flow & Answering Engine
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import (
    generate_conversational_response,
    format_sentiment_badge,
    format_priority_badge,
    render_token_visualization_html,
)

class TestConversationalFlow(unittest.TestCase):
    def setUp(self):
        # Create a mock doc_state structure that mirrors the real output
        self.mock_doc_state = {
            "filename": "test_financial_report.pdf",
            "extraction_result": {
                "total_pages": 3,
                "total_word_count": 550,
                "extracted_pages_count": 3,
                "total_char_count": 3200,
                "full_text": "Sample financial text"
            },
            "sections": [
                {"section_name": "Executive Summary", "text": "...", "word_count": 150},
                {"section_name": "Risk Factors", "text": "...", "word_count": 200},
                {"section_name": "Revenue & Growth", "text": "...", "word_count": 200},
            ],
            "chunks": [
                {"chunk_id": 1, "section_name": "Executive Summary", "text": "Revenue increased...", "word_count": 50},
                {"chunk_id": 2, "section_name": "Risk Factors", "text": "Supply chain disruption...", "word_count": 50},
            ],
            "annotated_chunks": [
                {
                    "chunk_id": 1,
                    "section_name": "Executive Summary",
                    "text": "Revenue increased substantially across operations.",
                    "word_count": 50,
                    "predicted_label": "Positive",
                    "confidence_score": 0.94,
                    "positive_score": 0.94,
                    "negative_score": 0.02,
                    "neutral_score": 0.04
                },
                {
                    "chunk_id": 2,
                    "section_name": "Risk Factors",
                    "text": "Supply chain disruption and regulatory investigations pose headwinds.",
                    "word_count": 50,
                    "predicted_label": "Negative",
                    "confidence_score": 0.89,
                    "positive_score": 0.03,
                    "negative_score": 0.89,
                    "neutral_score": 0.08
                }
            ],
            "aggregation": {
                "overall_sentiment": "Positive",
                "overall_confidence": 0.85,
                "net_sentiment_score": 0.25,
                "mean_probabilities": {"positive": 0.55, "negative": 0.30, "neutral": 0.15},
                "sentiment_distribution": {"Positive": 1, "Negative": 1, "Neutral": 0},
                "distribution_percentage": {"Positive": 50.0, "Negative": 50.0, "Neutral": 0.0},
                "total_chunks": 2,
                "top_positive_chunks": [],
                "top_negative_chunks": []
            },
            "section_summaries": [
                {
                    "section_name": "Risk Factors",
                    "sentiment": "Negative",
                    "confidence": 0.89,
                    "net_sentiment_score": -0.86,
                    "mean_positive": 0.03,
                    "mean_negative": 0.89,
                    "mean_neutral": 0.08,
                    "chunk_count": 1,
                    "priority": "High Risk"
                },
                {
                    "section_name": "Executive Summary",
                    "sentiment": "Positive",
                    "confidence": 0.94,
                    "net_sentiment_score": +0.92,
                    "mean_positive": 0.94,
                    "mean_negative": 0.02,
                    "mean_neutral": 0.04,
                    "chunk_count": 1,
                    "priority": "Normal"
                }
            ],
            "shap_results": {
                "positive": {
                    "section_name": "Executive Summary",
                    "sentiment": "Positive",
                    "explanation": {
                        "tokens": ["[CLS]", "revenue", "increased", "substantially", "[SEP]"],
                        "shap_values": {
                            "positive": [0.0, 0.42, 0.38, 0.25, 0.0],
                            "negative": [0.0, -0.1, -0.05, -0.02, 0.0],
                            "neutral": [0.0, -0.32, -0.33, -0.23, 0.0]
                        },
                        "predicted_class": "positive",
                        "predicted_proba": 0.94,
                        "elapsed_seconds": 1.2
                    }
                },
                "negative": {
                    "section_name": "Risk Factors",
                    "sentiment": "Negative",
                    "explanation": {
                        "tokens": ["[CLS]", "disruption", "regulatory", "headwinds", "[SEP]"],
                        "shap_values": {
                            "positive": [0.0, -0.2, -0.1, -0.15, 0.0],
                            "negative": [0.0, 0.45, 0.30, 0.35, 0.0],
                            "neutral": [0.0, -0.25, -0.20, -0.20, 0.0]
                        },
                        "predicted_class": "negative",
                        "predicted_proba": 0.89,
                        "elapsed_seconds": 1.1
                    }
                }
            },
            "shap_pos_drivers": [{"token": "revenue", "contribution": 0.42}],
            "shap_neg_drivers": [{"token": "disruption", "contribution": 0.45}],
            "timings": {"total_pipeline": 4.5}
        }

    def test_full_analysis_intent(self):
        resp = generate_conversational_response("Analyze this report.", self.mock_doc_state)
        self.assertEqual(resp["intent"], "full_analysis")
        self.assertIn("doc_state", resp)

    def test_concerning_sections_intent(self):
        resp = generate_conversational_response("Identify the most concerning sections.", self.mock_doc_state)
        self.assertEqual(resp["intent"], "risk_analysis")
        self.assertTrue(len(resp["risk_sections"]) > 0)
        self.assertEqual(resp["risk_sections"][0]["section_name"], "Risk Factors")

    def test_summary_findings_intent(self):
        resp = generate_conversational_response("Summarize positive and negative findings.", self.mock_doc_state)
        self.assertEqual(resp["intent"], "summary_analysis")
        self.assertIn("top_positive", resp)
        self.assertIn("top_negative", resp)

    def test_explanation_intent(self):
        resp = generate_conversational_response("Explain why a section received its sentiment prediction.", self.mock_doc_state)
        self.assertEqual(resp["intent"], "explanation_analysis")
        self.assertIn("shap_results", resp)

    def test_specific_section_query(self):
        resp = generate_conversational_response("Tell me about the Risk Factors section", self.mock_doc_state)
        self.assertEqual(resp["intent"], "section_detail")
        self.assertEqual(resp["section"]["section_name"], "Risk Factors")
        self.assertEqual(len(resp["chunks"]), 1)

    def test_badge_helpers(self):
        pos_badge = format_sentiment_badge("Positive")
        self.assertIn("badge-pos", pos_badge)
        self.assertIn("Positive", pos_badge)

        risk_badge = format_priority_badge("High Risk")
        self.assertIn("badge-neg", risk_badge)

    def test_token_visualization_html(self):
        explanation = self.mock_doc_state["shap_results"]["positive"]["explanation"]
        html = render_token_visualization_html(explanation, "positive")
        self.assertIn("shap-container-dark", html)
        self.assertIn("revenue", html)

if __name__ == "__main__":
    unittest.main()
