"""
Sentiment Aggregation Engine for Document-Level and Section-Level Metrics
"""

import numpy as np
from collections import defaultdict
from typing import List, Dict, Any
from config import TOP_K_CHUNKS, PRIORITY_HIGH_RISK_THRESHOLD, PRIORITY_WATCH_THRESHOLD


class SentimentAggregator:
    """
    Aggregates chunk-level sentiment probabilities into document-wide
    and section-level metrics, and identifies key highlights.
    """

    @staticmethod
    def aggregate_document_sentiment(annotated_chunks: List[Dict[str, Any]], top_k: int = TOP_K_CHUNKS) -> Dict[str, Any]:
        """
        Computes overall document sentiment, distribution, and top positive/negative chunks.

        Args:
            annotated_chunks: List of chunk dictionaries containing sentiment scores.
            top_k: Number of top chunks to extract for positive/negative rankings.

        Returns:
            Dict containing document summary statistics.
        """
        if not annotated_chunks:
            return {
                "overall_sentiment": "Neutral",
                "overall_confidence": 0.0,
                "net_sentiment_score": 0.0,
                "mean_probabilities": {"positive": 0.0, "negative": 0.0, "neutral": 0.0},
                "sentiment_distribution": {"Positive": 0, "Negative": 0, "Neutral": 0},
                "distribution_percentage": {"Positive": 0.0, "Negative": 0.0, "Neutral": 0.0},
                "total_chunks": 0,
                "top_positive_chunks": [],
                "top_negative_chunks": []
            }

        total_chunks = len(annotated_chunks)

        # Extract score arrays
        pos_scores = [c["positive_score"] for c in annotated_chunks]
        neg_scores = [c["negative_score"] for c in annotated_chunks]
        neu_scores = [c["neutral_score"] for c in annotated_chunks]

        # Calculate mean probabilities across document
        mean_pos = float(np.mean(pos_scores))
        mean_neg = float(np.mean(neg_scores))
        mean_neu = float(np.mean(neu_scores))

        mean_probs = {
            "Positive": mean_pos,
            "Negative": mean_neg,
            "Neutral": mean_neu
        }

        # Determine overall document sentiment based on dominant mean probability
        overall_sentiment = max(mean_probs, key=mean_probs.get)
        overall_confidence = mean_probs[overall_sentiment]
        net_sentiment_score = mean_pos - mean_neg

        # Compute label distribution counts
        label_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
        for chunk in annotated_chunks:
            label = chunk["predicted_label"]
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        # Percentage distribution
        distribution_percentage = {
            k: round((v / total_chunks) * 100, 2)
            for k, v in label_counts.items()
        }

        # Extract Top K Most Positive Chunks
        sorted_pos = sorted(annotated_chunks, key=lambda x: x["positive_score"], reverse=True)
        top_positive_chunks = sorted_pos[:top_k]

        # Extract Top K Most Negative Chunks
        sorted_neg = sorted(annotated_chunks, key=lambda x: x["negative_score"], reverse=True)
        top_negative_chunks = sorted_neg[:top_k]

        return {
            "overall_sentiment": overall_sentiment,
            "overall_confidence": overall_confidence,
            "net_sentiment_score": net_sentiment_score,
            "mean_probabilities": {
                "positive": mean_pos,
                "negative": mean_neg,
                "neutral": mean_neu
            },
            "sentiment_distribution": label_counts,
            "distribution_percentage": distribution_percentage,
            "total_chunks": total_chunks,
            "top_positive_chunks": top_positive_chunks,
            "top_negative_chunks": top_negative_chunks
        }

    @staticmethod
    def aggregate_by_section(annotated_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Groups annotated chunks by section_name and computes per-section
        sentiment summary with priority classification.

        Args:
            annotated_chunks: List of chunk dicts, each containing
                'section_name', 'positive_score', 'negative_score', 'neutral_score'.

        Returns:
            List of section summary dicts sorted by priority
            (High Risk first, then Watch, then Normal — within each tier
            sorted by descending mean_negative).
        """
        if not annotated_chunks:
            return []

        # Group chunks by section name, preserving insertion order
        section_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for chunk in annotated_chunks:
            section_name = chunk.get("section_name", "Uncategorized")
            section_groups[section_name].append(chunk)

        section_summaries: List[Dict[str, Any]] = []

        for section_name, chunks in section_groups.items():
            pos_scores = [c["positive_score"] for c in chunks]
            neg_scores = [c["negative_score"] for c in chunks]
            neu_scores = [c["neutral_score"] for c in chunks]

            mean_pos = float(np.mean(pos_scores))
            mean_neg = float(np.mean(neg_scores))
            mean_neu = float(np.mean(neu_scores))

            # Determine section-level dominant sentiment
            class_means = {
                "Positive": mean_pos,
                "Negative": mean_neg,
                "Neutral": mean_neu
            }
            sentiment = max(class_means, key=class_means.get)
            confidence = class_means[sentiment]
            net_score = mean_pos - mean_neg

            # Priority classification based on mean negative probability
            if mean_neg > PRIORITY_HIGH_RISK_THRESHOLD:
                priority = "High Risk"
                priority_rank = 0
            elif mean_neg > PRIORITY_WATCH_THRESHOLD:
                priority = "Watch"
                priority_rank = 1
            else:
                priority = "Normal"
                priority_rank = 2

            section_summaries.append({
                "section_name": section_name,
                "sentiment": sentiment,
                "confidence": round(confidence, 4),
                "net_sentiment_score": round(net_score, 4),
                "mean_positive": round(mean_pos, 4),
                "mean_negative": round(mean_neg, 4),
                "mean_neutral": round(mean_neu, 4),
                "chunk_count": len(chunks),
                "priority": priority,
                "_priority_rank": priority_rank,  # Internal sort key
            })

        # Sort: High Risk first → Watch → Normal; within each tier, highest negative first
        section_summaries.sort(key=lambda s: (s["_priority_rank"], -s["mean_negative"]))

        # Remove internal sort key before returning
        for s in section_summaries:
            del s["_priority_rank"]

        return section_summaries

