"""
Centralized Configuration for Financial Sentiment Analysis Pipeline
"""

import os
import torch

# Model Configuration
MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Text Chunking Configuration
# FinBERT has a max context length of 512 tokens.
# Using 384 tokens with an overlap of 64 tokens ensures complete sentence context without truncation.
MAX_TOKENS_PER_CHUNK = 384
OVERLAP_TOKENS = 64
APPROX_WORDS_PER_CHUNK = 250
APPROX_WORD_OVERLAP = 40

# Sentiment Labels & Scoring Mapping
SENTIMENT_LABELS = ["positive", "negative", "neutral"]

# UI & Display Configuration
TOP_K_CHUNKS = 5

# Section Priority Classification Thresholds (Milestone 2)
# Based on mean negative probability across a section's chunks.
PRIORITY_HIGH_RISK_THRESHOLD = 0.5   # mean_negative > 0.5 → "High Risk"
PRIORITY_WATCH_THRESHOLD = 0.3       # mean_negative > 0.3 → "Watch"
# Everything else → "Normal"

# SHAP Explainability Configuration (Milestone 3)
# Maximum word count per text sent to SHAP explainer.
# Limits computational cost: 200 words ≈ ~300 tokens, well within FinBERT's 512 limit.
SHAP_MAX_TOKENS = 200
# Maximum SHAP model evaluations per explanation.
# Lower = faster but less precise attributions. 100 gives good accuracy in < 5s.
SHAP_MAX_EVALS = 100
# Number of top driver words to display in the dashboard
SHAP_TOP_K_DRIVERS = 10
# Debug mode: when True, logs detailed SHAP internals (shapes, sums, per-call info).
# Set to False in production to suppress verbose output.
SHAP_DEBUG_MODE = False

