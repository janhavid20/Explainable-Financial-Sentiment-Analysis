"""
SHAP Explainability Module for FinBERT Sentiment Analysis
Milestone 3 — Targeted Explainable AI (Production-Ready)

Generates token-level SHAP explanations for targeted sections only:
  1. Most positive section (highest positive probability)
  2. Most negative / high-risk section (highest negative probability)
  3. Overall representative chunk (most aligned with document sentiment)

Design decisions:
  - max_evals limits SHAP to ~100 model calls per explanation (< 5s each)
  - output_names explicitly binds SHAP columns to [positive, negative, neutral]
  - Explains the ACTUAL predicted class, not the assumed/expected class
  - Debug logging gated behind SHAP_DEBUG_MODE config flag
"""

import time
import logging
import numpy as np
import shap
import string
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline as hf_pipeline
from config import MODEL_NAME, SHAP_MAX_TOKENS, SHAP_MAX_EVALS, SHAP_DEBUG_MODE

logger = logging.getLogger(__name__)

# Canonical class ordering — must match _pipeline_predict() column layout
CLASS_LABELS = ["positive", "negative", "neutral"]


class SHAPExplainer:
    """
    Provides targeted SHAP explanations for FinBERT sentiment predictions.

    Designed for production use with large financial reports:
    - Runs SHAP on at most 3 text chunks (positive, negative, overall)
    - Extracts token-level contribution scores per sentiment class
    - Returns structured data for dashboard rendering
    """

    def __init__(self, model_name: str = MODEL_NAME):
        """
        Initialize the SHAP text explainer with the FinBERT pipeline.

        Args:
            model_name: Hugging Face model identifier (default: ProsusAI/finbert).
        """
        logger.info(f"[SHAP] Initializing SHAP explainer with model '{model_name}'...")
        init_start = time.time()

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        # Log and validate the model's label mapping
        self._id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        logger.info(f"[SHAP] Model id2label: {self._id2label}")

        # Build HuggingFace text-classification pipeline (top_k=None returns all classes)
        self._pipe = hf_pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            top_k=None,
            truncation=True,
            max_length=512,
        )

        # Create SHAP Text explainer with explicit output_names
        # This binds SHAP value columns to named classes
        self.masker = shap.maskers.Text(self.tokenizer)
        self.explainer = shap.Explainer(
            self._pipeline_predict,
            self.masker,
            output_names=CLASS_LABELS,
        )

        logger.info(f"[SHAP] Explainer initialized in {time.time() - init_start:.2f}s")

    # -------------------------------------------------------------------------
    # SHAP-Compatible Prediction Function
    # -------------------------------------------------------------------------

    def _pipeline_predict(self, texts) -> np.ndarray:
        """
        SHAP-compatible prediction wrapper.

        SHAP calls this with masked/perturbed text variants. The input may be:
        - A Python list of strings
        - A numpy array of strings

        Always returns np.ndarray of shape (N, 3) with columns:
          [positive_prob, negative_prob, neutral_prob]
        """
        # Normalize input — SHAP may pass numpy arrays or lists
        if isinstance(texts, np.ndarray):
            text_list = texts.tolist()
        else:
            text_list = list(texts)

        if SHAP_DEBUG_MODE:
            logger.debug(f"[SHAP-predict] Batch size: {len(text_list)}")

        # Run the HuggingFace pipeline
        # With top_k=None, output for list input is: [[dict, dict, dict], [dict, dict, dict], ...]
        # Each inner list contains one dict per class with 'label' and 'score'
        raw_results = self._pipe(text_list, truncation=True, max_length=512)

        output = []
        for result in raw_results:
            score_map = self._parse_pipeline_result(result)
            output.append([
                score_map.get("positive", 0.0),
                score_map.get("negative", 0.0),
                score_map.get("neutral", 0.0),
            ])

        return np.array(output, dtype=np.float64)

    @staticmethod
    def _parse_pipeline_result(result) -> Dict[str, float]:
        """
        Robustly parse a single sample's pipeline output into a {label: score} dict.

        Handles all known HuggingFace pipeline output formats:
        - List of dicts: [{'label': 'positive', 'score': 0.95}, ...]
        - Single dict: {'label': 'positive', 'score': 0.95}
        - Nested list: [[{'label': ..., 'score': ...}, ...]]
        """
        # Unwrap nested lists (can happen with certain pipeline/batch combos)
        while isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
            result = result[0]

        # At this point, result should be a list of dicts or a single dict
        if isinstance(result, dict):
            predictions = [result]
        elif isinstance(result, list):
            predictions = result
        else:
            logger.warning(f"[SHAP] Unexpected pipeline result type: {type(result)}")
            return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

        score_map = {}
        for item in predictions:
            if isinstance(item, dict) and "label" in item and "score" in item:
                label = str(item["label"]).lower()
                score_map[label] = float(item["score"])

        return score_map

    # -------------------------------------------------------------------------
    # Section Selection Logic
    # -------------------------------------------------------------------------

    @staticmethod
    def select_target_sections(
        section_summaries: List[Dict[str, Any]],
        annotated_chunks: List[Dict[str, Any]],
        overall_sentiment: str,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Selects exactly 3 target texts for SHAP explanation:
        1. Positive: section with highest mean_positive probability
        2. Negative: section with highest mean_negative probability
        3. Overall: representative chunk most aligned with document-level sentiment

        Args:
            section_summaries: Section-level aggregated metrics.
            annotated_chunks: All annotated chunks with sentiment scores.
            overall_sentiment: Document-level dominant sentiment label.

        Returns:
            Dict with keys 'positive', 'negative', 'overall', each containing:
              - section_name, text, sentiment, confidence, chunk
        """
        targets = {}

        if not section_summaries or not annotated_chunks:
            logger.warning("[SHAP] Empty sections or chunks — cannot select targets.")
            return targets

        # --- Positive Section ---
        pos_section = max(section_summaries, key=lambda s: s["mean_positive"])
        pos_chunks = [c for c in annotated_chunks if c.get("section_name") == pos_section["section_name"]]
        if pos_chunks:
            best_pos_chunk = max(pos_chunks, key=lambda c: c["positive_score"])
            targets["positive"] = {
                "section_name": pos_section["section_name"],
                "text": best_pos_chunk["text"],
                "sentiment": pos_section["sentiment"],
                "confidence": pos_section["confidence"],
                "mean_positive": pos_section["mean_positive"],
                "chunk": best_pos_chunk,
            }
            logger.info(
                f"[SHAP] Positive target: {pos_section['section_name']} "
                f"(mean_pos={pos_section['mean_positive']:.4f}, "
                f"chunk predicted={best_pos_chunk['predicted_label']})"
            )

        # --- Negative / High-Risk Section ---
        high_risk_sections = [s for s in section_summaries if s.get("priority") == "High Risk"]
        if high_risk_sections:
            neg_section = max(high_risk_sections, key=lambda s: s["mean_negative"])
        else:
            neg_section = max(section_summaries, key=lambda s: s["mean_negative"])

        neg_chunks = [c for c in annotated_chunks if c.get("section_name") == neg_section["section_name"]]
        if neg_chunks:
            best_neg_chunk = max(neg_chunks, key=lambda c: c["negative_score"])
            targets["negative"] = {
                "section_name": neg_section["section_name"],
                "text": best_neg_chunk["text"],
                "sentiment": neg_section["sentiment"],
                "confidence": neg_section["confidence"],
                "mean_negative": neg_section["mean_negative"],
                "chunk": best_neg_chunk,
            }
            logger.info(
                f"[SHAP] Negative target: {neg_section['section_name']} "
                f"(mean_neg={neg_section['mean_negative']:.4f}, "
                f"chunk predicted={best_neg_chunk['predicted_label']})"
            )

        # --- Overall Representative Chunk ---
        sentiment_key = overall_sentiment.lower() + "_score"
        matching_chunks = [
            c for c in annotated_chunks
            if c.get("predicted_label", "").lower() == overall_sentiment.lower()
        ]
        if not matching_chunks:
            matching_chunks = annotated_chunks

        best_overall = max(matching_chunks, key=lambda c: c.get(sentiment_key, c["confidence_score"]))
        targets["overall"] = {
            "section_name": best_overall.get("section_name", "Unknown"),
            "text": best_overall["text"],
            "sentiment": best_overall["predicted_label"],
            "confidence": best_overall["confidence_score"],
            "chunk": best_overall,
        }
        logger.info(
            f"[SHAP] Overall target: chunk {best_overall['chunk_id']} "
            f"from '{best_overall.get('section_name')}' "
            f"(predicted={best_overall['predicted_label']}, conf={best_overall['confidence_score']:.4f})"
        )

        return targets

    # -------------------------------------------------------------------------
    # SHAP Explanation Generation
    # -------------------------------------------------------------------------

    def explain_text(
        self,
        text: str,
        max_tokens: int = SHAP_MAX_TOKENS,
        max_evals: int = SHAP_MAX_EVALS,
    ) -> Dict[str, Any]:
        """
        Generate SHAP token-level explanation for a single text.

        Args:
            text: Input text string to explain.
            max_tokens: Max words to keep (truncates long texts for speed).
            max_evals: Maximum SHAP model evaluations (controls speed vs accuracy).

        Returns:
            Dict containing:
              - tokens: list of token strings
              - shap_values: dict with 'positive', 'negative', 'neutral' arrays
              - base_values: dict with base values for each class
              - predicted_class: str (what FinBERT actually predicted)
              - predicted_proba: float
              - all_probabilities: dict of all class probabilities
        """
        start = time.time()

        # Truncate text to limit SHAP computation
        words = text.split()
        if len(words) > max_tokens:
            text = " ".join(words[:max_tokens])
            logger.info(f"[SHAP] Truncated text from {len(words)} to {max_tokens} words")

        logger.info(f"[SHAP] Generating explanation ({len(text.split())} words, max_evals={max_evals})...")

        # --- Step 1: Get FinBERT's actual prediction BEFORE SHAP ---
        prediction = self._pipeline_predict([text])[0]  # [pos, neg, neu]
        pred_idx = int(np.argmax(prediction))
        predicted_class = CLASS_LABELS[pred_idx]
        predicted_proba = float(prediction[pred_idx])
        all_probs = {label: float(prediction[i]) for i, label in enumerate(CLASS_LABELS)}

        logger.info(
            f"[SHAP] FinBERT prediction: {predicted_class} ({predicted_proba:.4f}) | "
            f"all={all_probs}"
        )

        # --- Step 2: Run SHAP with evaluation budget ---
        shap_values = self.explainer([text], max_evals=max_evals)

        # Validate SHAP output shape
        tokens = list(shap_values.data[0])
        values = shap_values.values[0]   # (num_tokens, num_classes)
        base = shap_values.base_values[0]  # (num_classes,)

        if SHAP_DEBUG_MODE:
            logger.info(f"[SHAP-debug] values.shape={values.shape}, output_names={shap_values.output_names}")
            logger.info(f"[SHAP-debug] base_values={base}")
            if values.ndim > 1:
                for i, lbl in enumerate(CLASS_LABELS):
                    logger.info(f"[SHAP-debug] {lbl} SHAP sum={values[:, i].sum():.4f}")

        # --- Step 3: Package per-class SHAP values ---
        shap_dict = {}
        base_dict = {}
        for idx, label in enumerate(CLASS_LABELS):
            if values.ndim > 1:
                shap_dict[label] = values[:, idx].tolist()
            else:
                # Single-class fallback (shouldn't happen with output_names)
                shap_dict[label] = values.tolist()
            base_dict[label] = float(base[idx]) if hasattr(base, '__len__') else float(base)

        # --- Step 4: Clean, merge subwords, and filter stopwords ---
        merged_tokens, merged_shap_dict = self._clean_and_merge(tokens, shap_dict)

        elapsed = time.time() - start
        logger.info(f"[SHAP] Explanation generated in {elapsed:.2f}s — {len(merged_tokens)} tokens (after cleanup)")

        return {
            "tokens": merged_tokens,
            "shap_values": merged_shap_dict,
            "base_values": base_dict,
            "predicted_class": predicted_class,
            "predicted_proba": predicted_proba,
            "all_probabilities": all_probs,
            "elapsed_seconds": round(elapsed, 3),
        }

    def explain_targets(
        self, targets: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run SHAP explanation on all selected targets (positive, negative, overall).

        Args:
            targets: Dict from select_target_sections().

        Returns:
            Dict keyed by target type, each containing section metadata + SHAP explanation.
        """
        total_start = time.time()
        results = {}

        for target_key, target_data in targets.items():
            logger.info(f"[SHAP] Explaining '{target_key}' target: {target_data['section_name']}...")
            explanation = self.explain_text(target_data["text"])

            # Log mismatch between expected and actual prediction
            expected_class = target_key if target_key in ("positive", "negative") else None
            actual_class = explanation["predicted_class"]
            if expected_class and actual_class != expected_class:
                logger.warning(
                    f"[SHAP] Class mismatch for '{target_key}' target: "
                    f"expected={expected_class}, actual={actual_class} "
                    f"(probs={explanation['all_probabilities']})"
                )

            results[target_key] = {
                **target_data,
                "explanation": explanation,
            }

        total_elapsed = time.time() - total_start
        logger.info(f"[SHAP] All {len(results)} explanations completed in {total_elapsed:.2f}s")
        return results

    @staticmethod
    def _clean_and_merge(tokens: List[str], shap_dict: Dict[str, List[float]]) -> Tuple[List[str], Dict[str, List[float]]]:
        """
        Merge BERT subwords (e.g. 'profit' + '##ability' -> 'profitability'), sum their SHAP values,
        and remove stopwords, punctuation, and special tokens.
        """
        merged_tokens = []
        merged_vals = {k: [] for k in shap_dict.keys()}
        
        curr_tok = ""
        curr_v = {k: 0.0 for k in shap_dict.keys()}
        
        for i, tok in enumerate(tokens):
            tok_clean = tok.strip()
            if not tok_clean:
                continue
                
            if tok_clean.startswith("##"):
                if curr_tok:
                    curr_tok += tok_clean[2:]
                    for k in shap_dict.keys():
                        curr_v[k] += shap_dict[k][i]
                else:
                    curr_tok = tok_clean[2:]
                    for k in shap_dict.keys():
                        curr_v[k] = shap_dict[k][i]
            else:
                if curr_tok:
                    merged_tokens.append(curr_tok)
                    for k in shap_dict.keys():
                        merged_vals[k].append(curr_v[k])
                curr_tok = tok_clean
                for k in shap_dict.keys():
                    curr_v[k] = shap_dict[k][i]
                    
        if curr_tok:
            merged_tokens.append(curr_tok)
            for k in shap_dict.keys():
                merged_vals[k].append(curr_v[k])
                
        # Filter stopwords, punctuation, special tokens
        STOPWORDS = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "as", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "shall", "should", "can", "could", "may", "might",
            "must", "it", "its", "they", "their", "them", "we", "our", "us", "i", "my", "me", "you",
            "your", "this", "that", "these", "those", "which", "who", "whom", "whose", "what", "where",
            "when", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
            "can", "will", "just", "don", "should", "now", "from", "into", "about", "against", "between",
            "into", "through", "during", "before", "after", "above", "below", "up", "down", "out", "off",
            "over", "under", "again", "further", "then", "once", "here", "there"
        }
        
        final_tokens = []
        final_vals = {k: [] for k in shap_dict.keys()}
        
        for i, tok in enumerate(merged_tokens):
            t = tok.lower()
            if t in STOPWORDS or len(t) <= 1 or all(c in string.punctuation for c in t):
                continue
            if t in ["[cls]", "[sep]", "[pad]", "[mask]", "[unk]"]:
                continue
            
            final_tokens.append(tok)
            for k in shap_dict.keys():
                final_vals[k].append(merged_vals[k][i])
                
        return final_tokens, final_vals

    @staticmethod
    def generate_nl_explanation(predicted_class: str, pos_drivers: List[Dict], neg_drivers: List[Dict]) -> str:
        """
        Generate a short natural-language explanation summarizing the key drivers.
        """
        if predicted_class == "positive":
            drivers = [d["token"] for d in pos_drivers[:4]]
            if not drivers:
                return "FinBERT predicted Positive, but no strong meaningful drivers were identified."
            if len(drivers) > 1:
                return f"FinBERT predicted Positive mainly due to {', '.join(drivers[:-1])}, and {drivers[-1]}."
            return f"FinBERT predicted Positive mainly due to {drivers[0]}."
        elif predicted_class == "negative":
            drivers = [d["token"] for d in neg_drivers[:4]]
            if not drivers:
                return "FinBERT predicted Negative, but no strong meaningful drivers were identified."
            if len(drivers) > 1:
                return f"FinBERT predicted Negative mainly due to {', '.join(drivers[:-1])}, and {drivers[-1]}."
            return f"FinBERT predicted Negative mainly due to {drivers[0]}."
        else:
            pos_d = [d["token"] for d in pos_drivers[:2]]
            neg_d = [d["token"] for d in neg_drivers[:2]]
            if pos_d and neg_d:
                return f"FinBERT predicted Neutral due to a balance between positive factors ({', '.join(pos_d)}) and negative factors ({', '.join(neg_d)})."
            return "FinBERT predicted Neutral as no strong directional drivers were present."

    # -------------------------------------------------------------------------
    # Token Contribution Extraction for Dashboard
    # -------------------------------------------------------------------------

    @staticmethod
    def extract_top_drivers(
        explanation: Dict[str, Any],
        sentiment_class: str,
        top_k: int = 10,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract the top positive and negative driver tokens for a given sentiment class.

        Args:
            explanation: SHAP explanation dict from explain_text().
            sentiment_class: Which class to analyze ('positive', 'negative', 'neutral').
            top_k: Number of top drivers to return per direction.

        Returns:
            Tuple of (top_positive_drivers, top_negative_drivers).
        """
        tokens = explanation.get("tokens", [])
        values = explanation.get("shap_values", {}).get(sentiment_class, [])

        if not tokens or not values:
            return [], []

        # Aggregate contributions for repeated tokens
        aggregated: Dict[str, float] = defaultdict(float)
        for tok, val in zip(tokens, values):
            tok_clean = tok.strip()
            if tok_clean:
                aggregated[tok_clean] += float(val)

        sorted_contribs = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)

        positive_drivers = [
            {"token": tok, "contribution": val}
            for tok, val in sorted_contribs if val > 0
        ][:top_k]

        negative_drivers = [
            {"token": tok, "contribution": val}
            for tok, val in sorted(aggregated.items(), key=lambda x: x[1])
            if val < 0
        ][:top_k]

        return positive_drivers, negative_drivers

    @staticmethod
    def get_overall_document_drivers(
        shap_results: Dict[str, Dict[str, Any]],
        overall_sentiment: str,
        top_k: int = 10,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Aggregate token-level drivers across all explained targets to produce
        the overall top positive-sentiment and negative-sentiment word drivers.

        Uses the "positive" class SHAP values for positive drivers and
        the "negative" class SHAP values for negative drivers.

        Args:
            shap_results: Full SHAP results dict from explain_targets().
            overall_sentiment: Document-level sentiment label.
            top_k: Number of top drivers per direction.

        Returns:
            Tuple of (top_positive_drivers, top_negative_drivers).
        """
        combined_pos: Dict[str, float] = defaultdict(float)
        combined_neg: Dict[str, float] = defaultdict(float)

        for _key, result_data in shap_results.items():
            explanation = result_data.get("explanation", {})
            tokens = explanation.get("tokens", [])

            # Tokens that push POSITIVE class probability UP
            pos_values = explanation.get("shap_values", {}).get("positive", [])
            for tok, val in zip(tokens, pos_values):
                tok_clean = tok.strip().lower()
                if tok_clean:
                    combined_pos[tok_clean] += float(val)

            # Tokens that push NEGATIVE class probability UP
            neg_values = explanation.get("shap_values", {}).get("negative", [])
            for tok, val in zip(tokens, neg_values):
                tok_clean = tok.strip().lower()
                if tok_clean:
                    combined_neg[tok_clean] += float(val)

        # Top positive drivers: words that most increase positive probability
        pos_drivers = sorted(
            [{"token": tok, "contribution": val} for tok, val in combined_pos.items() if val > 0],
            key=lambda x: x["contribution"],
            reverse=True,
        )[:top_k]

        # Top negative drivers: words that most increase negative probability
        neg_drivers = sorted(
            [{"token": tok, "contribution": val} for tok, val in combined_neg.items() if val > 0],
            key=lambda x: x["contribution"],
            reverse=True,
        )[:top_k]

        return pos_drivers, neg_drivers
