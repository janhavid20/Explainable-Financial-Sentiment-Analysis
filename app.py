"""
Streamlit Web Application: Explainable Financial Sentiment Analysis Using FinBERT
Professional, Minimal, Editorial Financial Research Interface
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("financial_analyst_app")

from src.ingestion import PDFExtractor
from src.segmentation import TextChunker, SectionDetector
from src.models import FinBERTClassifier
from src.aggregation import SentimentAggregator
from src.explainability import SHAPExplainer
from config import SHAP_TOP_K_DRIVERS, PRIORITY_HIGH_RISK_THRESHOLD, PRIORITY_WATCH_THRESHOLD

# -----------------------------------------------------------------------------
# Page Configuration & Dark Editorial Design System
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Financial Document Analyst",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Editorial Theme CSS
# Background: #101216 | Surfaces: #191C22 | Borders: #30343D | Text: #F1F2F4 / #A1A6B0
st.markdown("""
<style>
    /* ── Global Theme & Typography ─────────────────────────────────────────── */
    .stApp {
        background-color: #101216 !important;
        color: #F1F2F4 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* ── Sidebar (230 px) ─────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: #191C22 !important;
        border-right: 1px solid #30343D !important;
        min-width: 230px !important;
        max-width: 230px !important;
    }
    [data-testid="stSidebar"] * {
        color: #F1F2F4;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: #A1A6B0;
        font-size: 0.85rem;
    }
    [data-testid="stSidebar"] code {
        word-break: break-all;
        white-space: pre-wrap;
        background-color: #14171D !important;
        border: 1px solid #30343D !important;
        color: #F1F2F4 !important;
    }

    /* ── Main Content Container ───────────────────────────────────────────── */
    .main .block-container {
        max-width: 1040px;
        padding-top: 1.5rem;
        padding-bottom: 120px !important;
    }

    /* ── Editorial Header ─────────────────────────────────────────────────── */
    .app-header {
        padding: 4px 0 16px 0;
        border-bottom: 1px solid #30343D;
        margin-bottom: 20px;
    }
    .app-title {
        font-size: 1.55rem;
        font-weight: 600;
        color: #F1F2F4;
        letter-spacing: -0.015em;
        margin: 0;
        line-height: 1.25;
    }
    .app-subtitle {
        font-size: 0.82rem;
        color: #A1A6B0;
        margin: 5px 0 0 0;
    }

    /* ── Welcome / Ingestion Container ───────────────────────────────────── */
    .welcome-container {
        background-color: #191C22;
        border: 1px solid #30343D;
        border-radius: 8px;
        padding: 28px 24px;
        max-width: 720px;
        margin: 20px auto 28px auto;
        text-align: center;
    }
    .welcome-title {
        font-size: 1.45rem;
        font-weight: 600;
        color: #F1F2F4;
        margin-bottom: 6px;
        letter-spacing: -0.01em;
    }
    .welcome-subtitle {
        font-size: 0.88rem;
        color: #A1A6B0;
        line-height: 1.55;
    }

    /* ── Surfaces & Editorial Panels ─────────────────────────────────────── */
    .dark-card {
        background-color: #191C22;
        border: 1px solid #30343D;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    .editorial-subheading {
        font-size: 0.92rem;
        font-weight: 600;
        color: #F1F2F4;
        letter-spacing: -0.005em;
    }

    /* ── Compact Metric Cards ─────────────────────────────────────────────── */
    .metric-card-dark {
        background-color: #191C22;
        border: 1px solid #30343D;
        border-radius: 8px;
        padding: 14px 16px;
        min-height: 90px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .metric-card-label {
        color: #A1A6B0;
        font-size: 0.72rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .metric-card-value {
        font-size: 1.35rem;
        font-weight: 600;
        margin: 4px 0 2px 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .metric-card-sub {
        color: #A1A6B0;
        font-size: 0.73rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* ── Restrained Sentiment Badges ──────────────────────────────────────── */
    .badge-dark {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.74rem;
        font-weight: 500;
        letter-spacing: 0.01em;
        line-height: 1.4;
    }
    .badge-pos {
        background-color: rgba(85, 184, 154, 0.12);
        color: #55B89A;
        border: 1px solid rgba(85, 184, 154, 0.32);
    }
    .badge-neg {
        background-color: rgba(200, 107, 112, 0.12);
        color: #C86B70;
        border: 1px solid rgba(200, 107, 112, 0.32);
    }
    .badge-neu {
        background-color: rgba(146, 153, 165, 0.12);
        color: #9299A5;
        border: 1px solid rgba(146, 153, 165, 0.32);
    }
    .badge-watch {
        background-color: rgba(212, 163, 89, 0.12);
        color: #D4A359;
        border: 1px solid rgba(212, 163, 89, 0.32);
    }

    /* ── SHAP Token Highlights ───────────────────────────────────────────── */
    .shap-container-dark {
        background-color: #14171D;
        border: 1px solid #30343D;
        border-radius: 6px;
        padding: 12px 14px;
        line-height: 2.1;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 0.85rem;
        margin: 8px 0;
        color: #F1F2F4;
    }
    .shap-token-dark {
        padding: 2px 5px;
        border-radius: 3px;
        margin: 1px;
        display: inline;
        border: 1px solid transparent;
    }

    /* ── Text Evidence Snippets ──────────────────────────────────────────── */
    .snippet-box-dark {
        background-color: #14171D;
        border-left: 3px solid #30343D;
        padding: 10px 14px;
        border-radius: 0 6px 6px 0;
        font-size: 0.84rem;
        line-height: 1.55;
        color: #F1F2F4;
        margin: 8px 0;
    }
    .snippet-box-risk {
        border-left-color: #C86B70;
        background-color: rgba(200, 107, 112, 0.06);
    }
    .snippet-box-pos {
        border-left-color: #55B89A;
        background-color: rgba(85, 184, 154, 0.06);
    }

    /* ── Disclosure Banner ───────────────────────────────────────────────── */
    .disclosure-banner {
        background-color: #14171D;
        border: 1px solid #30343D;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 0.77rem;
        color: #A1A6B0;
        margin-top: 14px;
        line-height: 1.5;
    }

    /* ── Findings Table ──────────────────────────────────────────────────── */
    .findings-table {
        background-color: #191C22;
        border: 1px solid #30343D;
        border-radius: 6px;
        overflow: hidden;
        margin-bottom: 14px;
    }
    .findings-row {
        display: grid;
        grid-template-columns: 2fr 100px 100px 90px 85px;
        gap: 0;
        align-items: center;
        padding: 8px 14px;
        border-bottom: 1px solid #30343D;
        font-size: 0.84rem;
    }
    .findings-row:last-child {
        border-bottom: none;
    }
    .findings-header {
        background-color: #14171D;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #A1A6B0;
    }

    /* ── Streamlit UI Overrides ──────────────────────────────────────────── */
    div[data-testid="stExpander"] {
        background-color: #191C22 !important;
        border: 1px solid #30343D !important;
        border-radius: 6px !important;
        margin-bottom: 8px !important;
    }
    div[data-testid="stExpander"] * {
        color: #F1F2F4 !important;
    }

    .stButton > button {
        background-color: #191C22;
        color: #F1F2F4;
        border: 1px solid #30343D;
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.82rem;
        padding: 6px 14px;
    }
    .stButton > button:hover {
        background-color: #21252D;
        border-color: #4A4F5C;
        color: #FFFFFF;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #14171D;
        border-radius: 6px;
        padding: 3px;
        gap: 4px;
        border: 1px solid #30343D;
        margin-bottom: 18px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 4px;
        color: #A1A6B0 !important;
        font-weight: 500;
        font-size: 0.86rem;
        padding: 6px 16px;
        border: none !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #F1F2F4 !important;
        background-color: #191C22;
    }
    .stTabs [aria-selected="true"] {
        background-color: #191C22 !important;
        color: #F1F2F4 !important;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* Chat Messages & Follow-Up Input */
    .chat-msg-user {
        background-color: #1D2128;
        border: 1px solid #30343D;
        border-radius: 6px;
        padding: 9px 14px;
        margin: 12px 0 8px auto;
        max-width: 80%;
        color: #F1F2F4;
        font-size: 0.87rem;
        font-weight: 400;
        display: table;
    }
    .chat-msg-assistant {
        background-color: #191C22;
        border: 1px solid #30343D;
        border-radius: 8px;
        padding: 18px 20px;
        margin: 8px 0 16px 0;
        color: #F1F2F4;
    }

    [data-testid="stChatInput"] {
        background-color: #191C22 !important;
        border: 1px solid #30343D !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4) !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #F1F2F4 !important;
        background-color: transparent !important;
        font-size: 0.86rem !important;
    }

    hr {
        border: none;
        border-top: 1px solid #30343D;
        margin: 18px 0;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Model Loader
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_finbert_model():
    """Loads and caches the FinBERT classification model in memory."""
    return FinBERTClassifier()


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
def init_session_state():
    """Initializes session state keys for persistent document analysis and conversation."""
    if "doc_state" not in st.session_state:
        st.session_state.doc_state = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "active_file_signature" not in st.session_state:
        st.session_state.active_file_signature = None

init_session_state()


# -----------------------------------------------------------------------------
# Document Processing Pipeline (Executed once per PDF upload)
# -----------------------------------------------------------------------------
def run_document_analysis(pdf_bytes: bytes, filename: str, chunk_size: int, overlap_words: int) -> Dict[str, Any]:
    """
    Executes the analytical pipeline:
    1. Validation -> 2. PyMuPDF Extraction -> 3. Section Detection -> 4. Chunking ->
    5. FinBERT Inference -> 6. Sentiment Aggregation -> 7. Targeted SHAP Attribution
    """
    pipeline_start = time.time()
    timings = {}
    
    progress_bar = st.progress(5, text="Validating PDF document...")
    
    # Stage 1: Validation
    s1_start = time.time()
    if len(pdf_bytes) < 100:
        raise ValueError(f"Uploaded file is too small ({len(pdf_bytes)} bytes) and appears invalid or corrupt.")
    timings["1_upload_validation"] = round(time.time() - s1_start, 3)
    
    # Stage 2: Text Extraction
    progress_bar.progress(15, text="Extracting text from PDF (PyMuPDF)...")
    s2_start = time.time()
    extraction_result = PDFExtractor.extract_text_from_bytes(pdf_bytes, timeout_seconds=40.0)
    full_text = extraction_result["full_text"]
    if not full_text.strip():
        raise ValueError("No extractable text found in PDF. Scanned documents require OCR.")
    timings["2_text_extraction"] = round(time.time() - s2_start, 3)
    
    # Stage 3: Section Detection
    progress_bar.progress(25, text="Detecting report sections...")
    s3_start = time.time()
    detector = SectionDetector()
    sections = detector.detect_sections(full_text)
    timings["3_section_detection"] = round(time.time() - s3_start, 3)
    
    # Stage 4: Text Chunking
    progress_bar.progress(35, text="Chunking sections into sentence-aware blocks...")
    s4_start = time.time()
    chunker = TextChunker(target_words_per_chunk=chunk_size, overlap_words=overlap_words)
    chunks = chunker.chunk_sections(sections)
    if not chunks:
        raise ValueError("Text chunking produced 0 chunks from the extracted content.")
    timings["4_text_chunking"] = round(time.time() - s4_start, 3)
    
    # Stage 5: FinBERT Inference
    progress_bar.progress(45, text="Loading FinBERT weights...")
    s5a_start = time.time()
    classifier = load_finbert_model()
    timings["5a_model_loading"] = round(time.time() - s5a_start, 3)
    
    def on_progress(done: int, total: int):
        pct = int(45 + (done / total) * 20)
        progress_bar.progress(pct, text=f"FinBERT batch inference ({done}/{total} chunks)...")
    
    s5b_start = time.time()
    annotated_chunks = classifier.predict_chunks(chunks, progress_callback=on_progress)
    timings["5b_inference"] = round(time.time() - s5b_start, 3)
    
    # Stage 6: Aggregation
    progress_bar.progress(70, text="Aggregating document and section metrics...")
    s6_start = time.time()
    aggregation = SentimentAggregator.aggregate_document_sentiment(annotated_chunks)
    section_summaries = SentimentAggregator.aggregate_by_section(annotated_chunks)
    timings["6_aggregation"] = round(time.time() - s6_start, 3)
    
    # Stage 7: SHAP Explainability
    progress_bar.progress(80, text="Computing targeted SHAP token attributions...")
    s7_start = time.time()
    shap_results = None
    shap_pos_drivers, shap_neg_drivers = [], []
    
    try:
        shap_explainer = SHAPExplainer()
        shap_targets = SHAPExplainer.select_target_sections(
            section_summaries, annotated_chunks, aggregation["overall_sentiment"]
        )
        if shap_targets:
            shap_results = shap_explainer.explain_targets(shap_targets)
            shap_pos_drivers, shap_neg_drivers = SHAPExplainer.get_overall_document_drivers(
                shap_results, aggregation["overall_sentiment"], top_k=SHAP_TOP_K_DRIVERS
            )
        timings["7_shap_explainability"] = round(time.time() - s7_start, 3)
    except Exception as e:
        logger.warning(f"SHAP explanation computation error: {e}")
        timings["7_shap_explainability"] = round(time.time() - s7_start, 3)
    
    progress_bar.progress(100, text="Analysis complete.")
    time.sleep(0.3)
    progress_bar.empty()
    
    timings["total_pipeline"] = round(time.time() - pipeline_start, 2)
    
    return {
        "filename": filename,
        "extraction_result": extraction_result,
        "sections": sections,
        "chunks": chunks,
        "annotated_chunks": annotated_chunks,
        "aggregation": aggregation,
        "section_summaries": section_summaries,
        "shap_results": shap_results,
        "shap_pos_drivers": shap_pos_drivers,
        "shap_neg_drivers": shap_neg_drivers,
        "timings": timings,
    }


# -----------------------------------------------------------------------------
# Formatting Helpers
# -----------------------------------------------------------------------------
def format_sentiment_badge(label: str) -> str:
    """Returns an HTML badge for a sentiment label."""
    norm = label.capitalize()
    cls_map = {"Positive": "badge-pos", "Negative": "badge-neg", "Neutral": "badge-neu"}
    return f'<span class="badge-dark {cls_map.get(norm, "badge-neu")}">{norm}</span>'

def format_priority_badge(priority: str) -> str:
    """Returns an HTML badge for section risk priority."""
    cls_map = {"High Risk": "badge-neg", "Watch": "badge-watch", "Normal": "badge-pos"}
    return f'<span class="badge-dark {cls_map.get(priority, "badge-neu")}">{priority}</span>'

def render_token_visualization_html(explanation: dict, target_class: str) -> str:
    """Builds readable HTML for SHAP token attributions in restrained dark editorial theme."""
    tokens = explanation.get("tokens", [])
    values = explanation.get("shap_values", {}).get(target_class, [])
    if not tokens or not values:
        return "<em style='color:#A1A6B0;'>No token-level attribution available.</em>"
    
    abs_max = max(abs(v) for v in values) if values else 1.0
    if abs_max == 0:
        abs_max = 1.0
    
    parts = []
    for tok, val in zip(tokens, values):
        norm = val / abs_max
        if norm > 0.05:
            # Pushes towards target class -> muted green #55B89A
            alpha = min(max(norm * 0.35, 0.12), 0.45)
            bg = f"rgba(85, 184, 154, {alpha:.2f})"
            color = "#8CE0C6"
            border = "rgba(85, 184, 154, 0.35)"
        elif norm < -0.05:
            # Pushes away from target class -> muted red #C86B70
            alpha = min(max(abs(norm) * 0.35, 0.12), 0.45)
            bg = f"rgba(200, 107, 112, {alpha:.2f})"
            color = "#EAA8AB"
            border = "rgba(200, 107, 112, 0.35)"
        else:
            bg = "rgba(146, 153, 165, 0.06)"
            color = "#9299A5"
            border = "transparent"
        
        parts.append(
            f'<span class="shap-token-dark" style="background:{bg}; color:{color}; border-color:{border};" '
            f'title="SHAP attribution: {val:+.4f}">{tok}</span>'
        )
    
    return f'<div class="shap-container-dark">{" ".join(parts)}</div>'


# -----------------------------------------------------------------------------
# Plotly Dark Editorial Layout Helper
# -----------------------------------------------------------------------------
def _dark_layout(**overrides) -> dict:
    """Returns a consistent Plotly layout dict for the restrained editorial dark theme."""
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif", color="#F1F2F4", size=12),
        margin=dict(t=24, b=24, l=36, r=16),
        legend=dict(
            bgcolor="rgba(25,28,34,0.9)",
            bordercolor="#30343D",
            borderwidth=1,
            font=dict(color="#A1A6B0", size=11),
        ),
        xaxis=dict(
            gridcolor="#30343D",
            zerolinecolor="#30343D",
            tickfont=dict(color="#A1A6B0", size=11),
            title_font=dict(color="#F1F2F4", size=12),
        ),
        yaxis=dict(
            gridcolor="#30343D",
            zerolinecolor="#30343D",
            tickfont=dict(color="#A1A6B0", size=11),
            title_font=dict(color="#F1F2F4", size=12),
        ),
    )
    base.update(overrides)
    return base


# -----------------------------------------------------------------------------
# Query Routing & Conversational Response Generator (Cached, No Re-Inference)
# -----------------------------------------------------------------------------
def generate_conversational_response(user_query: str, doc_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Answers natural language queries about the analyzed document using cached inference results.
    Never reruns heavy model inference. Routes intent strictly to evidence-based responses.
    """
    q = user_query.lower().strip()
    agg = doc_state["aggregation"]
    summaries = doc_state["section_summaries"]
    annotated = doc_state["annotated_chunks"]
    shap_res = doc_state.get("shap_results") or {}
    pos_drivers = doc_state.get("shap_pos_drivers") or []
    neg_drivers = doc_state.get("shap_neg_drivers") or []
    
    # 1. Intent: Specific detected section query
    for sec in summaries:
        s_name = sec["section_name"].lower()
        if s_name in q or (len(s_name) > 4 and s_name.replace("&", "and") in q.replace("&", "and")):
            sec_chunks = [c for c in annotated if c.get("section_name") == sec["section_name"]]
            return {
                "intent": "section_detail",
                "title": f"Section Spotlight: {sec['section_name']}",
                "section": sec,
                "chunks": sec_chunks,
                "shap_results": shap_res
            }

    # 2. Intent: Identify Most Concerning Sections / Risk Analysis
    if any(k in q for k in ["concerning", "risk", "hazard", "threat", "headwind", "watch", "worst"]):
        risk_sections = [s for s in summaries if s["priority"] in ("High Risk", "Watch")]
        if not risk_sections:
            risk_sections = sorted(summaries, key=lambda s: s["mean_negative"], reverse=True)[:2]
        return {
            "intent": "risk_analysis",
            "title": "Risk Analysis & Concerning Sections",
            "risk_sections": risk_sections,
            "overall_nsi": agg["net_sentiment_score"],
            "neg_drivers": neg_drivers,
            "all_summaries": summaries,
            "doc_state": doc_state,
        }
        
    # 3. Intent: Summarize Positive and Negative Findings
    if any(k in q for k in ["positive and negative", "summarize", "summary", "pros and cons", "findings", "balance"]):
        top_pos = sorted(summaries, key=lambda s: s["mean_positive"], reverse=True)[:3]
        top_neg = sorted(summaries, key=lambda s: s["mean_negative"], reverse=True)[:3]
        return {
            "intent": "summary_analysis",
            "title": "Positive Highlights & Cautionary Findings",
            "top_positive": top_pos,
            "top_negative": top_neg,
            "agg": agg,
            "pos_drivers": pos_drivers,
            "neg_drivers": neg_drivers
        }
        
    # 4. Intent: Explain Why a Section Received Its Sentiment (SHAP)
    if any(k in q for k in ["explain", "why", "shap", "driver", "attribution", "token"]):
        matched_target = None
        for key, tdata in shap_res.items():
            sec_name = tdata.get("section_name", "").lower()
            if sec_name and (sec_name in q or any(w in q for w in sec_name.split())):
                matched_target = key
                break
        
        return {
            "intent": "explanation_analysis",
            "title": "FinBERT Sentiment Explanations (SHAP Attribution)",
            "shap_results": shap_res,
            "matched_target": matched_target,
            "pos_drivers": pos_drivers,
            "neg_drivers": neg_drivers
        }
            
    # 5. Intent: Default / Full Report Analysis ("Analyze this report")
    return {
        "intent": "full_analysis",
        "title": "Comprehensive Financial Sentiment & Risk Analysis",
        "doc_state": doc_state
    }


# -----------------------------------------------------------------------------
# Three-Tab Analysis Renderers
# -----------------------------------------------------------------------------

def _render_tab_report_overview(doc_state: Dict[str, Any]):
    """
    Tab 1: Narrative-first overview designed for research analysts.
    Organized into strict editorial hierarchy:
    1. Executive summary in plain English.
    2. Overall sentiment and concise explanation of tone.
    3. Three compact metrics: overall sentiment, model confidence, analyzed coverage.
    4. Key observations highlighting positive and negative findings.
    5. Clean sentiment-distribution chart.
    6. Brief methodological limitation note.
    """
    agg = doc_state["aggregation"]
    summaries = doc_state["section_summaries"]
    extraction = doc_state["extraction_result"]
    
    overall = agg["overall_sentiment"]
    conf = agg["overall_confidence"]
    chunks_count = agg["total_chunks"]
    risk_sections = [s for s in summaries if s["priority"] in ("High Risk", "Watch")]
    
    # Color mapping for sentiment
    if overall == "Positive":
        sentiment_meaning = (
            "The document exhibits a predominantly constructive tone. Narrative disclosures "
            "frequently reflect operational expansion, revenue growth, or affirmative strategic progress."
        )
        overall_color = "#55B89A"
    elif overall == "Negative":
        sentiment_meaning = (
            "The document exhibits a cautious and risk-sensitive tone. Narrative disclosures "
            "frequently detail regulatory obligations, competitive headwinds, operational risks, or market uncertainties."
        )
        overall_color = "#C86B70"
    else:
        sentiment_meaning = (
            "The document exhibits a balanced and objective reporting tone. Narrative disclosures "
            "primarily comprise factual descriptions, routine corporate governance, and measured status updates."
        )
        overall_color = "#9299A5"

    # 1. Executive Summary in plain English
    st.markdown("### Executive Summary")
    st.markdown(
        f"This report presents an automated financial sentiment analysis of `{doc_state['filename']}`. "
        f"The filing comprises **{extraction['total_pages']} pages** and **{extraction['total_word_count']:,} words**, "
        f"partitioned across **{len(summaries)} distinct sections** and **{chunks_count} analyzed passages** without truncation."
    )

    # 2. Overall Sentiment & Contextual Interpretation
    st.markdown(
        f"<div class='dark-card' style='border-left: 3px solid {overall_color};'>"
        f"<div class='editorial-subheading' style='color:{overall_color};'>Overall Sentiment: {overall}</div>"
        f"<div style='margin-top:5px;color:#F1F2F4;line-height:1.55;font-size:0.9rem;'>{sentiment_meaning}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    # 3. Three Compact Metrics: Overall Sentiment, Model Confidence, Analyzed Coverage
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f'<div class="metric-card-dark">'
            f'<div class="metric-card-label">Overall Sentiment</div>'
            f'<div class="metric-card-value" style="color:{overall_color};">{overall}</div>'
            f'<div class="metric-card-sub">Dominant classification</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with m2:
        st.markdown(
            f'<div class="metric-card-dark">'
            f'<div class="metric-card-label">Model Confidence</div>'
            f'<div class="metric-card-value" style="color:#F1F2F4;">{conf*100:.1f}%</div>'
            f'<div class="metric-card-sub">Mean class probability agreement</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with m3:
        st.markdown(
            f'<div class="metric-card-dark">'
            f'<div class="metric-card-label">Analyzed Coverage</div>'
            f'<div class="metric-card-value" style="color:#F1F2F4;">{chunks_count} Passages</div>'
            f'<div class="metric-card-sub">{len(summaries)} sections &middot; {extraction["total_pages"]} pages</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 4 & 5. Key Observations & Sentiment Distribution Chart
    col_chart, col_obs = st.columns([1, 1])

    with col_chart:
        st.markdown("##### Passage Tone Distribution")
        dist_df = pd.DataFrame([
            {"Sentiment": k, "Passages": v}
            for k, v in agg["sentiment_distribution"].items()
        ])
        SENT_COLORS = {"Positive": "#55B89A", "Negative": "#C86B70", "Neutral": "#9299A5"}
        fig_pie = px.pie(
            dist_df, names="Sentiment", values="Passages", color="Sentiment",
            color_discrete_map=SENT_COLORS, hole=0.52
        )
        fig_pie.update_traces(
            textposition="inside",
            textinfo="percent+label",
            textfont=dict(color="#F1F2F4", size=12),
            marker=dict(line=dict(color="#101216", width=2)),
        )
        fig_pie.update_layout(
            **_dark_layout(height=230, showlegend=False, margin=dict(t=15, b=10, l=10, r=10))
        )
        st.plotly_chart(fig_pie, width="stretch")
        st.caption("Distribution of segmented document passages categorized by tone.")

    with col_obs:
        st.markdown("##### Key Observations")
        
        top_pos = max(summaries, key=lambda s: s["mean_positive"]) if summaries else None
        top_neg = max(summaries, key=lambda s: s["mean_negative"]) if summaries else None
        
        if top_pos and top_pos["mean_positive"] > 0.35:
            st.markdown(
                f"**Constructive Highlights:** `{top_pos['section_name']}` demonstrates the highest concentration of affirmative statements "
                f"({format_sentiment_badge(top_pos['sentiment'])}, {top_pos['chunk_count']} passages analyzed).",
                unsafe_allow_html=True
            )
        else:
            st.markdown("**Constructive Highlights:** Narrative maintains consistent positive statements without extreme upward deviation.")
            
        if risk_sections:
            flagged_names = ", ".join(f"`{s['section_name']}`" for s in risk_sections[:2])
            st.markdown(
                f"**Cautionary Disclosures:** Concentrated cautionary language identified in {flagged_names}. "
                f"These sections reflect standard regulatory risk and uncertainty disclosures.",
                unsafe_allow_html=True
            )
        elif top_neg and top_neg["mean_negative"] > 0.2:
            st.markdown(
                f"**Cautionary Disclosures:** `{top_neg['section_name']}` contains the highest concentration of cautionary disclosures, "
                f"within standard reporting parameters.",
                unsafe_allow_html=True
            )
        else:
            st.markdown("**Cautionary Disclosures:** No section exhibited elevated cautionary language thresholds.")
            
        st.markdown(
            f"**Coverage Integrity:** 100% of document content was processed across {chunks_count} continuous passages without truncation."
        )

    # 6. Brief Limitation Note
    st.markdown(
        "<div class='disclosure-banner'>"
        "<strong>Methodological Limitation:</strong> This assessment evaluates linguistic sentiment using FinBERT, "
        "a domain-adapted language model trained on corporate communications. "
        "Mandatory regulatory disclosures (such as Risk Factors or legal proceedings) typically require cautionary language "
        "and do not constitute independently audited evidence of financial distress or business insolvency."
        "</div>",
        unsafe_allow_html=True
    )


def _render_tab_key_findings(doc_state: Dict[str, Any]):
    """
    Tab 2: Evidence-led section-level presentation.
    Structured table with readable alignment and clean expanders for direct text evidence.
    Explains priority thresholds once and avoids redundant risk cards.
    """
    summaries = doc_state["section_summaries"]
    annotated = doc_state["annotated_chunks"]
    
    st.markdown("### Section Findings and Evidence")
    st.markdown(
        "Section-level sentiment classifications, risk priorities, and verbatim text excerpts from the source document."
    )
    
    # Priority Threshold Explanation Note (Explained once)
    st.markdown(
        f"<div class='disclosure-banner' style='margin-bottom: 14px; margin-top: 4px;'>"
        f"<strong>Priority Thresholds:</strong> Sections are categorized by cautionary sentence density: "
        f"Negative concentration &gt;<strong>{PRIORITY_HIGH_RISK_THRESHOLD*100:.0f}%</strong> &rarr; <strong>High Risk</strong>; "
        f"between <strong>{PRIORITY_WATCH_THRESHOLD*100:.0f}%</strong> and <strong>{PRIORITY_HIGH_RISK_THRESHOLD*100:.0f}%</strong> &rarr; <strong>Watch</strong>; "
        f"below <strong>{PRIORITY_WATCH_THRESHOLD*100:.0f}%</strong> &rarr; <strong>Normal</strong>. "
        f"These flags identify concentrated risk language and do not imply verified commercial insolvency."
        f"</div>",
        unsafe_allow_html=True
    )
    
    # Section Summary Table
    rows_html = ""
    for s in summaries:
        sent_badge = format_sentiment_badge(s["sentiment"])
        pri_badge = format_priority_badge(s["priority"])
        rows_html += (
            f'<div class="findings-row">'
            f'<span style="color:#F1F2F4;font-weight:500;">{s["section_name"]}</span>'
            f'<span>{sent_badge}</span>'
            f'<span>{pri_badge}</span>'
            f'<span style="color:#A1A6B0;">{s["confidence"]*100:.1f}%</span>'
            f'<span style="color:#A1A6B0;">{s["chunk_count"]} passages</span>'
            f'</div>'
        )
    st.markdown(
        '<div class="findings-table">'
        '<div class="findings-row findings-header">'
        '<span>Section</span><span>Sentiment</span><span>Priority</span>'
        '<span>Certainty</span><span>Passages</span>'
        '</div>'
        + rows_html + '</div>',
        unsafe_allow_html=True
    )
    
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    
    # Expandable Section Evidence Excerpts
    st.markdown("##### Verbatim Evidence Excerpts")
    st.caption("Expand any section to inspect the individual passages analyzed:")
    
    for s in summaries:
        with st.expander(f"{s['section_name']} — {s['sentiment']} ({s['priority']}, {s['chunk_count']} passages)", expanded=False):
            st.markdown(
                f"**Sentiment:** {format_sentiment_badge(s['sentiment'])} &nbsp;|&nbsp; "
                f"**Priority:** {format_priority_badge(s['priority'])} &nbsp;|&nbsp; "
                f"**Certainty:** `{s['confidence']*100:.1f}%` &nbsp;|&nbsp; "
                f"**Cautionary Density:** `{s['mean_negative']*100:.1f}%` &nbsp;|&nbsp; "
                f"**NSI:** `{s['net_sentiment_score']:+.3f}`",
                unsafe_allow_html=True
            )
            sec_chunks = [c for c in annotated if c.get("section_name") == s["section_name"]]
            for sc in sec_chunks:
                box_cls = "snippet-box-pos" if sc["predicted_label"] == "Positive" else (
                    "snippet-box-risk" if sc["predicted_label"] == "Negative" else "snippet-box-dark"
                )
                st.markdown(
                    f"<div class='{box_cls}'>"
                    f"<strong>Passage #{sc['chunk_id']}</strong> "
                    f"({sc['predicted_label']} — Certainty: {sc['confidence_score']*100:.1f}%)<br/>"
                    f"{sc['text']}"
                    f"</div>",
                    unsafe_allow_html=True
                )


def _render_tab_ai_explainability(doc_state: Dict[str, Any]):
    """
    Tab 3: Technical detail without visual clutter.
    Presents Net Sentiment Index formulation, class probabilities, targeted SHAP token attributions,
    full chunk inspector, and pipeline execution diagnostics.
    """
    agg = doc_state["aggregation"]
    summaries = doc_state["section_summaries"]
    shap_res = doc_state.get("shap_results") or {}
    pos_drivers = doc_state.get("shap_pos_drivers") or []
    neg_drivers = doc_state.get("shap_neg_drivers") or []
    timings = doc_state.get("timings") or {}
    
    overall = agg["overall_sentiment"]
    nsi = agg["net_sentiment_score"]
    conf = agg["overall_confidence"]
    nsi_color = "#55B89A" if nsi >= 0 else "#C86B70"
    SENT_COLORS = {"Positive": "#55B89A", "Negative": "#C86B70", "Neutral": "#9299A5"}
    
    st.markdown("### Model Explainability and Attribution")
    st.markdown(
        "Technical metrics and mathematical attribution detailing FinBERT classification mechanics, "
        "Net Sentiment Index (NSI), and SHAP token contributions."
    )
    
    # 1. NSI Formulation Card
    st.markdown("##### Net Sentiment Index (NSI) Formulation")
    st.markdown(
        f"<div class='dark-card'>"
        f"<div style='font-size:0.78rem;color:#A1A6B0;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:6px;'>Mathematical Formulation</div>"
        f"<div style='font-family:monospace;font-size:0.95rem;color:#F1F2F4;'>"
        f"NSI = P(Positive) − P(Negative) &nbsp;&nbsp;&rarr;&nbsp;&nbsp; "
        f"<span style='color:{nsi_color};font-weight:600;'>{nsi:+.4f}</span>"
        f"</div>"
        f"<div style='font-size:0.8rem;color:#A1A6B0;margin-top:6px;line-height:1.5;'>"
        f"NSI scales from <code>-1.000</code> (purely negative) to <code>+1.000</code> (purely positive), "
        f"with <code>0.000</code> representing directional equilibrium. "
        f"Overall model certainty across all passages is <strong>{conf*100:.1f}%</strong>."
        f"</div>"
        f"</div>",
        unsafe_allow_html=True
    )
    
    col_p1, col_p2 = st.columns([1, 1])
    with col_p1:
        st.caption("Document-Level Mean Class Probabilities")
        mean_df = pd.DataFrame([
            {"Sentiment": "Positive", "Probability": agg["mean_probabilities"]["positive"]},
            {"Sentiment": "Negative", "Probability": agg["mean_probabilities"]["negative"]},
            {"Sentiment": "Neutral",  "Probability": agg["mean_probabilities"]["neutral"]},
        ])
        fig_bar = px.bar(
            mean_df, x="Sentiment", y="Probability", color="Sentiment",
            color_discrete_map=SENT_COLORS, text_auto=".3f"
        )
        fig_bar.update_traces(
            textfont=dict(color="#F1F2F4", size=11),
            textposition="outside",
            cliponaxis=False,
        )
        fig_bar.update_layout(
            **_dark_layout(
                height=220,
                showlegend=False,
                margin=dict(t=20, b=10, l=36, r=16),
                yaxis=dict(range=[0, 1.15], gridcolor="#30343D",
                           tickfont=dict(color="#A1A6B0", size=11),
                           zerolinecolor="#30343D"),
                xaxis=dict(showgrid=False, tickfont=dict(color="#F1F2F4", size=11)),
            )
        )
        st.plotly_chart(fig_bar, width="stretch")
        
    with col_p2:
        st.caption("Section Probability Distribution")
        sec_prob_data = [
            {
                "Section": s["section_name"],
                "Pos Prob": f"{s['mean_positive']:.3f}",
                "Neg Prob": f"{s['mean_negative']:.3f}",
                "Neu Prob": f"{s['mean_neutral']:.3f}",
                "NSI": f"{s['net_sentiment_score']:+.3f}"
            }
            for s in summaries
        ]
        st.dataframe(pd.DataFrame(sec_prob_data), width="stretch", hide_index=True)
        
    st.markdown("<hr/>", unsafe_allow_html=True)
    
    # 2. SHAP Token-Level Attributions
    st.markdown("##### Targeted SHAP Token-Level Attributions")
    st.markdown(
        "SHAP uses Shapley additive values to allocate marginal prediction contributions to individual word tokens. "
        "Muted green tokens pushed FinBERT toward positive sentiment; muted red tokens pushed toward cautionary sentiment."
    )
    
    if shap_res:
        with st.expander("Targeted Token Explanations (Positive, Cautionary, Representative)", expanded=True):
            for tkey, target_title in [
                ("positive", "Positive Target Section"),
                ("negative", "Cautionary Target Section"),
                ("overall", "Representative Document Passage")
            ]:
                if tkey in shap_res:
                    tdata = shap_res[tkey]
                    sec_name = tdata.get("section_name", "Unknown")
                    expl = tdata.get("explanation", {})
                    pred_class = expl.get("predicted_class", "neutral")
                    pred_proba = expl.get("predicted_proba", 0.0)
                    
                    st.markdown(f"**{target_title}: `{sec_name}`**")
                    st.markdown(
                        f"FinBERT Classification: {format_sentiment_badge(pred_class)} with **`{pred_proba*100:.1f}%`** certainty "
                        f"(Computed in `{expl.get('elapsed_seconds', 'N/A')}s`)",
                        unsafe_allow_html=True
                    )
                    
                    pos_d, neg_d = SHAPExplainer.extract_top_drivers(expl, pred_class, top_k=5)
                    nl_exp = SHAPExplainer.generate_nl_explanation(pred_class, pos_d, neg_d)
                    st.caption(f"Attribution rationale: {nl_exp}")
                    
                    st.markdown(render_token_visualization_html(expl, pred_class), unsafe_allow_html=True)
                    st.markdown(
                        "<hr style='margin:10px 0;border:none;border-top:1px dashed #30343D;'/>",
                        unsafe_allow_html=True
                    )

            if pos_drivers or neg_drivers:
                st.markdown("**Top Aggregated Document Token Drivers:**")
                drv_c1, drv_c2 = st.columns(2)
                with drv_c1:
                    st.markdown("<small style='color:#55B89A;font-weight:600;'>POSITIVE DRIVERS</small>", unsafe_allow_html=True)
                    for d in pos_drivers[:5]:
                        st.markdown(f"- **`{d['token']}`**: `+{d['contribution']:.4f}`")
                with drv_c2:
                    st.markdown("<small style='color:#C86B70;font-weight:600;'>NEGATIVE DRIVERS</small>", unsafe_allow_html=True)
                    for d in neg_drivers[:5]:
                        st.markdown(f"- **`{d['token']}`**: `+{d['contribution']:.4f}`")
    else:
        st.info("SHAP explanations were not computed or were bypassed.")
        
    st.markdown("<hr/>", unsafe_allow_html=True)
    
    # 3. Chunk Inspector
    st.markdown("##### Document Chunk Inspector")
    st.caption("Audit every segmented passage, its FinBERT class scores, and text content:")
    chunk_inspector = [
        {
            "Chunk ID": c["chunk_id"],
            "Section": c.get("section_name", "N/A"),
            "Label": c["predicted_label"],
            "Certainty": f"{c['confidence_score']*100:.1f}%",
            "Pos": round(c["positive_score"], 3),
            "Neg": round(c["negative_score"], 3),
            "Neu": round(c["neutral_score"], 3),
            "Words": c["word_count"],
            "Snippet": c["text"][:110] + "..." if len(c["text"]) > 110 else c["text"]
        }
        for c in doc_state["annotated_chunks"]
    ]
    st.dataframe(pd.DataFrame(chunk_inspector), width="stretch", hide_index=True)
    
    # 4. Technical Diagnostics (Collapsed by default)
    st.markdown("##### Pipeline Diagnostics")
    with st.expander("Pipeline Diagnostics and Execution Timings", expanded=False):
        t_rows = [
            {"Stage": "1. Upload Validation", "Duration (s)": timings.get("1_upload_validation", "N/A")},
            {"Stage": "2. PyMuPDF Text Extraction", "Duration (s)": timings.get("2_text_extraction", "N/A")},
            {"Stage": "3. Section Detection", "Duration (s)": timings.get("3_section_detection", "N/A")},
            {"Stage": "4. Section-Aware Chunking", "Duration (s)": timings.get("4_text_chunking", "N/A")},
            {"Stage": "5a. Model Weights Load", "Duration (s)": timings.get("5a_model_loading", "N/A")},
            {"Stage": "5b. FinBERT Batch Inference", "Duration (s)": timings.get("5b_inference", "N/A")},
            {"Stage": "6. Sentiment Aggregation", "Duration (s)": timings.get("6_aggregation", "N/A")},
            {"Stage": "7. Targeted SHAP Attribution", "Duration (s)": timings.get("7_shap_explainability", "N/A")},
            {"Stage": "TOTAL PIPELINE DURATION", "Duration (s)": timings.get("total_pipeline", "N/A")},
        ]
        st.dataframe(pd.DataFrame(t_rows), width="stretch", hide_index=True)


def render_full_analysis_report(doc_state: Dict[str, Any]):
    """
    Renders structured analysis results organized into three primary tabs:
    1. Report Overview — plain-English summary, overall sentiment meaning, and high-level distribution.
    2. Key Findings — section-level evidence, verbatim excerpts, and priority classifications.
    3. AI Explainability — detailed probabilities, NSI formula, full probability chart, SHAP, and diagnostics.
    """
    tab_overview, tab_findings, tab_explain = st.tabs([
        "Report Overview",
        "Key Findings",
        "AI Explainability"
    ])
    
    with tab_overview:
        _render_tab_report_overview(doc_state)
        
    with tab_findings:
        _render_tab_key_findings(doc_state)
        
    with tab_explain:
        _render_tab_ai_explainability(doc_state)


# -----------------------------------------------------------------------------
# Assistant Message Dispatcher
# -----------------------------------------------------------------------------
def render_assistant_content(payload: Dict[str, Any]):
    """Renders structured assistant responses in the conversational interface."""
    intent = payload.get("intent")
    
    if intent == "full_analysis":
        render_full_analysis_report(payload["doc_state"])
        
    elif intent == "risk_analysis":
        risk_sections = payload["risk_sections"]
        neg_drivers = payload.get("neg_drivers", [])
        doc_state = payload.get("doc_state", {})

        st.markdown(f"### {payload['title']}")
        st.markdown(
            f"Sections are classified by mean negative sentence probability: "
            f"exceeding **`{PRIORITY_HIGH_RISK_THRESHOLD*100:.0f}%`** &rarr; **High Risk**; "
            f"between **`{PRIORITY_WATCH_THRESHOLD*100:.0f}%`** and **`{PRIORITY_HIGH_RISK_THRESHOLD*100:.0f}%`** &rarr; **Watch**."
        )

        for sec in risk_sections:
            p_badge = format_priority_badge(sec["priority"])
            s_badge = format_sentiment_badge(sec["sentiment"])
            st.markdown(
                f"<div class='dark-card' style='border-left:3px solid #C86B70;'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                f"<strong style='color:#F1F2F4;'>{sec['section_name']}</strong>"
                f"<div>{s_badge} {p_badge}</div></div>"
                f"<div style='margin-top:5px;font-size:0.83rem;color:#A1A6B0;'>"
                f"Cautionary Density: <strong style='color:#C86B70;'>{sec['mean_negative']*100:.1f}%</strong>"
                f" &nbsp;&middot;&nbsp; NSI: <strong>{sec['net_sentiment_score']:+.3f}</strong>"
                f" &nbsp;&middot;&nbsp; Passages: <strong>{sec['chunk_count']}</strong></div>"
                f"</div>",
                unsafe_allow_html=True
            )
            if doc_state and "annotated_chunks" in doc_state:
                r_chunks = [
                    c for c in doc_state["annotated_chunks"]
                    if c.get("section_name") == sec["section_name"] and c["predicted_label"] == "Negative"
                ]
                if r_chunks:
                    top_r = max(r_chunks, key=lambda c: c["negative_score"])
                    st.markdown(
                        f"<div class='snippet-box-dark snippet-box-risk'>"
                        f"<strong>Evidence Excerpt (Passage #{top_r['chunk_id']}):</strong><br/>"
                        f"{top_r['text']}"
                        f"</div>",
                        unsafe_allow_html=True
                    )

        if neg_drivers:
            st.markdown("##### Prominent Negative Drivers (SHAP):")
            for d in neg_drivers[:4]:
                st.markdown(f"- **`{d['token']}`**: `+{d['contribution']:.4f}`")

        st.markdown(
            "<div class='disclosure-banner'>"
            "<strong>Analytical Notice:</strong> High Risk / Watch classifications highlight concentrated "
            "cautionary language — not audited commercial distress."
            "</div>",
            unsafe_allow_html=True
        )

    elif intent == "summary_analysis":
        top_pos = payload["top_positive"]
        top_neg = payload["top_negative"]
        agg = payload["agg"]
        pos_drivers = payload.get("pos_drivers", [])
        neg_drivers = payload.get("neg_drivers", [])
        
        st.markdown(f"### {payload['title']}")
        st.markdown(
            f"The document exhibits an overall **{format_sentiment_badge(agg['overall_sentiment'])}** orientation with a "
            f"Net Sentiment Index of **`{agg['net_sentiment_score']:+.3f}`**.",
            unsafe_allow_html=True
        )
        
        c_pos, c_neg = st.columns(2)
        with c_pos:
            st.markdown("#### Key Constructive Sections")
            for sec in top_pos:
                st.markdown(
                    f"<div class='snippet-box-dark snippet-box-pos'>"
                    f"<strong>{sec['section_name']}</strong> — Pos: <code>{sec['mean_positive']*100:.1f}%</code> | NSI: <code>{sec['net_sentiment_score']:+.3f}</code>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            if pos_drivers:
                st.markdown("<small style='color:#A1A6B0;'>Positive Drivers: </small>" + ", ".join([f"`{d['token']}`" for d in pos_drivers[:4]]), unsafe_allow_html=True)
                
        with c_neg:
            st.markdown("#### Key Cautionary Sections")
            for sec in top_neg:
                st.markdown(
                    f"<div class='snippet-box-dark snippet-box-risk'>"
                    f"<strong>{sec['section_name']}</strong> ({sec['priority']}) — Neg: <code>{sec['mean_negative']*100:.1f}%</code> | NSI: <code>{sec['net_sentiment_score']:+.3f}</code>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            if neg_drivers:
                st.markdown("<small style='color:#A1A6B0;'>Risk Drivers: </small>" + ", ".join([f"`{d['token']}`" for d in neg_drivers[:4]]), unsafe_allow_html=True)

    elif intent == "explanation_analysis":
        shap_res = payload["shap_results"]
        matched_target = payload.get("matched_target")
        
        st.markdown(f"### {payload['title']}")
        st.markdown(
            "SHAP (SHapley Additive exPlanations) isolates the exact contribution of each word token to the model's prediction. "
            "Muted green tokens pushed FinBERT toward the predicted class; muted red tokens pulled against it."
        )
        
        target_keys = [matched_target] if (matched_target and matched_target in shap_res) else list(shap_res.keys())
        for tkey in target_keys:
            tdata = shap_res[tkey]
            expl = tdata.get("explanation", {})
            pred_class = expl.get("predicted_class", "neutral")
            pred_proba = expl.get("predicted_proba", 0.0)
            
            st.markdown(f"#### Target: `{tdata.get('section_name', tkey)}`")
            st.markdown(
                f"FinBERT Prediction: **{format_sentiment_badge(pred_class)}** with **`{pred_proba*100:.1f}%`** certainty.",
                unsafe_allow_html=True
            )
            
            pos_d, neg_d = SHAPExplainer.extract_top_drivers(expl, pred_class, top_k=5)
            nl_exp = SHAPExplainer.generate_nl_explanation(pred_class, pos_d, neg_d)
            st.caption(f"Attribution rationale: {nl_exp}")
            
            st.markdown(render_token_visualization_html(expl, pred_class), unsafe_allow_html=True)

    elif intent == "section_detail":
        sec = payload["section"]
        chunks = payload["chunks"]
        
        st.markdown(f"### {payload['title']}")
        st.markdown(
            f"Section **`{sec['section_name']}`** is categorized with sentiment **{format_sentiment_badge(sec['sentiment'])}** "
            f"and priority **{format_priority_badge(sec['priority'])}**."
            f"<br/><small style='color:#A1A6B0;'>Pos: <code>{sec['mean_positive']*100:.1f}%</code> | "
            f"Neg: <code>{sec['mean_negative']*100:.1f}%</code> | "
            f"Neu: <code>{sec['mean_neutral']*100:.1f}%</code> | NSI: <code>{sec['net_sentiment_score']:+.3f}</code></small>",
            unsafe_allow_html=True
        )
        
        st.markdown("##### Passages in this Section:")
        for c in chunks:
            box_cls = "snippet-box-pos" if c["predicted_label"] == "Positive" else (
                "snippet-box-risk" if c["predicted_label"] == "Negative" else "snippet-box-dark"
            )
            st.markdown(
                f"<div class='{box_cls}'>"
                f"<strong>Passage #{c['chunk_id']}</strong> ({format_sentiment_badge(c['predicted_label'])}, Certainty: <code>{c['confidence_score']*100:.1f}%</code>)<br/>"
                f"{c['text']}"
                f"</div>",
                unsafe_allow_html=True
            )


# -----------------------------------------------------------------------------
# Sidebar: Compact Metadata & Section Directory
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Financial Analyst")

    if st.session_state.doc_state is not None:
        doc = st.session_state.doc_state
        fname = doc["filename"]
        st.markdown(
            f"<div style='font-size:0.75rem;color:#A1A6B0;margin-bottom:2px;'>Active Document</div>"
            f"<code style='font-size:0.74rem;word-break:break-all;white-space:pre-wrap;'>{fname}</code>",
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"<small style='color:#A1A6B0;'>Pages</small><br/><strong>{doc['extraction_result']['total_pages']}</strong>", unsafe_allow_html=True)
            st.markdown(f"<small style='color:#A1A6B0;'>Sections</small><br/><strong>{len(doc['section_summaries'])}</strong>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<small style='color:#A1A6B0;'>Words</small><br/><strong>{doc['extraction_result']['total_word_count']:,}</strong>", unsafe_allow_html=True)
            st.markdown(f"<small style='color:#A1A6B0;'>Passages</small><br/><strong>{len(doc['annotated_chunks'])}</strong>", unsafe_allow_html=True)

        st.markdown("<hr/>", unsafe_allow_html=True)

        st.markdown("###### Section Navigator")
        sec_names = [s["section_name"] for s in doc["section_summaries"]]
        selected_nav = st.selectbox(
            "Jump to section:", options=["(Select)"] + sec_names,
            label_visibility="collapsed"
        )
        if selected_nav != "(Select)":
            prompt = f"Tell me about the {selected_nav} section"
            st.session_state.chat_history.append({"role": "user", "text": prompt})
            resp = generate_conversational_response(prompt, doc)
            st.session_state.chat_history.append({"role": "assistant", "payload": resp})
            st.rerun()

        with st.expander("Pipeline Configuration", expanded=False):
            chunk_size = st.slider("Words / Chunk", min_value=100, max_value=400, value=250, step=25)
            overlap_words = st.slider("Overlap Words", min_value=10, max_value=80, value=40, step=5)
            st.caption("Model: `ProsusAI/finbert` (PyTorch)")

        with st.expander("Diagnostic Timings", expanded=False):
            for stg, t_val in doc["timings"].items():
                st.markdown(f"- `{stg}`: {t_val}s")

        st.markdown("<hr/>", unsafe_allow_html=True)
        if st.button("Reset Workspace", width="stretch"):
            st.session_state.doc_state = None
            st.session_state.chat_history = []
            st.session_state.active_file_signature = None
            st.rerun()

    else:
        st.info("Upload a financial PDF report to begin analysis.")
        with st.expander("Pipeline Configuration", expanded=False):
            chunk_size = st.slider("Words / Chunk", min_value=100, max_value=400, value=250, step=25)
            overlap_words = st.slider("Overlap Words", min_value=10, max_value=80, value=40, step=5)
            st.caption("Model: `ProsusAI/finbert` (PyTorch)")


# -----------------------------------------------------------------------------
# Main Application Flow
# -----------------------------------------------------------------------------

# ── Welcome / Ingestion Screen ────────────────────────────────────────────────
if st.session_state.doc_state is None:
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-title">Financial Document Analyst</div>
        <div class="welcome-subtitle">
            Upload a financial report in PDF format to perform FinBERT sentiment analysis<br/>
            with targeted SHAP explainability and section risk prioritization.
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, welcome_col2, _ = st.columns([1, 4, 1])
    with welcome_col2:
        uploaded_file = st.file_uploader(
            "Select financial report in PDF format (10-K, 10-Q, Earnings, Audit):",
            type=["pdf"],
            help="Upload a digitally generated PDF financial report."
        )

else:
    # ── Active Document Header ──────────────────────────────────────────────
    st.markdown(
        f'<div class="app-header">'
        f'<div class="app-title">Financial Document Analyst</div>'
        f'<div class="app-subtitle">Active Report: <strong>{st.session_state.doc_state["filename"]}</strong></div>'
        f'</div>',
        unsafe_allow_html=True
    )
    
    # Optional file re-upload bar in main area
    with st.expander("Change or Upload Different Document", expanded=False):
        uploaded_file = st.file_uploader("Upload new PDF document:", type=["pdf"])


# Auto-process newly uploaded file
if uploaded_file is not None:
    current_sig = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.active_file_signature != current_sig:
        try:
            with st.spinner("Analyzing document with section detection, FinBERT, and SHAP..."):
                file_bytes = uploaded_file.getvalue()
                new_doc = run_document_analysis(
                    file_bytes, uploaded_file.name, chunk_size, overlap_words
                )
                st.session_state.doc_state = new_doc
                st.session_state.active_file_signature = current_sig
                initial_resp = generate_conversational_response("Analyze this report.", new_doc)
                st.session_state.chat_history = [{"role": "assistant", "payload": initial_resp}]
                st.rerun()
        except Exception as err:
            st.error(f"Analysis failed: {type(err).__name__}: {str(err)}")
            logger.exception("Pipeline execution failed")


# -----------------------------------------------------------------------------
# Conversation History — Render non-empty messages
# -----------------------------------------------------------------------------
for msg in st.session_state.chat_history:
    if msg["role"] == "user" and msg.get("text", "").strip():
        st.markdown(
            f'<div class="chat-msg-user">{msg["text"]}</div>',
            unsafe_allow_html=True
        )
    elif msg["role"] == "assistant" and msg.get("payload"):
        st.markdown('<div class="chat-msg-assistant">', unsafe_allow_html=True)
        render_assistant_content(msg["payload"])
        st.markdown('</div>', unsafe_allow_html=True)



