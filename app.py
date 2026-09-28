"""
Streamlit Web Application: Explainable Financial Sentiment Analysis Using FinBERT
Milestone 2 — Section-Aware Sentiment Analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import logging

# Configure logging to console so terminal shows all pipeline events
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("app")

from src.ingestion import PDFExtractor
from src.segmentation import TextChunker, SectionDetector
from src.models import FinBERTClassifier
from src.aggregation import SentimentAggregator
from src.explainability import SHAPExplainer
from config import SHAP_TOP_K_DRIVERS

# -----------------------------------------------------------------------------
# Page Configuration & Custom CSS Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Financial Sentiment Analysis | FinBERT",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium aesthetics
st.markdown("""
<style>
    /* Card Container styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(10px);
        margin-bottom: 15px;
    }
    
    .chunk-box {
        background-color: #1E293B;
        border-left: 4px solid #3B82F6;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 12px;
    }
    
    .chunk-box-positive {
        border-left-color: #10B981;
    }
    
    .chunk-box-negative {
        border-left-color: #EF4444;
    }

    /* SHAP Token Visualization Styles (Milestone 3) */
    .shap-token-container {
        background: #0F172A;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 18px 22px;
        line-height: 2.2;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 0.92rem;
        margin: 10px 0;
    }
    .shap-token {
        padding: 3px 5px;
        border-radius: 4px;
        margin: 1px;
        display: inline;
        transition: opacity 0.2s;
    }
    .shap-token:hover {
        opacity: 0.8;
    }
    .shap-section-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.85));
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
    }
    .shap-section-header {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .shap-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Model Loader — separated from pipeline to avoid false "stuck" perception
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_finbert_model():
    """
    Loads and caches the FinBERT classification model in memory.
    First call downloads ~440MB model weights from Hugging Face.
    """
    return FinBERTClassifier()


# -----------------------------------------------------------------------------
# App Header & Description
# -----------------------------------------------------------------------------
st.title("📊 Explainable Financial Sentiment Analysis")
st.markdown("---")

# -----------------------------------------------------------------------------
# Sidebar: Controls & Options
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="position: sticky; top: 0; padding-top: 20px;">
        <h3 style="margin-top: 0;">🧭 Quick Navigation</h3>
        <ul style="list-style-type: none; padding-left: 0; line-height: 2.0; font-size: 1.05rem;">
            <li><a href="#executive-summary-key-metrics" style="text-decoration: none; color: #38BDF8;">📄 Executive Summary</a></li>
            <li><a href="#sentiment-distribution-breakdown" style="text-decoration: none; color: #38BDF8;">📊 Sentiment Overview</a></li>
            <li><a href="#section-wise-sentiment-summary" style="text-decoration: none; color: #38BDF8;">📑 Section Analysis</a></li>
            <li><a href="#key-section-drivers" style="text-decoration: none; color: #38BDF8;">⚠️ Risk Analysis</a></li>
            <li><a href="#explainable-ai-insights" style="text-decoration: none; color: #38BDF8;">🧠 Explainable AI Insights</a></li>
            <li><a href="#overall-document-sentiment-drivers" style="text-decoration: none; color: #38BDF8;">📈 Overall Drivers</a></li>
            <li><a href="#full-document-chunk-explorer" style="text-decoration: none; color: #38BDF8;">⚙️ Processing Details</a></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Analysis Settings
# -----------------------------------------------------------------------------
with st.expander("⚙️ Analysis Settings", expanded=False):
    st.info("Using pretrained model: **ProsusAI/finbert**")
    chunk_size = st.slider("Target Words per Chunk", min_value=100, max_value=400, value=250, step=25)
    overlap_words = st.slider("Overlap Words", min_value=10, max_value=80, value=40, step=5)

st.markdown("---")

# -----------------------------------------------------------------------------
# PDF File Upload Component
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Financial PDF Document (e.g., 10-K, 10-Q, Audit Report, Earnings Release)",
    type=["pdf"],
    help="Select a digitally generated financial PDF report to analyze."
)

if uploaded_file is not None:
    # Total pipeline timer
    pipeline_start = time.time()
    
    # Create persistent containers for each stage so they update independently
    stage_header = st.empty()
    progress_bar = st.progress(0, text="Initializing pipeline...")
    debug_panel = st.expander("🔬 Debug Log (Stage Timings & Diagnostics)", expanded=True)
    
    stage_timings = {}  # Collects timing for all stages
    
    # =========================================================================
    # STAGE 1: PDF Upload Validation
    # =========================================================================
    stage_header.info("⏳ **Stage 1/5:** Validating uploaded PDF...")
    progress_bar.progress(5, text="Stage 1/5: Validating PDF upload...")
    logger.info("=== STAGE 1: PDF Upload Validation ===")
    
    s1_start = time.time()
    pdf_bytes = uploaded_file.getvalue()
    file_size_kb = len(pdf_bytes) / 1024
    file_size_mb = file_size_kb / 1024
    
    with debug_panel:
        st.write(f"**File Name:** `{uploaded_file.name}`")
        st.write(f"**File Size:** `{file_size_kb:,.1f} KB` ({file_size_mb:.2f} MB)")
    
    # Guard: empty or suspiciously small files
    if len(pdf_bytes) < 100:
        st.error(f"PDF file is too small ({len(pdf_bytes)} bytes). It is likely empty or corrupted.")
        logger.error(f"Aborting: PDF too small ({len(pdf_bytes)} bytes)")
        st.stop()
    
    stage_timings["1_upload_validation"] = round(time.time() - s1_start, 3)
    logger.info(f"Stage 1 complete: {stage_timings['1_upload_validation']}s")

    # =========================================================================
    # STAGE 2: Text Extraction via PyMuPDF
    # =========================================================================
    stage_header.info("⏳ **Stage 2/7:** Extracting text from PDF via PyMuPDF...")
    progress_bar.progress(10, text="Stage 2/7: Extracting text from PDF...")
    logger.info("=== STAGE 2: Text Extraction ===")
    
    s2_start = time.time()
    try:
        extraction_result = PDFExtractor.extract_text_from_bytes(pdf_bytes, timeout_seconds=30.0)
    except TimeoutError as e:
        st.error(f"⏱️ Extraction Timeout: {str(e)}")
        logger.error(f"Extraction timeout: {e}")
        st.stop()
    except ValueError as e:
        st.error(f"📄 Invalid PDF: {str(e)}")
        logger.error(f"Invalid PDF: {e}")
        st.stop()
    except RuntimeError as e:
        st.error(f"💥 PDF Parsing Error: {str(e)}")
        logger.error(f"PDF parsing error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Unexpected Extraction Error: {type(e).__name__}: {str(e)}")
        logger.exception("Unexpected extraction error")
        st.stop()
    
    stage_timings["2_text_extraction"] = round(time.time() - s2_start, 3)
    
    full_text = extraction_result["full_text"]
    
    with debug_panel:
        st.write(f"**Total Pages:** `{extraction_result['total_pages']}`")
        st.write(f"**Pages with Text:** `{extraction_result['extracted_pages_count']}`")
        st.write(f"**Total Characters:** `{extraction_result['total_char_count']:,}`")
        st.write(f"**Total Words:** `{extraction_result['total_word_count']:,}`")
        st.write(f"**Extraction Time:** `{stage_timings['2_text_extraction']}s`")
    
    # Guard: no extractable text
    if not full_text.strip():
        st.error(
            "No extractable text found in the PDF. "
            "This may be a scanned/image-only document (OCR not enabled in Milestone 1)."
        )
        logger.error("Aborting: No text extracted from PDF")
        st.stop()
    
    # Warn if very little text was extracted
    if extraction_result["total_word_count"] < 20:
        st.warning(
            f"Only {extraction_result['total_word_count']} words extracted. "
            f"The PDF may be mostly images/tables. Results may be unreliable."
        )
    
    logger.info(
        f"Stage 2 complete: {extraction_result['extracted_pages_count']}/{extraction_result['total_pages']} pages, "
        f"{extraction_result['total_word_count']:,} words in {stage_timings['2_text_extraction']}s"
    )
    
    # =========================================================================
    # STAGE 3: Section Detection & Section-Aware Chunking
    # =========================================================================
    stage_header.info("⏳ **Stage 3/7:** Detecting financial report sections...")
    progress_bar.progress(20, text="Stage 3/7: Detecting sections...")
    logger.info("=== STAGE 3: Section Detection ===")
    
    s3_start = time.time()
    detector = SectionDetector()
    sections = detector.detect_sections(full_text)
    stage_timings["3_section_detection"] = round(time.time() - s3_start, 3)
    
    with debug_panel:
        st.write(f"**Sections Detected:** `{len(sections)}`")
        for sec in sections:
            st.write(f"  - `{sec['section_name']}` ({sec['word_count']:,} words)")
        st.write(f"**Detection Time:** `{stage_timings['3_section_detection']}s`")
    
    logger.info(f"Stage 3 complete: {len(sections)} sections in {stage_timings['3_section_detection']}s")
    
    # =========================================================================
    # STAGE 4: Section-Aware Text Chunking
    # =========================================================================
    stage_header.info("⏳ **Stage 4/7:** Chunking sections into sentence-aware blocks...")
    progress_bar.progress(30, text="Stage 4/7: Chunking text by section...")
    logger.info("=== STAGE 4: Section-Aware Chunking ===")
    
    s4_start = time.time()
    chunker = TextChunker(target_words_per_chunk=chunk_size, overlap_words=overlap_words)
    chunks = chunker.chunk_sections(sections)
    stage_timings["4_text_chunking"] = round(time.time() - s4_start, 3)
    
    with debug_panel:
        st.write(f"**Chunks Generated:** `{len(chunks)}`")
        st.write(f"**Config:** target=`{chunk_size}` words/chunk, overlap=`{overlap_words}` words")
        if chunks:
            word_counts = [c["word_count"] for c in chunks]
            st.write(f"**Chunk Size Range:** `{min(word_counts)}` to `{max(word_counts)}` words (avg: `{sum(word_counts)//len(word_counts)}`)")
            # Show chunks per section
            from collections import Counter
            section_chunk_counts = Counter(c["section_name"] for c in chunks)
            for sec_name, count in section_chunk_counts.items():
                st.write(f"  - `{sec_name}`: {count} chunks")
        st.write(f"**Chunking Time:** `{stage_timings['4_text_chunking']}s`")
    
    if not chunks:
        st.error("Text chunking produced 0 chunks. The extracted text may be too short or malformed.")
        logger.error("Aborting: 0 chunks produced")
        st.stop()
    
    logger.info(f"Stage 4 complete: {len(chunks)} chunks in {stage_timings['4_text_chunking']}s")
    
    # =========================================================================
    # STAGE 5: FinBERT Model Loading & Inference
    # =========================================================================
    stage_header.info("⏳ **Stage 5/7:** Loading FinBERT model & running sentiment inference...")
    progress_bar.progress(35, text="Stage 5/7: Loading FinBERT model (first run downloads ~440MB)...")
    logger.info("=== STAGE 5: FinBERT Loading & Inference ===")
    
    # 5a: Model loading (cached after first run)
    s5a_start = time.time()
    try:
        classifier = load_finbert_model()
    except Exception as e:
        st.error(f"❌ FinBERT Model Loading Failed: {type(e).__name__}: {str(e)}")
        logger.exception("FinBERT model loading failed")
        st.stop()
    
    stage_timings["5a_model_loading"] = round(time.time() - s5a_start, 3)
    
    with debug_panel:
        st.write(f"**Model:** `{classifier.model_name}`")
        st.write(f"**Device:** `{classifier.device}`")
        st.write(f"**Label Map:** `{classifier.id2label}`")
        st.write(f"**Model Load Time:** `{stage_timings['5a_model_loading']}s` {'(cached)' if stage_timings['5a_model_loading'] < 0.5 else '(fresh load)'}")
    
    logger.info(f"Model loaded in {stage_timings['5a_model_loading']}s")
    
    # 5b: Batch inference with progress updates
    progress_bar.progress(40, text=f"Stage 5/7: Running FinBERT inference on {len(chunks)} chunks...")
    
    s4b_start = time.time()
    inference_progress = st.empty()
    
    def on_inference_progress(processed: int, total: int):
        """Callback to update Streamlit progress bar during inference."""
        pct = int(40 + (processed / total) * 20)  # Map to 40%-60% of progress bar
        progress_bar.progress(pct, text=f"Stage 5/7: FinBERT inference — {processed}/{total} chunks...")
        inference_progress.caption(f"Processed {processed}/{total} chunks...")
    
    try:
        annotated_chunks = classifier.predict_chunks(chunks, progress_callback=on_inference_progress)
    except Exception as e:
        st.error(f"❌ FinBERT Inference Error: {type(e).__name__}: {str(e)}")
        logger.exception("FinBERT inference failed")
        st.stop()
    
    inference_progress.empty()  # Clean up progress text
    stage_timings["5b_inference"] = round(time.time() - s4b_start, 3)
    
    with debug_panel:
        st.write(f"**Chunks Classified:** `{len(annotated_chunks)}`")
        st.write(f"**Inference Time:** `{stage_timings['5b_inference']}s`")
        if annotated_chunks:
            avg_conf = sum(c["confidence_score"] for c in annotated_chunks) / len(annotated_chunks)
            st.write(f"**Avg Confidence:** `{avg_conf*100:.1f}%`")
    
    logger.info(f"Inference complete: {len(annotated_chunks)} chunks in {stage_timings['5b_inference']}s")
    
    # =========================================================================
    # STAGE 6: Document & Section Aggregation
    # =========================================================================
    stage_header.info("⏳ **Stage 6/7:** Aggregating document & section-level metrics...")
    progress_bar.progress(62, text="Stage 6/7: Aggregating results...")
    logger.info("=== STAGE 6: Aggregation ===")
    
    s6_start = time.time()
    aggregation = SentimentAggregator.aggregate_document_sentiment(annotated_chunks)
    section_summaries = SentimentAggregator.aggregate_by_section(annotated_chunks)
    stage_timings["6_aggregation"] = round(time.time() - s6_start, 3)
    
    logger.info(f"Stage 6 complete in {stage_timings['6_aggregation']}s")
    
    with debug_panel:
        st.write(f"**Aggregation Time:** `{stage_timings['6_aggregation']}s`")
        st.write(f"**Section Summaries:** `{len(section_summaries)}`")
        high_risk_count = sum(1 for s in section_summaries if s['priority'] == 'High Risk')
        watch_count = sum(1 for s in section_summaries if s['priority'] == 'Watch')
        st.write(f"**Priority Breakdown:** `{high_risk_count}` High Risk, `{watch_count}` Watch")

    # =========================================================================
    # STAGE 7: SHAP Explainability (Milestone 3)
    # =========================================================================
    stage_header.info("⏳ **Stage 7/7:** Generating SHAP explanations for key sections...")
    progress_bar.progress(70, text="Stage 7/7: Running targeted SHAP explainability...")
    logger.info("=== STAGE 7: SHAP Explainability ===")

    s7_start = time.time()
    shap_results = None
    shap_overall_pos_drivers = []
    shap_overall_neg_drivers = []

    try:
        # 7a: Initialize SHAP explainer (reuses cached model weights)
        shap_explainer = SHAPExplainer()

        # 7b: Select target sections (most positive, most negative, overall)
        shap_targets = SHAPExplainer.select_target_sections(
            section_summaries, annotated_chunks, aggregation["overall_sentiment"]
        )

        progress_bar.progress(78, text="Stage 7/7: Computing SHAP token attributions...")

        # 7c: Generate SHAP explanations for the 3 targets
        if shap_targets:
            shap_results = shap_explainer.explain_targets(shap_targets)

            # 7d: Extract overall document drivers
            shap_overall_pos_drivers, shap_overall_neg_drivers = (
                SHAPExplainer.get_overall_document_drivers(
                    shap_results, aggregation["overall_sentiment"], top_k=SHAP_TOP_K_DRIVERS
                )
            )

        stage_timings["7_shap_explainability"] = round(time.time() - s7_start, 3)
        logger.info(f"Stage 7 complete in {stage_timings['7_shap_explainability']}s")

        with debug_panel:
            st.write(f"**SHAP Targets Explained:** `{len(shap_results) if shap_results else 0}`")
            st.write(f"**SHAP Time:** `{stage_timings['7_shap_explainability']}s`")
            if shap_results:
                for tkey, tdata in shap_results.items():
                    elapsed = tdata.get('explanation', {}).get('elapsed_seconds', 'N/A')
                    st.write(f"  - `{tkey}`: {tdata['section_name']} ({elapsed}s)")

    except Exception as e:
        stage_timings["7_shap_explainability"] = round(time.time() - s7_start, 3)
        logger.exception("SHAP explainability failed")
        with debug_panel:
            st.warning(f"⚠️ SHAP explainability encountered an error: {type(e).__name__}: {str(e)}")
            st.write(f"**SHAP Time (failed):** `{stage_timings['7_shap_explainability']}s`")

    progress_bar.progress(95, text="Finalizing results...")

    total_pipeline_time = round(time.time() - pipeline_start, 2)
    stage_timings["total_pipeline"] = total_pipeline_time

    logger.info(f"=== PIPELINE COMPLETE: {total_pipeline_time}s total ===")

    # Final debug timing summary
    with debug_panel:
        st.markdown("---")
        st.write("**Pipeline Timing Summary:**")
        timing_rows = [
            {"Stage": "1. Upload Validation", "Time (s)": stage_timings["1_upload_validation"]},
            {"Stage": "2. Text Extraction", "Time (s)": stage_timings["2_text_extraction"]},
            {"Stage": "3. Section Detection", "Time (s)": stage_timings["3_section_detection"]},
            {"Stage": "4. Section-Aware Chunking", "Time (s)": stage_timings["4_text_chunking"]},
            {"Stage": "5a. Model Loading", "Time (s)": stage_timings["5a_model_loading"]},
            {"Stage": "5b. FinBERT Inference", "Time (s)": stage_timings["5b_inference"]},
            {"Stage": "6. Aggregation", "Time (s)": stage_timings["6_aggregation"]},
        ]
        if "7_shap_explainability" in stage_timings:
            timing_rows.append({"Stage": "7. SHAP Explainability", "Time (s)": stage_timings["7_shap_explainability"]})
        timing_rows.append({"Stage": "TOTAL", "Time (s)": total_pipeline_time})
        timing_df = pd.DataFrame(timing_rows)
        st.dataframe(timing_df, use_container_width=True, hide_index=True)
    
    # Update header and progress to success
    shap_stage_msg = ""
    if shap_results:
        shap_stage_msg = f", SHAP: {stage_timings.get('7_shap_explainability', 'N/A')}s"
    stage_header.success(
        f"✅ **Analysis Complete!** Processed {extraction_result['total_pages']} pages, "
        f"{len(sections)} sections, {len(chunks)} chunks in {total_pipeline_time}s{shap_stage_msg}"
    )
    progress_bar.progress(100, text="Pipeline complete!")

    # =========================================================================
    # DASHBOARD RESULTS VISUALIZATION
    # =========================================================================
    tab_exec, tab_section, tab_explain, tab_tech = st.tabs(["📄 Executive Summary", "📑 Section Analysis", "🧠 Explainable AI", "⚙️ Technical Details"])

    with tab_exec:
        st.subheader("📋 Executive Summary & Key Metrics", anchor="executive-summary-key-metrics")
    
        m1, m2, m3, m4 = st.columns(4)
    
        sentiment_color = "#10B981" if aggregation["overall_sentiment"] == "Positive" else (
            "#EF4444" if aggregation["overall_sentiment"] == "Negative" else "#94A3B8"
        )
    
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <span style="color: #94A3B8; font-size: 0.85rem;">Overall Document Sentiment</span>
                <h2 style="color: {sentiment_color}; margin: 5px 0;">{aggregation['overall_sentiment']}</h2>
                <span style="color: #64748B; font-size: 0.8rem;">Dominant Score</span>
            </div>
            """, unsafe_allow_html=True)
        
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <span style="color: #94A3B8; font-size: 0.85rem;">Average Confidence</span>
                <h2 style="color: #38BDF8; margin: 5px 0;">{aggregation['overall_confidence'] * 100:.1f}%</h2>
                <span style="color: #64748B; font-size: 0.8rem;">Model Probability</span>
            </div>
            """, unsafe_allow_html=True)
        
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <span style="color: #94A3B8; font-size: 0.85rem;">Total Chunks Analyzed</span>
                <h2 style="color: #F59E0B; margin: 5px 0;">{aggregation['total_chunks']}</h2>
                <span style="color: #64748B; font-size: 0.8rem;">{extraction_result['total_word_count']:,} total words</span>
            </div>
            """, unsafe_allow_html=True)
        
        with m4:
            nsi = aggregation['net_sentiment_score']
            nsi_color = "#10B981" if nsi >= 0 else "#EF4444"
            st.markdown(f"""
            <div class="metric-card">
                <span style="color: #94A3B8; font-size: 0.85rem;">Net Sentiment Index (NSI)</span>
                <h2 style="color: {nsi_color}; margin: 5px 0;">{nsi:+.3f}</h2>
                <span style="color: #64748B; font-size: 0.8rem;">(Positive - Negative)</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # -------------------------------------------------------------------------
        # Sentiment Distribution Charts
        # -------------------------------------------------------------------------
        c1, c2 = st.columns([1, 1])
    
        with c1:
            st.subheader("📊 Sentiment Distribution Breakdown", anchor="sentiment-distribution-breakdown")
        
            dist_df = pd.DataFrame([
                {"Sentiment": k, "Chunk Count": v, "Percentage": aggregation["distribution_percentage"][k]}
                for k, v in aggregation["sentiment_distribution"].items()
            ])
        
            fig = px.pie(
                dist_df,
                names="Sentiment",
                values="Chunk Count",
                color="Sentiment",
                color_discrete_map={"Positive": "#10B981", "Negative": "#EF4444", "Neutral": "#64748B"},
                hole=0.4,
                hover_data=["Percentage"]
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#F8FAFC")
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            st.subheader("📈 Mean Confidence Probabilities")
        
            probs_df = pd.DataFrame([
                {"Sentiment": "Positive", "Probability": aggregation["mean_probabilities"]["positive"]},
                {"Sentiment": "Negative", "Probability": aggregation["mean_probabilities"]["negative"]},
                {"Sentiment": "Neutral", "Probability": aggregation["mean_probabilities"]["neutral"]}
            ])
        
            fig_bar = px.bar(
                probs_df,
                x="Sentiment",
                y="Probability",
                color="Sentiment",
                color_discrete_map={"Positive": "#10B981", "Negative": "#EF4444", "Neutral": "#64748B"},
                text_auto='.3f'
            )
            fig_bar.update_layout(
                yaxis=dict(range=[0, 1]),
                margin=dict(t=20, b=20, l=20, r=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#F8FAFC"),
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")

        # -------------------------------------------------------------------------
        # Section-Wise Summary Table (Milestone 2)
        # -------------------------------------------------------------------------
    with tab_section:
        st.subheader("📋 Section-Wise Sentiment Summary", anchor="section-wise-sentiment-summary")
    
        if section_summaries:
            priority_colors = {
                "High Risk": "#EF4444",
                "Watch": "#F59E0B",
                "Normal": "#10B981"
            }
            sentiment_colors = {
                "Positive": "#10B981",
                "Negative": "#EF4444",
                "Neutral": "#94A3B8"
            }
        
            section_table_data = []
            for sec in section_summaries:
                section_table_data.append({
                    "Section": sec["section_name"],
                    "Sentiment": sec["sentiment"],
                    "Confidence": f"{sec['confidence']*100:.1f}%",
                    "Net Score": f"{sec['net_sentiment_score']:+.3f}",
                    "Positive": f"{sec['mean_positive']:.3f}",
                    "Negative": f"{sec['mean_negative']:.3f}",
                    "Neutral": f"{sec['mean_neutral']:.3f}",
                    "Chunks": sec["chunk_count"],
                    "Priority": sec["priority"]
                })
        
            section_df = pd.DataFrame(section_table_data)
            st.dataframe(section_df, use_container_width=True, hide_index=True)
        else:
            st.info("No section-level data available.")
    
        st.markdown("---")

        # -------------------------------------------------------------------------
        # Top Positive Sections & High-Risk Sections (Milestone 2)
        # -------------------------------------------------------------------------
        st.subheader("🔍 Key Section Drivers", anchor="key-section-drivers")
    
        col_pos, col_neg = st.columns(2)
    
        # Top Positive Sections — sorted by mean_positive descending
        positive_sections = sorted(section_summaries, key=lambda s: s["mean_positive"], reverse=True)
    
        # High-Risk Sections — already sorted by priority in aggregate_by_section
        risk_sections = [s for s in section_summaries if s["priority"] in ("High Risk", "Watch")]
    
        with col_pos:
            st.markdown("### 🟢 Top Positive Sections")
            if positive_sections:
                for idx, sec in enumerate(positive_sections[:5], 1):
                    with st.expander(f"#{idx} | {sec['section_name']} (Pos: {sec['mean_positive']*100:.1f}%)"):
                        st.markdown(f"**Sentiment:** `{sec['sentiment']}` | **Confidence:** `{sec['confidence']*100:.1f}%`")
                        st.markdown(f"**Net Score:** `{sec['net_sentiment_score']:+.3f}` | **Chunks:** `{sec['chunk_count']}`")
                        st.markdown(f"**Probabilities:** `Pos: {sec['mean_positive']:.3f}` | `Neg: {sec['mean_negative']:.3f}` | `Neu: {sec['mean_neutral']:.3f}`")
            else:
                st.info("No positive sections detected.")
    
        with col_neg:
            st.markdown("### 🔴 High-Risk Sections")
            if risk_sections:
                for idx, sec in enumerate(risk_sections[:5], 1):
                    priority_badge_color = priority_colors.get(sec['priority'], '#94A3B8')
                    with st.expander(f"#{idx} | {sec['section_name']} ({sec['priority']} — Neg: {sec['mean_negative']*100:.1f}%)"):
                        st.markdown(
                            f"<span style='background-color: {priority_badge_color}22; color: {priority_badge_color}; "
                            f"border: 1px solid {priority_badge_color}; padding: 3px 8px; border-radius: 4px; "
                            f"font-weight: 600; font-size: 0.85rem;'>{sec['priority']}</span>",
                            unsafe_allow_html=True
                        )
                        st.markdown(f"**Sentiment:** `{sec['sentiment']}` | **Confidence:** `{sec['confidence']*100:.1f}%`")
                        st.markdown(f"**Net Score:** `{sec['net_sentiment_score']:+.3f}` | **Chunks:** `{sec['chunk_count']}`")
                        st.markdown(f"**Probabilities:** `Pos: {sec['mean_positive']:.3f}` | `Neg: {sec['mean_negative']:.3f}` | `Neu: {sec['mean_neutral']:.3f}`")
            else:
                st.success("No high-risk sections detected. All sections are within normal range.")

        st.markdown("---")

        # -------------------------------------------------------------------------
        # Detailed Chunk Explorer Table (preserved from Milestone 1, with section_name)
        # -------------------------------------------------------------------------
        # =========================================================================
        # SHAP EXPLAINABLE AI DASHBOARD SECTION (Milestone 3)
        # =========================================================================
    with tab_explain:
        st.subheader("🔬 Explainable AI Insights", anchor="explainable-ai-insights")
        st.caption("Targeted SHAP explanations for the most positive, most negative, and overall representative sections")

        if shap_results:

            def render_shap_tokens(explanation: dict, target_class: str):
                """
                Renders SHAP token-level visualization as colored HTML spans.
                Green = pushes toward target class, Red = pushes away, Gray = neutral.
                """
                tokens = explanation.get("tokens", [])
                values = explanation.get("shap_values", {}).get(target_class, [])
                if not tokens or not values:
                    st.info("No token-level data available.")
                    return

                # Normalize SHAP values for coloring intensity
                abs_max = max(abs(v) for v in values) if values else 1.0
                if abs_max == 0:
                    abs_max = 1.0

                html_parts = []
                for tok, val in zip(tokens, values):
                    normalized = val / abs_max  # Range: -1 to 1
                    if normalized > 0.05:
                        # Positive contribution → green
                        intensity = min(normalized, 1.0)
                        bg = f"rgba(16, 185, 129, {intensity * 0.6:.2f})"
                        color = "#D1FAE5" if intensity > 0.3 else "#A7F3D0"
                    elif normalized < -0.05:
                        # Negative contribution → red
                        intensity = min(abs(normalized), 1.0)
                        bg = f"rgba(239, 68, 68, {intensity * 0.6:.2f})"
                        color = "#FEE2E2" if intensity > 0.3 else "#FECACA"
                    else:
                        # Neutral → subtle gray
                        bg = "rgba(148, 163, 184, 0.1)"
                        color = "#94A3B8"

                    tooltip = f"SHAP: {val:+.4f}"
                    html_parts.append(
                        f'<span class="shap-token" style="background: {bg}; color: {color};" '
                        f'title="{tooltip}">{tok}</span>'
                    )

                html = f'<div class="shap-token-container">{" ".join(html_parts)}</div>'
                st.markdown(html, unsafe_allow_html=True)

            def render_section_explanation(target_key: str, result_data: dict, icon: str, accent_color: str):
                """
                Renders a complete section explanation card with metadata + SHAP tokens.
                Uses the ACTUAL predicted class from FinBERT (not the expected class).
                """
                section_name = result_data.get("section_name", "Unknown")
                sentiment = result_data.get("sentiment", "N/A")
                explanation = result_data.get("explanation", {})
                predicted_class = explanation.get("predicted_class", "neutral")
                predicted_proba = explanation.get("predicted_proba", 0.0)
                all_probs = explanation.get("all_probabilities", {})
                elapsed = explanation.get("elapsed_seconds", "N/A")

                # Always explain the ACTUAL predicted class — not what we expected
                target_class = predicted_class
                conf_val = predicted_proba

                # Check for mismatch between expected and actual
                expected_class = target_key if target_key in ("positive", "negative") else None
                has_mismatch = expected_class and expected_class != predicted_class

                # Section header card
                pred_color = {"positive": "#10B981", "negative": "#EF4444", "neutral": "#94A3B8"}.get(predicted_class, "#94A3B8")
                st.markdown(f"""
                <div class="shap-section-card">
                    <div class="shap-section-header" style="color: {accent_color};">
                        {icon} {section_name}
                    </div>
                    <span class="shap-badge" style="background: {pred_color}22; color: {pred_color}; border: 1px solid {pred_color};">
                        FinBERT: {predicted_class.capitalize()}
                    </span>
                    <span class="shap-badge" style="background: rgba(56,189,248,0.15); color: #38BDF8; border: 1px solid rgba(56,189,248,0.3);">
                        Confidence: {conf_val*100:.1f}%
                    </span>
                    <span style="color: #64748B; font-size: 0.8rem; margin-left: 8px;">
                        SHAP computed in {elapsed}s
                    </span>
                </div>
                """, unsafe_allow_html=True)

                # Show probability breakdown
                if all_probs:
                    prob_parts = " | ".join(
                        f"**{lbl.capitalize()}:** `{prob*100:.1f}%`"
                        for lbl, prob in sorted(all_probs.items(), key=lambda x: -x[1])
                    )
                    st.markdown(f"📊 {prob_parts}")

                # Mismatch warning
                if has_mismatch:
                    st.warning(
                        f"⚠️ **Class Mismatch:** This is the '{target_key}' target section, "
                        f"but FinBERT predicted **{predicted_class.capitalize()}** "
                        f"(not {expected_class.capitalize()}). "
                        f"SHAP explains the actual prediction below."
                    )

                # Render SHAP token visualization for the ACTUAL predicted class
                render_shap_tokens(explanation, target_class)

                # Show section-specific top drivers for the predicted class
                pos_drivers, neg_drivers = SHAPExplainer.extract_top_drivers(
                    explanation, target_class, top_k=5
                )
                
                # Show natural language explanation
                nl_explanation = SHAPExplainer.generate_nl_explanation(predicted_class, pos_drivers, neg_drivers)
                st.markdown(f"""
                <div style="background-color: rgba(56, 189, 248, 0.05); border-left: 4px solid #38BDF8; padding: 12px 16px; margin-bottom: 16px; border-radius: 4px;">
                    <span style="color: #E2E8F0; font-size: 0.95rem;"><em>"{nl_explanation}"</em></span>
                </div>
                """, unsafe_allow_html=True)
                
                d1, d2 = st.columns(2)
                with d1:
                    if pos_drivers:
                        st.markdown(f"**Top Drivers → {predicted_class.capitalize()}:**")
                        for d in pos_drivers:
                            bar_width = min(abs(d['contribution']) / (abs(pos_drivers[0]['contribution']) or 1) * 100, 100)
                            st.markdown(
                                f"<div style='display:flex;align-items:center;margin:2px 0;'>"
                                f"<span style='color:#10B981;width:100px;font-family:monospace;'>+ {d['token']}</span>"
                                f"<div style='background:rgba(16,185,129,0.3);height:14px;width:{bar_width}%;border-radius:3px;margin-left:8px;'></div>"
                                f"<span style='color:#64748B;font-size:0.75rem;margin-left:6px;'>{d['contribution']:+.4f}</span>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                with d2:
                    if neg_drivers:
                        st.markdown(f"**Top Opposing → {predicted_class.capitalize()}:**")
                        for d in neg_drivers:
                            bar_width = min(abs(d['contribution']) / (abs(neg_drivers[0]['contribution']) or 1) * 100, 100)
                            st.markdown(
                                f"<div style='display:flex;align-items:center;margin:2px 0;'>"
                                f"<span style='color:#EF4444;width:100px;font-family:monospace;'>- {d['token']}</span>"
                                f"<div style='background:rgba(239,68,68,0.3);height:14px;width:{bar_width}%;border-radius:3px;margin-left:8px;'></div>"
                                f"<span style='color:#64748B;font-size:0.75rem;margin-left:6px;'>{d['contribution']:+.4f}</span>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

            # --- Positive Section Explanation ---
            if "positive" in shap_results:
                with st.expander("🟢 Positive Section Explanation", expanded=True):
                    render_section_explanation("positive", shap_results["positive"], "🟢", "#10B981")

            # --- Negative Section Explanation ---
            if "negative" in shap_results:
                with st.expander("🔴 Negative / High-Risk Section Explanation", expanded=True):
                    render_section_explanation("negative", shap_results["negative"], "🔴", "#EF4444")

            # --- Overall Document Drivers ---
            with st.expander("📊 Overall Document Sentiment Drivers", expanded=True):
                st.markdown("""
                <div class="shap-section-card">
                    <div class="shap-section-header" style="color: #38BDF8;">
                        📊 Aggregated Word-Level Sentiment Drivers
                    </div>
                    <span style="color: #94A3B8; font-size: 0.85rem;">
                        Combined SHAP contributions across all explained sections
                    </span>
                </div>
                """, unsafe_allow_html=True)

                drv_col1, drv_col2 = st.columns(2)

                with drv_col1:
                    st.markdown("#### 🟢 Top Positive Drivers")
                    if shap_overall_pos_drivers:
                        driver_tokens = [d["token"] for d in shap_overall_pos_drivers]
                        driver_values = [d["contribution"] for d in shap_overall_pos_drivers]

                        fig_pos = go.Figure(go.Bar(
                            x=driver_values,
                            y=driver_tokens,
                            orientation="h",
                            marker_color="#10B981",
                            text=[f"+{v:.4f}" for v in driver_values],
                            textposition="outside",
                        ))
                        fig_pos.update_layout(
                            yaxis=dict(autorange="reversed"),
                            margin=dict(t=10, b=10, l=10, r=60),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#F8FAFC", size=12),
                            height=max(250, len(driver_tokens) * 32),
                            xaxis=dict(showgrid=False, zeroline=False),
                            yaxis_showgrid=False,
                        )
                        st.plotly_chart(fig_pos, use_container_width=True)
                    else:
                        st.info("No significant positive drivers detected.")

                with drv_col2:
                    st.markdown("#### 🔴 Top Negative Drivers")
                    if shap_overall_neg_drivers:
                        driver_tokens = [d["token"] for d in shap_overall_neg_drivers]
                        driver_values = [d["contribution"] for d in shap_overall_neg_drivers]

                        fig_neg = go.Figure(go.Bar(
                            x=driver_values,
                            y=driver_tokens,
                            orientation="h",
                            marker_color="#EF4444",
                            text=[f"+{v:.4f}" for v in driver_values],
                            textposition="outside",
                        ))
                        fig_neg.update_layout(
                            yaxis=dict(autorange="reversed"),
                            margin=dict(t=10, b=10, l=10, r=60),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#F8FAFC", size=12),
                            height=max(250, len(driver_tokens) * 32),
                            xaxis=dict(showgrid=False, zeroline=False),
                            yaxis_showgrid=False,
                        )
                        st.plotly_chart(fig_neg, use_container_width=True)
                    else:
                        st.info("No significant negative drivers detected.")

            # --- Overall Representative Chunk Explanation ---
            if "overall" in shap_results:
                with st.expander("🔍 Overall Representative Chunk Explanation", expanded=False):
                    render_section_explanation("overall", shap_results["overall"], "🔍", "#38BDF8")

        else:
            st.info("⚠️ SHAP explanations were not generated. This may be due to an error during processing.")

        st.markdown("---")

        # -------------------------------------------------------------------------
        # Detailed Chunk Explorer Table (preserved from Milestone 1, with section_name)
        # -------------------------------------------------------------------------
    with tab_tech:
        st.subheader("📑 Full Document Chunk Explorer", anchor="full-document-chunk-explorer")
    
        table_data = []
        for chunk in annotated_chunks:
            table_data.append({
                "Chunk ID": chunk["chunk_id"],
                "Section": chunk.get("section_name", "N/A"),
                "Predicted Label": chunk["predicted_label"],
                "Confidence": f"{chunk['confidence_score']*100:.1f}%",
                "Positive": round(chunk["positive_score"], 4),
                "Negative": round(chunk["negative_score"], 4),
                "Neutral": round(chunk["neutral_score"], 4),
                "Words": chunk["word_count"],
                "Snippet": chunk["text"][:120] + "..." if len(chunk["text"]) > 120 else chunk["text"]
            })
        
        explorer_df = pd.DataFrame(table_data)
        st.dataframe(explorer_df, use_container_width=True, hide_index=True)

else:
    # Initial state landing prompt
    st.info("👆 Upload a financial PDF document to begin section-aware sentiment analysis.")
