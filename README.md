# Explainable Financial Sentiment Analysis Using FinBERT and SHAP
## Milestone 1 Working Prototype

Production-quality working prototype for financial document text extraction, chunking, FinBERT sentiment inference, and interactive visualization via Streamlit.

---

## 1. Folder Structure

```
finbert-sentiment-analysis/
│
├── config.py                      # Centralized configuration (model, batching, devices)
├── requirements.txt               # Dependencies list
├── README.md                      # Setup and execution guide
├── app.py                         # Streamlit User Interface
│
└── src/                           # Clean Architecture Core Modules
    ├── __init__.py
    ├── ingestion/                 # Document Ingestion Engine
    │   ├── __init__.py
    │   └── pdf_extractor.py       # PyMuPDF (fitz) text extractor
    │
    ├── segmentation/              # Text Segmentation & Chunking
    │   ├── __init__.py
    │   └── text_chunker.py        # Sentence-aware chunker with sliding window overlap
    │
    ├── models/                    # Sentiment Model Wrapper
    │   ├── __init__.py
    │   └── finbert_classifier.py  # ProsusAI/finbert model loader & batch inferencer
    │
    └── aggregation/               # Metric Aggregation
        ├── __init__.py
        └── sentiment_aggregator.py# Document-level aggregation & top 5 positive/negative chunk rankings
```

---

## 2. Setup Instructions

### Prerequisites
- Python 3.10 or 3.11 installed
- Git

### Step-by-Step Installation

1. **Clone or navigate to project directory**:
   ```bash
   cd d:/vit/7sem/aml/finbert
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**:
   - **Windows (PowerShell)**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **Windows (CMD)**:
     ```cmd
     .\venv\Scripts\activate.bat
     ```
   - **Linux / macOS**:
     ```bash
     source venv/bin/activate
     ```

4. **Install PyTorch & Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 3. Run Instructions

1. **Launch Streamlit Web Application**:
   ```bash
   streamlit run app.py
   ```

2. **Access Dashboard**:
   Open your web browser and navigate to `http://localhost:8501`.

3. **Upload Financial PDF**:
   Upload any financial PDF document (e.g., SEC 10-K, 10-Q report, earnings release, or financial news PDF).

4. **View Outputs**:
   - **Executive Summary Metrics**: Overall document sentiment, average model confidence, total chunks analyzed, Net Sentiment Index (NSI).
   - **Interactive Charts**: Donut chart distribution of Positive/Negative/Neutral chunks and bar chart of mean probabilities.
   - **Top 5 Highlights**: Expandable views of Top 5 Most Positive and Top 5 Most Negative chunks with probabilities.
   - **Full Chunk Explorer**: Complete tabular breakdown of all parsed chunks.
