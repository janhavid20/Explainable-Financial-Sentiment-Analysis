# Explainable Financial Sentiment Analysis Using FinBERT and SHAP

[![Python](https://img.shields.io/badge/Python-3.11-blue)]
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)]
[![Transformers](https://img.shields.io/badge/HuggingFace-FinBERT-yellow)]
[![SHAP](https://img.shields.io/badge/XAI-SHAP-green)]

An Explainable AI-powered Financial NLP system that analyzes annual reports, earnings reports, risk disclosures, and other financial documents using FinBERT while providing transparent model explanations through SHAP.

---

## Author

### Janhavi Badgujar
Final Year B.Tech (Artificial Intelligence & Machine Learning)

Primary Author, System Designer, Developer, and Research Contributor

### Contributions
- Financial NLP Pipeline Design
- FinBERT Integration
- SHAP Explainability Framework
- PDF Processing Pipeline
- Section-Level Risk Analysis
- Streamlit Dashboard Development
- Data Aggregation & Visualization
- Testing and Validation

---

## Project Overview

Financial reports often contain hundreds of pages of information that are difficult to analyze manually.

This project automates financial document analysis by:

- Extracting text from PDF reports
- Identifying document sections
- Performing sentiment analysis using FinBERT
- Detecting high-risk financial disclosures
- Generating document-level insights
- Explaining predictions using SHAP
- Presenting results through an interactive dashboard

---

## System Architecture

PDF Upload
↓
Text Extraction (PyMuPDF)
↓
Section Detection
↓
Smart Chunking
↓
FinBERT Sentiment Analysis
↓
Risk Assessment
↓
Document Aggregation
↓
SHAP Explainability
↓
Interactive Dashboard

---

## Features

✔ Financial Document Analysis

✔ FinBERT Sentiment Classification

✔ Section-wise Sentiment Analysis

✔ Risk Detection Framework

✔ Explainable AI using SHAP

✔ PDF Processing Pipeline

✔ Interactive Streamlit Dashboard

✔ Financial Insight Generation

✔ Annual Report Analysis

✔ Earnings Report Analysis

✔ Auditor Report Analysis

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
