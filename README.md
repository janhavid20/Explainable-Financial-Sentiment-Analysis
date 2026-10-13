# Explainable Financial Sentiment Analysis Using FinBERT and SHAP

An explainable NLP application that analyzes financial reports, identifies positive, negative, and neutral sentiment, highlights potentially concerning sections, and uses SHAP-based explanations to help users understand model predictions.

## Overview

Financial reports contain large amounts of text, making it difficult to identify important sentiment patterns and potentially concerning disclosures manually.

This project combines **FinBERT, Natural Language Processing (NLP), and SHAP explainability** to analyze financial document text and present the results through an interactive Streamlit dashboard.

Users can upload a PDF report, review document-level and section-level sentiment, inspect highlighted findings, and explore token-level explanations of model predictions.

## Key Features

- **PDF Report Analysis:** Extract text from uploaded financial reports.
- **Section Detection:** Organize report content into identifiable sections.
- **Text Chunking:** Divide long documents into smaller text segments for model inference.
- **FinBERT Sentiment Classification:** Classify text into positive, negative, and neutral sentiment.
- **Sentiment Aggregation:** Combine chunk-level predictions into document-level and section-level results.
- **Sentiment Index:** Calculate a Net Sentiment Index (NSI) using positive and negative sentiment probabilities.
- **Risk-Priority Indicators:** Highlight sections with comparatively high negative sentiment for further review.
- **SHAP Explainability:** Examine token-level contributions to model predictions.
- **Interactive Dashboard:** Explore findings through visualizations and dedicated analysis tabs.
- **Conversational Interface:** Ask questions about the analyzed report and its findings.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| NLP Model | FinBERT (`ProsusAI/finbert`) |
| Model Framework | Hugging Face Transformers and PyTorch |
| Explainable AI | SHAP |
| PDF Text Extraction | PyMuPDF |
| Data Processing | Pandas, NumPy |
| Visualization | Streamlit charts and plotting components |
| Testing | Pytest |

## System Architecture

```text
          Financial PDF Report
                   |
                   v
          PDF Text Extraction
                   |
                   v
            Section Detection
                   |
                   v
             Text Chunking
                   |
                   v
          FinBERT Classification
                   |
          +--------+--------+
          |        |        |
          v        v        v
       Positive  Negative  Neutral
       Sentiment Sentiment Sentiment
          |        |        |
          +--------+--------+
                   |
                   v
          Sentiment Aggregation
                   |
          +--------+---------+
          |                  |
          v                  v
   Section-Level Results   Document-Level
                           Sentiment Index
          |                  |
          +--------+---------+
                   |
                   v
          Risk-Priority Analysis
                   |
                   v
           SHAP Explanations
                   |
                   v
          Streamlit Dashboard
```

## How It Works

### 1. PDF Ingestion

The application extracts text from an uploaded PDF using PyMuPDF. The extracted content becomes the input for the subsequent processing stages.

### 2. Section Detection and Chunking

The text is organized into report sections where recognizable headings are detected. Longer sections are divided into manageable chunks to support model inference within practical input-length limits.

### 3. FinBERT Sentiment Classification

FinBERT is a BERT-based language model adapted for financial text. It predicts the probability of three sentiment classes:

- **Positive:** Text expressing favorable financial sentiment.
- **Negative:** Text expressing unfavorable financial sentiment.
- **Neutral:** Text without a clearly positive or negative sentiment classification.

For a text segment, let \(P_{positive}\) and \(P_{negative}\) represent the model's predicted probabilities.

The Net Sentiment Index is calculated as:

\[
NSI = P_{positive} - P_{negative}
\]

The index ranges from -1 to +1. A negative value indicates that the model assigns greater probability to negative than positive sentiment for the analyzed text.

### 4. Sentiment Aggregation

Chunk-level predictions are combined to produce section-level and document-level summaries. This helps users examine individual report sections rather than relying exclusively on a single overall score.

### 5. Risk-Priority Analysis

Sections with comparatively high negative sentiment can be highlighted for closer inspection. These indicators help prioritize review of potentially concerning passages.

**Important:** Negative sentiment is not equivalent to financial risk. The indicators are sentiment-based screening signals, not validated predictions of bankruptcy, investment performance, creditworthiness, or company solvency.

### 6. SHAP Explainability

SHAP (SHapley Additive exPlanations) is used to investigate how input tokens contribute to a model prediction relative to a selected sentiment class and explanation baseline.

This provides additional insight into why the model may classify a passage as positive, negative, or neutral. Token attributions explain model behavior; they do not establish financial causation or guarantee that a prediction is correct.

### 7. Interactive Dashboard

The Streamlit interface presents the analysis through three main tabs:

- **Report Overview:** Document-level sentiment and summary metrics.
- **Key Findings:** Section-level results and passages prioritized for review.
- **AI Explainability:** SHAP-based visual explanations of model predictions.

The application also provides a conversational interface for exploring report findings.

## Project Structure

```text
Explainable-Financial-Sentiment-Analysis/
│
├── app.py
├── config.py
├── requirements.txt
├── README.md
│
├── .streamlit/
│   └── config.toml
│
├── src/
│   ├── ingestion/
│   │   └── pdf_extractor.py
│   │
│   ├── segmentation/
│   │   ├── section_detector.py
│   │   └── text_chunker.py
│   │
│   ├── models/
│   │   └── finbert_classifier.py
│   │
│   ├── aggregation/
│   │   └── sentiment_aggregator.py
│   │
│   └── explainability/
│       └── shap_explainer.py
│
└── tests/
    ├── test_pipeline.py
    ├── test_milestone2.py
    └── test_conversational_flow.py
```

*Note: This structure reflects the intended module organization. Keep only files and tests that actually exist in your repository.*

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version supported by the project's dependencies
- Git
- Internet access for downloading the pretrained model on its first run

### 1. Clone the Repository

```bash
git clone https://github.com/janhavid20/Explainable-Financial-Sentiment-Analysis.git
cd Explainable-Financial-Sentiment-Analysis
```

### 2. Create a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```bat
venv\Scripts\activate.bat
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to use the application.

The first model initialization may take longer because the pretrained FinBERT model and its dependencies may need to be downloaded.

## Usage

1. Launch the Streamlit application.
2. Upload a supported financial report in PDF format.
3. Allow the application to extract and process the document text.
4. Review the overall sentiment and sentiment index.
5. Explore section-level findings and passages flagged for further review.
6. Open the explainability tab to inspect SHAP token attributions.
7. Use the conversational interface to ask questions about the analyzed report.

**Input limitation:** The application analyzes extractable PDF text. Scanned PDFs that contain only page images may require OCR, depending on the implemented extraction pipeline.

## Testing

Run the available automated tests from the project root:

```bash
python -m pytest tests/ -v
```

Tests cover the pipeline and conversational functionality where implemented. Actual test coverage depends on the test files and the current implementation.

## Explainability and Limitations

- FinBERT provides sentiment classifications, not comprehensive financial advice.
- Results depend on the quality and completeness of extracted PDF text.
- Section detection may not correctly identify every report format.
- Long or complex financial disclosures may lose context when divided into chunks.
- Aggregated sentiment scores can hide important details within individual passages.
- SHAP explanations are model-dependent and should be interpreted with the selected output class and attribution legend in mind.
- Negative sentiment does not automatically indicate poor financial health.
- The application does not replace professional financial analysis, auditing, legal review, or investment advice.

## Future Enhancements

- Support for scanned PDFs through OCR.
- Improved section detection for different financial-report formats.
- Benchmarking against manually annotated financial sentiment datasets.
- Evaluation using appropriate classification metrics, including precision, recall, and F1-score.
- Improved handling of long-range context across report sections.
- Additional comparisons across financial reports and reporting periods.
- Deployment with resource management and model-inference optimizations.

## Research and Learning Objectives

This project explores the integration of:

1. Domain-specific transformer models for financial sentiment analysis.
2. Document processing and hierarchical sentiment aggregation.
3. Explainable AI techniques for interpreting NLP predictions.
4. Interactive visualization for making model outputs more accessible.
5. Sentiment-based screening to support more focused financial-report review.

## Disclaimer

This project is intended for educational and research purposes. Its sentiment classifications, scores, and review-priority indicators may be inaccurate or incomplete. They should not be treated as verified financial risk assessments or used as the sole basis for financial or investment decisions.

## Author

**Janhavi Badgujar**

GitHub: [@janhavid20](https://github.com/janhavid20)

## License

No license has been specified. Unless a license is added to this repository, the code is not automatically granted an open-source license for reuse or redistribution.
