"""
PDF Ingestion and Text Extraction Module using PyMuPDF (fitz)
"""

import fitz  # PyMuPDF
import time
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class PDFExtractor:
    """
    Extracts structured text content from financial PDF documents.
    """

    @staticmethod
    def extract_text_from_bytes(pdf_bytes: bytes, timeout_seconds: float = 30.0) -> Dict[str, Any]:
        """
        Extracts narrative text page-by-page from raw PDF byte stream.

        Args:
            pdf_bytes: Byte stream of the uploaded PDF file.
            timeout_seconds: Maximum seconds allowed for extraction before aborting.

        Returns:
            Dict containing full concatenated text, page breakdown, and document metadata.

        Raises:
            ValueError: If pdf_bytes is empty or None.
            RuntimeError: If PyMuPDF fails to parse the document.
            TimeoutError: If extraction exceeds timeout_seconds.
        """
        start_time = time.time()

        # Validate input bytes
        if not pdf_bytes:
            raise ValueError("Empty PDF byte stream provided.")

        byte_size = len(pdf_bytes)
        logger.info(f"[PDF Extractor] Received PDF: {byte_size:,} bytes ({byte_size / 1024:.1f} KB)")

        if byte_size < 100:
            raise ValueError(f"PDF file is too small ({byte_size} bytes) - likely corrupted or empty.")

        # Open document
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            raise RuntimeError(f"Failed to parse PDF document with PyMuPDF: {str(e)}")

        total_pages = len(doc)
        logger.info(f"[PDF Extractor] Document opened: {total_pages} pages detected")

        if total_pages == 0:
            doc.close()
            raise ValueError("PDF document contains 0 pages.")

        pages_data: List[Dict[str, Any]] = []
        full_text_blocks: List[str] = []

        for page_idx in range(total_pages):
            # Timeout guard per page
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                doc.close()
                raise TimeoutError(
                    f"PDF extraction timed out after {elapsed:.1f}s "
                    f"(processed {page_idx}/{total_pages} pages). "
                    f"The document may be too large or contain complex vector graphics."
                )

            page = doc[page_idx]
            page_text = page.get_text("text")

            # Basic cleaning: strip null bytes and normalize
            cleaned_page_text = page_text.replace("\x00", "").strip()

            if cleaned_page_text:
                pages_data.append({
                    "page_number": page_idx + 1,
                    "text": cleaned_page_text,
                    "char_count": len(cleaned_page_text),
                    "word_count": len(cleaned_page_text.split())
                })
                full_text_blocks.append(cleaned_page_text)

            # Log progress every 20 pages
            if (page_idx + 1) % 20 == 0:
                logger.info(f"[PDF Extractor] Processed {page_idx + 1}/{total_pages} pages...")

        # Capture total_pages before closing
        doc.close()

        full_text = "\n\n".join(full_text_blocks)
        extraction_time = round(time.time() - start_time, 3)

        result = {
            "full_text": full_text,
            "pages": pages_data,
            "total_pages": total_pages,
            "extracted_pages_count": len(pages_data),
            "total_char_count": len(full_text),
            "total_word_count": len(full_text.split()),
            "extraction_time_seconds": extraction_time
        }

        logger.info(
            f"[PDF Extractor] Extraction complete in {extraction_time}s: "
            f"{total_pages} pages, {result['extracted_pages_count']} with text, "
            f"{result['total_word_count']:,} words, {result['total_char_count']:,} chars"
        )

        return result
