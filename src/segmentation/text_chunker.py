"""
Text Chunking Module for Financial Document Segmentation
"""

import re
from typing import List, Dict, Any
from config import APPROX_WORDS_PER_CHUNK, APPROX_WORD_OVERLAP


class TextChunker:
    """
    Splits long financial text into semantically coherent, sentence-aware text chunks
    suitable for transformer model inference (FinBERT context limit).
    """

    def __init__(self, target_words_per_chunk: int = APPROX_WORDS_PER_CHUNK, overlap_words: int = APPROX_WORD_OVERLAP):
        """
        Initialize chunker with target word count and overlap bounds.
        """
        self.target_words_per_chunk = target_words_per_chunk
        self.overlap_words = overlap_words

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Splits text into sentences using regular expression rules, preserving financial abbreviations.
        """
        # Normalize double newlines to paragraph markers
        text = re.sub(r'\n+', ' ', text)
        
        # Split on sentence-ending punctuation followed by whitespace and capital letter
        sentence_end = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])')
        sentences = sentence_end.split(text)
        
        # Clean whitespace per sentence
        cleaned_sentences = [s.strip() for s in sentences if s and len(s.strip()) > 5]
        return cleaned_sentences

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Chunks input text into sentence-aware blocks with sliding window overlap.

        Args:
            text: Full raw narrative string.

        Returns:
            List of chunk dictionaries with metadata.
        """
        if not text or not text.strip():
            return []

        sentences = self._split_into_sentences(text)
        if not sentences:
            # Fallback if no clean sentences were parsed
            words = text.split()
            sentences = [" ".join(words[i:i+50]) for i in range(0, len(words), 50)]

        chunks: List[Dict[str, Any]] = []
        current_chunk_sentences: List[str] = []
        current_word_count = 0
        chunk_id = 1

        for sentence in sentences:
            sentence_words = len(sentence.split())

            # If adding this sentence exceeds chunk size and we already have content
            if current_word_count + sentence_words > self.target_words_per_chunk and current_chunk_sentences:
                chunk_str = " ".join(current_chunk_sentences)
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_str,
                    "word_count": len(chunk_str.split()),
                    "sentence_count": len(current_chunk_sentences)
                })
                chunk_id += 1

                # Retain overlapping sentences for context continuity
                overlap_sentences: List[str] = []
                accumulated_overlap = 0
                for prev_sent in reversed(current_chunk_sentences):
                    prev_words = len(prev_sent.split())
                    if accumulated_overlap + prev_words <= self.overlap_words:
                        overlap_sentences.insert(0, prev_sent)
                        accumulated_overlap += prev_words
                    else:
                        break

                current_chunk_sentences = overlap_sentences
                current_word_count = accumulated_overlap

            current_chunk_sentences.append(sentence)
            current_word_count += sentence_words

        # Append final remaining chunk
        if current_chunk_sentences:
            chunk_str = " ".join(current_chunk_sentences)
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_str,
                "word_count": len(chunk_str.split()),
                "sentence_count": len(current_chunk_sentences)
            })

        return chunks

    def chunk_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunks each section independently and stamps section_name on every chunk.

        Section boundaries reset the sliding window — no overlap carries across
        sections. This prevents sentiment contamination between semantically
        distinct report areas (e.g., Risk Factors → Legal Proceedings).

        Args:
            sections: List of section dicts from SectionDetector, each containing
                      at minimum 'section_name' and 'text'.

        Returns:
            Globally-ordered list of chunk dicts, each containing:
                chunk_id, section_name, text, word_count, sentence_count.
        """
        if not sections:
            return []

        all_chunks: List[Dict[str, Any]] = []
        global_chunk_id = 1

        for section in sections:
            section_name = section.get("section_name", "Uncategorized")
            section_text = section.get("text", "")

            if not section_text or not section_text.strip():
                continue

            # Chunk this section using the existing sentence-aware chunker
            section_chunks = self.chunk_text(section_text)

            # Stamp section metadata and reassign globally unique chunk IDs
            for chunk in section_chunks:
                chunk["chunk_id"] = global_chunk_id
                chunk["section_name"] = section_name
                global_chunk_id += 1
                all_chunks.append(chunk)

        return all_chunks

