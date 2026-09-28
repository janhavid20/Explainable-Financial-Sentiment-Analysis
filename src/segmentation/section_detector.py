"""
Financial Report Section Detection Module

Detects section headings in financial documents (10-K, 10-Q, earnings reports)
and splits the full text into labeled sections for section-aware sentiment analysis.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Curated taxonomy of financial report section heading patterns.
# Keys are canonical section names; values are case-insensitive regex patterns.
SECTION_PATTERNS: Dict[str, str] = {
    "Executive Summary":     r"executive\s+summary|overview|highlights|at\s+a\s+glance",
    "Business Description":  r"business\s+(description|overview)|item\s+1[.\s]|our\s+business|company\s+overview",
    "Risk Factors":          r"risk\s+factors|item\s+1a[.\s]|principal\s+risks|key\s+risks",
    "MD&A":                  r"management.s?\s+discussion|item\s+7[.\s]|md\s*&\s*a|results\s+of\s+operations",
    "Financial Statements":  r"financial\s+statements|item\s+8[.\s]|consolidated\s+(balance|statements|income)",
    "Legal Proceedings":     r"legal\s+proceedings|item\s+3[.\s]|litigation|contingencies",
    "Market Risk":           r"market\s+risk|item\s+7a[.\s]|quantitative.*disclosures",
    "Controls & Procedures": r"controls\s+and\s+procedures|item\s+9a|internal\s+control",
    "Outlook":               r"outlook|forward.looking|future\s+plans|guidance|projections",
    "Compensation":          r"compensation|executive\s+pay|remuneration|item\s+11[.\s]",
    "Governance":            r"corporate\s+governance|board\s+of\s+directors|item\s+10[.\s]",
    "Notes to Financial":    r"notes\s+to\s+(the\s+)?financial|footnotes",
    "Auditor Report":        r"auditor.?s?\s+report|independent\s+auditor|report\s+of\s+independent",
    "Revenue & Growth":      r"revenue|growth|sales\s+(performance|analysis)|top.?line",
    "Debt & Liquidity":      r"debt|liquidity|capital\s+resources|borrowings|credit\s+facilit",
    "Dividends":             r"dividends?|shareholder\s+returns|distributions",
}

# Heading heuristics scoring thresholds
MIN_HEADING_SCORE = 3
MAX_HEADING_LINE_WORDS = 15


class SectionDetector:
    """
    Detects financial report section boundaries using heading heuristics
    and splits document text into labeled sections.
    """

    def __init__(self, custom_patterns: Optional[Dict[str, str]] = None):
        """
        Initialize with built-in or custom section heading patterns.

        Args:
            custom_patterns: Optional dict of {section_name: regex_pattern} to
                             merge with or override built-in patterns.
        """
        self.patterns = {**SECTION_PATTERNS}
        if custom_patterns:
            self.patterns.update(custom_patterns)

        # Pre-compile all regex patterns for performance
        self._compiled_patterns: List[Tuple[str, re.Pattern]] = [
            (name, re.compile(pattern, re.IGNORECASE))
            for name, pattern in self.patterns.items()
        ]

    def _score_heading_line(self, line: str) -> Tuple[int, str]:
        """
        Score a line's likelihood of being a section heading.

        Returns:
            Tuple of (score, matched_section_name).
            Score of 0 means not a heading. Higher = more confident.
        """
        stripped = line.strip()
        if not stripped or len(stripped) < 3:
            return 0, ""

        word_count = len(stripped.split())

        # Headings are typically short (< 15 words)
        if word_count > MAX_HEADING_LINE_WORDS:
            return 0, ""

        score = 0
        matched_section = ""

        # Heuristic 1: Pattern keyword match (strongest signal, +4)
        for section_name, compiled_re in self._compiled_patterns:
            if compiled_re.search(stripped):
                score += 4
                matched_section = section_name
                break

        # If no keyword matched, this line is unlikely a heading
        if score == 0:
            return 0, ""

        # Heuristic 2: ALL CAPS or Title Case (+1)
        if stripped.isupper():
            score += 1
        elif stripped.istitle():
            score += 1

        # Heuristic 3: Short line — likely a standalone heading (+1)
        if word_count <= 8:
            score += 1

        # Heuristic 4: Starts with numbering like "Item 1A." or "PART II" (+1)
        if re.match(r"^(item\s+\d|part\s+[ivx\d]|\d+[.\)]\s)", stripped, re.IGNORECASE):
            score += 1

        return score, matched_section

    def detect_sections(self, full_text: str) -> List[Dict[str, Any]]:
        """
        Detect section boundaries in the full document text and split into labeled sections.

        Args:
            full_text: Complete extracted document text.

        Returns:
            List of section dictionaries, each containing:
                - section_name: Canonical section name or "Uncategorized"
                - text: Full text content of the section
                - start_line: Starting line index in original text
                - word_count: Number of words in the section
                - heading_line: Original heading text that triggered detection
        """
        if not full_text or not full_text.strip():
            logger.warning("[SectionDetector] Empty text provided.")
            return []

        lines = full_text.split("\n")
        total_lines = len(lines)
        logger.info(f"[SectionDetector] Scanning {total_lines} lines for section headings...")

        # Phase 1: Score every line and identify heading boundaries
        boundaries: List[Dict[str, Any]] = []

        for line_idx, line in enumerate(lines):
            score, section_name = self._score_heading_line(line)
            if score >= MIN_HEADING_SCORE:
                boundaries.append({
                    "line_idx": line_idx,
                    "section_name": section_name,
                    "heading_line": line.strip(),
                    "score": score
                })

        logger.info(f"[SectionDetector] Found {len(boundaries)} section heading(s)")

        # Phase 2: Deduplicate — if same section_name appears multiple times,
        # keep only the first occurrence (common in tables of contents vs actual section)
        seen_sections: set = set()
        deduped_boundaries: List[Dict[str, Any]] = []
        for b in boundaries:
            if b["section_name"] not in seen_sections:
                seen_sections.add(b["section_name"])
                deduped_boundaries.append(b)
            else:
                logger.debug(
                    f"[SectionDetector] Skipping duplicate heading '{b['section_name']}' at line {b['line_idx']}"
                )
        boundaries = deduped_boundaries

        # Phase 3: Fallback — if no headings detected, wrap entire document as one section
        if not boundaries:
            logger.info("[SectionDetector] No section headings detected — treating as single section.")
            return [{
                "section_name": "Full Document",
                "text": full_text.strip(),
                "start_line": 0,
                "word_count": len(full_text.split()),
                "heading_line": ""
            }]

        # Phase 4: Build section text blocks from boundary ranges
        sections: List[Dict[str, Any]] = []

        # Handle preamble text before the first detected heading
        first_boundary_line = boundaries[0]["line_idx"]
        if first_boundary_line > 0:
            preamble_lines = lines[:first_boundary_line]
            preamble_text = "\n".join(preamble_lines).strip()
            if preamble_text and len(preamble_text.split()) > 10:
                sections.append({
                    "section_name": "Preamble / Cover Page",
                    "text": preamble_text,
                    "start_line": 0,
                    "word_count": len(preamble_text.split()),
                    "heading_line": ""
                })

        # Build sections from consecutive heading boundaries
        for i, boundary in enumerate(boundaries):
            start_line = boundary["line_idx"]

            # End line is the next boundary's start, or end of document
            if i + 1 < len(boundaries):
                end_line = boundaries[i + 1]["line_idx"]
            else:
                end_line = total_lines

            section_lines = lines[start_line:end_line]
            section_text = "\n".join(section_lines).strip()

            if section_text:
                sections.append({
                    "section_name": boundary["section_name"],
                    "text": section_text,
                    "start_line": start_line,
                    "word_count": len(section_text.split()),
                    "heading_line": boundary["heading_line"]
                })

        logger.info(
            f"[SectionDetector] Produced {len(sections)} sections: "
            f"{[s['section_name'] for s in sections]}"
        )

        return sections
