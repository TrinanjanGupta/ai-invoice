"""
preprocessing/document_segmenter.py

Document Segmentation & Merged-Invoice Classifier.
Detects invoice boundaries across multi-page document uploads, determining whether
a multi-page file contains:
1. A single multi-page invoice (e.g. Page 1 header + line items, Page 2 totals).
2. Multiple merged sub-invoices (e.g. Invoice A on pp. 1-2, Invoice B on p. 3, Invoice C on pp. 4-5).

Employs multi-signal boundary detection:
- Header & document title indicators ("TAX INVOICE", "BILL OF SUPPLY", "ORIGINAL FOR RECIPIENT")
- Discontinuous invoice numbers across pages
- Distinct vendor GSTINs on later pages
- Page numbering continuity ("Page 1 of 2" vs "Page 1 of 1")
- Explicit continuation markers ("Continued...", "Contd...")
- Totals block termination
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Any
from loguru import logger

INVOICE_TITLE_PATTERNS = [
    re.compile(r"\b(?:tax\s+invoice|bill\s+of\s+supply|commercial\s+invoice|retail\s+invoice|cash\s+memo|e-invoice)\b", re.IGNORECASE),
    re.compile(r"\b(?:original\s+for\s+recipient|duplicate\s+for\s+transporter|triplicate\s+for\s+supplier)\b", re.IGNORECASE),
]

CONTINUATION_PATTERNS = [
    re.compile(r"\b(?:contd|continued|cont\.\.\.|page\s+\d+\s+of\s+[2-9]\d*)\b", re.IGNORECASE),
    re.compile(r"\b(?:carried\s+forward|c/f|b/f|brought\s+forward)\b", re.IGNORECASE),
]

INVOICE_NO_PATTERNS = [
    re.compile(r"(?:invoice|bill|inv)\s*(?:no|num|number|#)\s*[:.\s-]*([A-Z0-9/\-_]{2,30})", re.IGNORECASE),
    re.compile(r"(?:invoice|bill|inv)\s*[:#-]\s*([A-Z0-9/\-_]{2,30})", re.IGNORECASE),
]

GSTIN_PATTERN = re.compile(r"\b([0-3][0-9][A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1})\b")
TOTALS_PATTERN = re.compile(r"\b(?:grand\s+total|total\s+amount|net\s+payable|amount\s+payable|balance\s+due)\b", re.IGNORECASE)


@dataclass
class DocumentSegment:
    segment_index: int
    page_indices: list[int]  # 1-indexed page numbers (e.g. [1, 2])
    invoice_number_hint: Optional[str] = None
    vendor_gstin_hint: Optional[str] = None
    is_multi_page: bool = False
    confidence: float = 1.0


@dataclass
class PageSignals:
    page_num: int
    invoice_number: Optional[str] = None
    vendor_gstin: Optional[str] = None
    has_invoice_title: bool = False
    has_continuation_marker: bool = False
    has_totals_block: bool = False
    text_length: int = 0


class DocumentSegmenter:
    """
    Classifies page boundaries in multi-page document uploads to prevent multi-invoice aggregation errors.
    """

    def extract_page_signals(self, page_num: int, page_text: str) -> PageSignals:
        clean_text = page_text.strip()

        # 1. Invoice title
        has_title = any(p.search(clean_text) for p in INVOICE_TITLE_PATTERNS)

        # 2. Continuation markers
        has_continuation = any(p.search(clean_text) for p in CONTINUATION_PATTERNS)

        # 3. Totals block
        has_totals = bool(TOTALS_PATTERN.search(clean_text))

        # 4. Invoice number candidate (requires digits / alphanumeric token)
        inv_no = None
        for pat in INVOICE_NO_PATTERNS:
            for m in pat.finditer(clean_text):
                cand = m.group(1).strip()
                cand_clean = re.sub(r"[^A-Z0-9]", "", cand.upper())
                if cand_clean in ("INVOICE", "DATE", "TOTAL", "TAX", "GSTIN", "BILL", "SUPPLY", "MEMO", "ORIGINAL"):
                    continue
                if len(cand) >= 2 and any(ch.isalnum() for ch in cand):
                    inv_no = cand
                    break
            if inv_no:
                break

        # 5. Vendor GSTIN candidate (first occurring in top portion)
        gstin = None
        gstin_matches = GSTIN_PATTERN.findall(clean_text)
        if gstin_matches:
            gstin = gstin_matches[0]

        return PageSignals(
            page_num=page_num,
            invoice_number=inv_no,
            vendor_gstin=gstin,
            has_invoice_title=has_title,
            has_continuation_marker=has_continuation,
            has_totals_block=has_totals,
            text_length=len(clean_text),
        )

    def segment(self, pages_text: dict[int, str]) -> list[DocumentSegment]:
        """
        Segments a multi-page document into distinct invoice page groups.
        pages_text: dict mapping 1-indexed page numbers to raw extracted page text.
        """
        if not pages_text:
            return [DocumentSegment(segment_index=0, page_indices=[1], is_multi_page=False)]

        sorted_pages = sorted(pages_text.keys())
        if len(sorted_pages) == 1:
            signals = self.extract_page_signals(sorted_pages[0], pages_text[sorted_pages[0]])
            return [
                DocumentSegment(
                    segment_index=0,
                    page_indices=[sorted_pages[0]],
                    invoice_number_hint=signals.invoice_number,
                    vendor_gstin_hint=signals.vendor_gstin,
                    is_multi_page=False,
                )
            ]

        # Extract signals across all pages
        all_signals: dict[int, PageSignals] = {}
        for p in sorted_pages:
            all_signals[p] = self.extract_page_signals(p, pages_text[p])

        segments: list[DocumentSegment] = []
        current_pages: list[int] = [sorted_pages[0]]
        current_inv_no = all_signals[sorted_pages[0]].invoice_number
        current_gstin = all_signals[sorted_pages[0]].vendor_gstin

        for idx in range(1, len(sorted_pages)):
            p = sorted_pages[idx]
            prev_p = sorted_pages[idx - 1]
            sig = all_signals[p]
            prev_sig = all_signals[prev_p]

            is_new_segment = False

            # Condition A: Discontinuous Invoice Number on later page
            if sig.invoice_number and current_inv_no:
                clean_curr = re.sub(r"[^A-Z0-9]", "", current_inv_no.upper())
                clean_new = re.sub(r"[^A-Z0-9]", "", sig.invoice_number.upper())
                if clean_curr and clean_new and clean_curr != clean_new and clean_new not in clean_curr:
                    is_new_segment = True

            # Condition B: Different Vendor GSTIN on later page
            if not is_new_segment and sig.vendor_gstin and current_gstin:
                if sig.vendor_gstin.upper() != current_gstin.upper():
                    is_new_segment = True

            # Condition C: New explicit Invoice Title following a completed Totals block without continuation marker
            if not is_new_segment:
                if sig.has_invoice_title and prev_sig.has_totals_block and not sig.has_continuation_marker:
                    is_new_segment = True

            if is_new_segment:
                # Flush current segment
                segments.append(
                    DocumentSegment(
                        segment_index=len(segments),
                        page_indices=list(current_pages),
                        invoice_number_hint=current_inv_no,
                        vendor_gstin_hint=current_gstin,
                        is_multi_page=len(current_pages) > 1,
                    )
                )
                # Start new segment
                current_pages = [p]
                current_inv_no = sig.invoice_number
                current_gstin = sig.vendor_gstin
            else:
                current_pages.append(p)
                if not current_inv_no and sig.invoice_number:
                    current_inv_no = sig.invoice_number
                if not current_gstin and sig.vendor_gstin:
                    current_gstin = sig.vendor_gstin

        # Flush final segment
        if current_pages:
            segments.append(
                DocumentSegment(
                    segment_index=len(segments),
                    page_indices=list(current_pages),
                    invoice_number_hint=current_inv_no,
                    vendor_gstin_hint=current_gstin,
                    is_multi_page=len(current_pages) > 1,
                )
            )

        if len(segments) > 1:
            logger.info(
                f"[DocumentSegmenter] Detected {len(segments)} distinct invoices in multi-page upload: "
                f"{[s.page_indices for s in segments]}"
            )

        return segments
