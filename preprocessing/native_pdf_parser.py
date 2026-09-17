"""
preprocessing/native_pdf_parser.py

Authoritative Vector Geometry & Direct Text Extractor for Digital PDFs.

Uses PyMuPDF (fitz) to extract text blocks, lines, spans, font hierarchies,
vector table grids, and deterministic geometry zones directly from vector PDF layers.

Eliminates DocLayout-YOLO inaccuracies on digital native PDFs:
- Vendor Name: Derived from maximum font size / bold hierarchy in top 25% of page 1.
- Invoice Details: Mapped from deterministic vector anchors and neighboring tokens.
- Buyer Details: Mapped from 'Bill To' / 'Buyer' / 'Consignee' geometric blocks.
- Line Items: Extracted with precision from vector table grids (find_tables).
- Totals: Mapped from bottom-right numeric blocks and tax headers.
- Produces normalized 0-1000 RegionBlocks and WordTokens with 0.99 confidence.
"""

import re
import pymupdf
from dataclasses import dataclass, field
from typing import Optional, Any
from loguru import logger

from preprocessing.document_profile import WordToken, RegionBlock, normalize_box


GENERIC_HEADER_TERMS = {
    "invoice", "tax invoice", "bill of supply", "original for recipient",
    "duplicate for transporter", "triplicate for supplier", "cash memo",
    "retail invoice", "commercial invoice", "proforma invoice", "e-signed",
    "page 1 of 1", "page 1 of 2", "tax invoice/bill of supply"
}

ANCHOR_REGEXES = {
    "invoice_number": re.compile(r"\b(?:invoice|inv|bill)\b\s*(?:number|num|no|#)\b[\s.:#]*([A-Za-z0-9\-_/]+)", re.IGNORECASE),
    "invoice_date": re.compile(r"\b(?:invoice\s*date|bill\s*date|dated|issue\s*date)\b[\s.:#]*(\d{1,2}[-/.][A-Za-z0-9]+[-/.]\d{2,4})", re.IGNORECASE),
    "due_date": re.compile(r"\b(?:due\s*date|payment\s*due)\b[\s.:#]*(\d{1,2}[-/.][A-Za-z0-9]+[-/.]\d{2,4})", re.IGNORECASE),
    "po_number": re.compile(r"\b(?:p\.?o\.?|purchase\s*order)\b\s*(?:number|num|no|#)?\b[\s.:#]*([A-Za-z0-9\-_/]+)", re.IGNORECASE),
    "place_of_supply": re.compile(r"\b(?:place\s*of\s*supply|state)\b[\s.:#]*([A-Za-z\s]+?)(?:\s*\(|\n|$)", re.IGNORECASE),
    "gstin": re.compile(r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}\b"),
    "pan": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b"),
}


@dataclass
class NativeSpan:
    text: str
    size: float
    font: str
    flags: int
    is_bold: bool
    bbox: list[float]          # [x0, y0, x1, y1] in PDF points
    bbox_norm: list[int]       # [0..1000]


@dataclass
class NativeLine:
    spans: list[NativeSpan]
    text: str
    bbox: list[float]
    bbox_norm: list[int]
    block_no: int
    line_no: int


@dataclass
class NativeBlock:
    lines: list[NativeLine]
    text: str
    bbox: list[float]
    bbox_norm: list[int]
    block_no: int
    zone_label: Optional[str] = None


@dataclass
class NativeVectorPage:
    page_num: int
    page_width: float
    page_height: float
    full_text: str
    blocks: list[NativeBlock]
    lines: list[NativeLine]
    words: list[WordToken]
    regions: list[RegionBlock]
    table_items: list[dict[str, Any]] = field(default_factory=list)
    table_bbox: Optional[list[float]] = None
    extracted_fields: dict[str, dict[str, Any]] = field(default_factory=dict)
    vendor_name_candidate: Optional[str] = None
    buyer_name_candidate: Optional[str] = None


class NativePDFParser:
    """
    Direct PyMuPDF vector geometry and text layer parser.
    Extracts structured layout and field evidence without rasterization degradation.
    """

    def __init__(self):
        pass

    def parse_page(self, page: pymupdf.Page, page_num: int = 1) -> NativeVectorPage:
        """
        Extracts native vector hierarchy, geometry zones, and field candidates from a single PDF page.
        """
        pw = float(page.rect.width)
        ph = float(page.rect.height)
        pw_safe = max(1.0, pw)
        ph_safe = max(1.0, ph)

        def to_norm(bbox: list[float]) -> list[int]:
            return [
                max(0, min(1000, int(round(1000.0 * bbox[0] / pw_safe)))),
                max(0, min(1000, int(round(1000.0 * bbox[1] / ph_safe)))),
                max(0, min(1000, int(round(1000.0 * bbox[2] / pw_safe)))),
                max(0, min(1000, int(round(1000.0 * bbox[3] / ph_safe)))),
            ]

        # 1. Direct PyMuPDF text dictionary extraction
        raw_dict = page.get_text("dict")
        raw_blocks = raw_dict.get("blocks", [])

        blocks: list[NativeBlock] = []
        all_lines: list[NativeLine] = []
        word_tokens: list[WordToken] = []

        w_counter = 0

        for b_idx, b in enumerate(raw_blocks):
            if b.get("type") != 0:  # 0 = text block, 1 = image block
                continue

            b_bbox = [float(x) for x in b.get("bbox", [0, 0, 0, 0])]
            b_norm = to_norm(b_bbox)
            block_lines: list[NativeLine] = []

            for l_idx, l in enumerate(b.get("lines", [])):
                l_bbox = [float(x) for x in l.get("bbox", [0, 0, 0, 0])]
                l_norm = to_norm(l_bbox)
                spans: list[NativeSpan] = []

                for s in l.get("spans", []):
                    s_text = s.get("text", "")
                    if not s_text or not s_text.strip():
                        continue

                    s_bbox = [float(x) for x in s.get("bbox", [0, 0, 0, 0])]
                    s_norm = to_norm(s_bbox)
                    font_name = str(s.get("font", ""))
                    flags = int(s.get("flags", 0))
                    is_bold = bool(flags & 2 or "bold" in font_name.lower() or "black" in font_name.lower())
                    size = float(s.get("size", 10.0))

                    span_obj = NativeSpan(
                        text=s_text,
                        size=size,
                        font=font_name,
                        flags=flags,
                        is_bold=is_bold,
                        bbox=s_bbox,
                        bbox_norm=s_norm,
                    )
                    spans.append(span_obj)

                    # Extract words from span
                    span_words = s_text.split()
                    if span_words:
                        span_w = max(1.0, s_bbox[2] - s_bbox[0])
                        word_char_w = span_w / max(1, len(s_text))
                        cur_char_offset = 0

                        for sw in span_words:
                            sw_len = len(sw)
                            # Approximate word bbox within span
                            sw_x0 = s_bbox[0] + cur_char_offset * word_char_w
                            sw_x1 = min(s_bbox[2], sw_x0 + sw_len * word_char_w)
                            sw_bbox = [sw_x0, s_bbox[1], sw_x1, s_bbox[3]]
                            sw_norm = to_norm(sw_bbox)

                            word_tokens.append(
                                WordToken(
                                    text=sw,
                                    bbox_norm=sw_norm,
                                    bbox_raw=sw_bbox,
                                    confidence=0.99,
                                    page=page_num,
                                    block_no=b_idx,
                                    line_no=l_idx,
                                    word_no=w_counter,
                                    source="native_pdf",
                                )
                            )
                            w_counter += 1
                            cur_char_offset += sw_len + 1

                if spans:
                    l_text = "".join(s.text for s in spans).strip()
                    line_obj = NativeLine(
                        spans=spans,
                        text=l_text,
                        bbox=l_bbox,
                        bbox_norm=l_norm,
                        block_no=b_idx,
                        line_no=l_idx,
                    )
                    block_lines.append(line_obj)
                    all_lines.append(line_obj)

            if block_lines:
                b_text = "\n".join(l.text for l in block_lines).strip()
                blocks.append(
                    NativeBlock(
                        lines=block_lines,
                        text=b_text,
                        bbox=b_bbox,
                        bbox_norm=b_norm,
                        block_no=b_idx,
                    )
                )

        full_text = "\n".join(b.text for b in blocks)

        # 2. Table detection using PyMuPDF vector table finder
        table_items: list[dict[str, Any]] = []
        table_bbox: Optional[list[float]] = None
        table_region_block: Optional[RegionBlock] = None

        try:
            from understanding.table_extractor import TableExtractor
            tbl_extractor = TableExtractor()
            tabs = page.find_tables()
            if tabs.tables:
                primary_tab = tabs.tables[0]
                table_bbox = list(primary_tab.bbox)
                tbl_norm = to_norm(table_bbox)
                table_region_block = RegionBlock(
                    label="line_items",
                    bbox_norm=tbl_norm,
                    bbox_raw=table_bbox,
                    confidence=0.98,
                    page=page_num,
                )
                raw_rows = primary_tab.extract()
                if raw_rows and len(raw_rows) >= 2:
                    table_items = tbl_extractor._parse_table_rows(raw_rows)
        except Exception as e:
            logger.debug(f"Native table extraction note on page {page_num}: {e}")

        # 3. Deterministic Geometric Zone Classification
        regions: list[RegionBlock] = []
        extracted_fields: dict[str, dict[str, Any]] = {}

        if table_region_block:
            regions.append(table_region_block)

        # Classify text blocks into Header, Vendor, Buyer, Totals, Footer
        vendor_spans: list[NativeSpan] = []
        vendor_name_candidate: Optional[str] = None
        buyer_name_candidate: Optional[str] = None

        vendor_blocks: list[NativeBlock] = []
        buyer_blocks: list[NativeBlock] = []
        header_blocks: list[NativeBlock] = []
        totals_blocks: list[NativeBlock] = []
        footer_blocks: list[NativeBlock] = []

        table_top_y = table_region_block.bbox_norm[1] if table_region_block else 500
        table_bot_y = table_region_block.bbox_norm[3] if table_region_block else 650

        # Scan top zone (y_norm <= 350) for Vendor Name & Header
        for b in blocks:
            bn = b.bbox_norm
            b_text_lower = b.text.lower()

            # Ignore running page headers / URL stamp (e.g. Issue Date: ... | URL: ... Page 1 of 1 at very top or bottom)
            if (bn[1] > 950 or (bn[1] < 60 and "url:" in b_text_lower and "page" in b_text_lower)):
                footer_blocks.append(b)
                continue

            # A. Header / Invoice Details (Invoice No, Date, PO, Place of Supply)
            if any(k in b_text_lower for k in ["invoice no", "inv no", "invoice date", "due date", "po no", "place of supply"]):
                header_blocks.append(b)
                continue

            # B. Buyer / Bill To block
            if any(k in b_text_lower for k in ["bill to", "billed to", "buyer", "consignee", "customer:"]):
                buyer_blocks.append(b)
                continue

            # C. Totals block (containing amounts/taxes)
            if any(k in b_text_lower for k in [
                "taxable amount", "net taxable", "subtotal", "sub total",
                "cgst", "sgst", "igst", "grand total", "total rs", "round off"
            ]):
                totals_blocks.append(b)
                continue

            # D. Footer / Bank Details
            if any(k in b_text_lower for k in ["bank name", "ifsc", "account no", "a/c no", "branch", "terms & conditions", "authorized signatory"]):
                footer_blocks.append(b)
                continue

            # E. Position-based assignment
            if bn[3] <= 320:
                if bn[0] < 500:
                    vendor_blocks.append(b)
                else:
                    header_blocks.append(b)
            elif bn[1] < table_top_y:
                if bn[0] < 550:
                    buyer_blocks.append(b)
                else:
                    header_blocks.append(b)
            elif bn[1] >= table_bot_y:
                if bn[0] >= 400:
                    totals_blocks.append(b)
                else:
                    footer_blocks.append(b)

        # ── 4. Extract Vendor Name via Font Hierarchy ──────────────────────
        # Inspect all spans in top 280 normalized vertical space (page 1)
        candidate_vendor_spans: list[tuple[float, NativeSpan]] = []
        for b in blocks:
            if b.bbox_norm[1] > 280:
                continue
            for l in b.lines:
                for s in l.spans:
                    clean_s = s.text.strip()
                    clean_lower = clean_s.lower()
                    if len(clean_s) < 3:
                        continue
                    if clean_lower in GENERIC_HEADER_TERMS:
                        continue
                    if any(clean_lower.startswith(g) for g in ["gstin", "pan", "phone", "email", "url:", "page ", "issue date"]):
                        continue
                    # Weight score: font size + bold bonus - distance from top left
                    score = s.size + (3.0 if s.is_bold else 0.0)
                    if s.bbox_norm[0] < 500:
                        score += 2.0  # Vendor names are typically left/top-aligned
                    candidate_vendor_spans.append((score, s))

        if candidate_vendor_spans:
            candidate_vendor_spans.sort(key=lambda x: x[0], reverse=True)
            best_score, best_span = candidate_vendor_spans[0]
            vendor_name_candidate = best_span.text.strip()
            # Clean common artifacts
            vendor_name_candidate = re.sub(r"^[:\s\-#*]+", "", vendor_name_candidate)
            extracted_fields["vendor_name"] = {
                "value": vendor_name_candidate,
                "bbox_norm": best_span.bbox_norm,
                "confidence": 0.99,
                "source": "native_vector_font_hierarchy",
                "selection_reason": f"Dominant vector font ({best_span.font} {best_span.size}pt) in header zone",
            }

        # ── 5. Extract Buyer Name Candidate ────────────────────────────────
        for b in buyer_blocks:
            lines = [l.text.strip() for l in b.lines if l.text.strip()]
            for idx, line in enumerate(lines):
                if re.match(r"^(?:bill\s*to|billed\s*to|buyer|consignee|customer)[:\s]*$", line, re.I):
                    if idx + 1 < len(lines):
                        cand = lines[idx + 1]
                        cand = re.sub(r"^(?:m/s|mr\.|ms\.|sri|smt)?\s*", "", cand, flags=re.I).strip()
                        buyer_name_candidate = cand
                        extracted_fields["buyer_name"] = {
                            "value": buyer_name_candidate,
                            "bbox_norm": b.lines[idx + 1].bbox_norm,
                            "confidence": 0.98,
                            "source": "native_vector_buyer_block",
                            "selection_reason": "First clean line following 'Bill To' vector block",
                        }
                        break
            if buyer_name_candidate:
                break

        if not buyer_name_candidate:
            m = re.search(r"(?:bill\s*to|billed\s*to|buyer|consignee|customer)[:\s]*\n\s*([^\n]+)", full_text, re.I)
            if m:
                cand = re.sub(r"^(?:m/s|mr\.|ms\.|sri|smt)?\s*", "", m.group(1), flags=re.I).strip()
                if cand and not any(cand.lower().startswith(g) for g in ["gstin", "pan", "phone", "address"]):
                    buyer_name_candidate = cand
                    matched_line = next((l for l in all_lines if cand in l.text or l.text in cand), None)
                    extracted_fields["buyer_name"] = {
                        "value": buyer_name_candidate,
                        "bbox_norm": matched_line.bbox_norm if matched_line else [0, 0, 0, 0],
                        "confidence": 0.98,
                        "source": "native_vector_buyer_block",
                        "selection_reason": "First clean line following 'Bill To' vector block",
                    }

        # ── 6. Extract Deterministic Header Fields (Inv No, Date, etc.) ────
        for k, pattern in ANCHOR_REGEXES.items():
            if k in extracted_fields:
                continue
            m = pattern.search(full_text)
            if m:
                val = (m.group(1) if m.lastindex and m.lastindex >= 1 else m.group(0)).strip()
                # Find matching token for bbox
                matched_token = next((w for w in word_tokens if val in w.text or w.text in val), None)
                extracted_fields[k] = {
                    "value": val,
                    "bbox_norm": matched_token.bbox_norm if matched_token else [0, 0, 0, 0],
                    "confidence": 0.98,
                    "source": "native_vector_regex",
                    "selection_reason": f"Exact regex match on native vector text ({k})",
                }

        # ── 7. Extract Deterministic Totals ─────────────────────────────────
        search_lines = [l for b in totals_blocks for l in b.lines] + all_lines
        for line_obj in search_lines:
            line_clean = line_obj.text.strip()
            # Grand Total
            if re.search(r"\bgrand\s*total\b", line_clean, re.I) and "grand_total" not in extracted_fields:
                num_m = re.findall(r"[\d,]+(?:\.\d{2})?", line_clean)
                if num_m:
                    extracted_fields["grand_total"] = {
                        "value": num_m[-1].replace(",", ""),
                        "bbox_norm": line_obj.bbox_norm,
                        "confidence": 0.98,
                        "source": "native_vector_totals",
                        "selection_reason": f"Extracted from native vector Grand Total line: '{line_clean}'",
                    }
            # Taxable / Subtotal
            elif re.search(r"\b(?:taxable\s*amount|net\s*taxable|sub\s*total)\b", line_clean, re.I) and "subtotal" not in extracted_fields:
                num_m = re.findall(r"[\d,]+(?:\.\d{2})?", line_clean)
                if num_m:
                    extracted_fields["subtotal"] = {
                        "value": num_m[-1].replace(",", ""),
                        "bbox_norm": line_obj.bbox_norm,
                        "confidence": 0.98,
                        "source": "native_vector_totals",
                        "selection_reason": f"Extracted from native vector subtotal line: '{line_clean}'",
                    }
            # CGST
            elif re.search(r"\bcgst\b", line_clean, re.I) and "cgst" not in extracted_fields:
                num_m = re.findall(r"[\d,]+(?:\.\d{2})?", line_clean)
                if num_m:
                    extracted_fields["cgst"] = {
                        "value": num_m[-1].replace(",", ""),
                        "bbox_norm": line_obj.bbox_norm,
                        "confidence": 0.98,
                        "source": "native_vector_totals",
                        "selection_reason": f"Extracted from native vector CGST line: '{line_clean}'",
                    }
            # SGST
            elif re.search(r"\bsgst\b", line_clean, re.I) and "sgst" not in extracted_fields:
                num_m = re.findall(r"[\d,]+(?:\.\d{2})?", line_clean)
                if num_m:
                    extracted_fields["sgst"] = {
                        "value": num_m[-1].replace(",", ""),
                        "bbox_norm": line_obj.bbox_norm,
                        "confidence": 0.98,
                        "source": "native_vector_totals",
                        "selection_reason": f"Extracted from native vector SGST line: '{line_clean}'",
                    }

        # ── 8. Assemble Authoritative Vector RegionBlocks ───────────────────
        def merge_blocks_to_region(blist: list[NativeBlock], label: str) -> Optional[RegionBlock]:
            if not blist:
                return None
            min_x = min(b.bbox_norm[0] for b in blist)
            min_y = min(b.bbox_norm[1] for b in blist)
            max_x = max(b.bbox_norm[2] for b in blist)
            max_y = max(b.bbox_norm[3] for b in blist)
            raw_min_x = min(b.bbox[0] for b in blist)
            raw_min_y = min(b.bbox[1] for b in blist)
            raw_max_x = max(b.bbox[2] for b in blist)
            raw_max_y = max(b.bbox[3] for b in blist)
            return RegionBlock(
                label=label,
                bbox_norm=[min_x, min_y, max_x, max_y],
                bbox_raw=[raw_min_x, raw_min_y, raw_max_x, raw_max_y],
                confidence=0.98,
                page=page_num,
            )

        v_reg = merge_blocks_to_region(vendor_blocks, "vendor_block")
        if v_reg:
            regions.append(v_reg)

        h_reg = merge_blocks_to_region(header_blocks, "header")
        if h_reg:
            regions.append(h_reg)

        b_reg = merge_blocks_to_region(buyer_blocks, "buyer_block")
        if b_reg:
            regions.append(b_reg)

        t_reg = merge_blocks_to_region(totals_blocks, "totals")
        if t_reg:
            regions.append(t_reg)

        f_reg = merge_blocks_to_region(footer_blocks, "footer")
        if f_reg:
            regions.append(f_reg)

        return NativeVectorPage(
            page_num=page_num,
            page_width=pw,
            page_height=ph,
            full_text=full_text,
            blocks=blocks,
            lines=all_lines,
            words=word_tokens,
            regions=regions,
            table_items=table_items,
            table_bbox=table_bbox,
            extracted_fields=extracted_fields,
            vendor_name_candidate=vendor_name_candidate,
            buyer_name_candidate=buyer_name_candidate,
        )
