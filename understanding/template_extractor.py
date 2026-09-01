"""
understanding/template_extractor.py

TIE (Template-based Information Extraction) Field Extraction Engine.

Extracts structured invoice fields directly from a matched TemplateVersion and DocumentProfile
without calling heavy neural models (LayoutLM / VLM / LLM).

Supported Field Strategies:
1. anchor_relative: Locates anchor token(s) and extracts words within a relative offset bounding box.
2. regex_pattern: Applies high-precision regexes (GSTIN, PAN, IFSC, Phone, Email, Date).
3. semantic_numeric: Extracts and ranks currency/numeric tokens with arithmetic verification.
4. spatial_table: Reconstructs table line items from spatial token columns.
5. text_region: Extracts multiline text within a normalized macro-region (vendor/buyer block).
"""

from __future__ import annotations
import re
import difflib
from dataclasses import dataclass, field
from typing import Optional, Any, Sequence
from loguru import logger

from preprocessing.document_profile import DocumentProfile, WordToken, RegionBlock
from understanding.layoutlm import ExtractedInvoice, ExtractedField, SpatialCandidate


# Common Indian Invoice Regex Patterns
GSTIN_RE = re.compile(r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}\b")
PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b")
IFSC_RE = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_RE = re.compile(r"(?:\+91[\s-]?)?[6-9]\d{9}\b")
DATE_RE = re.compile(
    r"\b(?:\d{1,2}[-/.](?:\d{1,2}|[A-Za-z]{3})[-/.]\d{2,4}|\d{4}[-/.](?:\d{1,2}|[A-Za-z]{3})[-/.]\d{1,2})\b"
)
NUMERIC_RE = re.compile(r"[-+]?[0-9]{1,3}(?:,[0-9]{2,3})*(?:\.[0-9]{1,2})?")


def clean_currency_str(val: str) -> Optional[float]:
    """Parse float from string e.g. 'Rs. 1,560.00' -> 1560.00"""
    if not val:
        return None
    s = str(val).strip()
    if "%" in s:
        return None
    s = s.replace("₹", "").replace("Rs.", "").replace("INR", "").replace(",", "").strip()
    # Remove surrounding parens if negative/discount
    is_neg = s.startswith("(") and s.endswith(")")
    s = s.replace("(", "").replace(")", "").strip()
    m = re.search(r"[-+]?\d*\.?\d+", s)
    if m:
        try:
            num = float(m.group(0))
            return -num if is_neg else num
        except ValueError:
            return None
    return None


MONTH_MAP = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06",
    "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12"
}


def clean_vendor_name(val: str) -> str:
    """Isolates clean vendor business name from header blocks."""
    if not val:
        return ""
    lines = [l.strip() for l in str(val).split("\n") if l.strip()]
    s = lines[0] if lines else str(val)
    s = re.sub(r"^(?:retail|tax|tax invoice|invoice|bill of supply|original for recipient|cash memo)\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^[:\s\-#*]+", "", s)
    s = re.split(r"\s+(?:plot\s*no|sector|road|street|sarani|lane|nagar|opp|near|floor|building|behind|kolkata|mumbai|delhi|bengaluru|west bengal|pin|ph|phone|gstin|email|cin|\d+/\d+)\b", s, flags=re.IGNORECASE)[0]
    return s.strip(":.,- ")


def clean_buyer_name(val: str) -> str:
    """Isolates clean buyer/client name from billing headers."""
    if not val:
        return ""
    lines = [l.strip() for l in str(val).split("\n") if l.strip()]
    lines = [l for l in lines if not re.match(r"^(?:bill\s*to|billed\s*to|buyer|consignee|customer)[:\s]*$", l, re.I)]
    s = lines[0] if lines else str(val)
    s = re.sub(r"^(?:bill\s*to|billed\s*to|buyer|consignee|customer|m/s|mr\.|ms\.|sri|smt)?\s*(?:name)?\s*[:.\-]?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^[:\s\-#*]+", "", s)
    s = re.split(r"\s+(?:invoice\s*no|invoice\s*date|sls\s*code|due\s*date|place\s*of\s*supply|category|sub\s*category|phone|mobile|gstin|pan|order|address|kolkata|mumbai|delhi)\b", s, flags=re.IGNORECASE)[0]
    return s.strip(":.,- ")


def clean_place_of_supply(val: str) -> str:
    """Isolates clean state/place of supply name."""
    if not val:
        return ""
    lines = [l.strip() for l in str(val).split("\n") if l.strip()]
    s = lines[0] if lines else str(val)
    s = re.sub(r"^(?:place\s*of\s*supply|state/ut\s*code|pos|state\s*code|supply\s*state)\s*[:.\-]?\s*", "", s, flags=re.IGNORECASE)
    s = s.strip(":.,- ")
    s = re.split(r"\s+(?:category|sub\s*category|phone|mobile|buyer|invoice|due|sls|ftoagency|\(sch\d+\))\b", s, flags=re.IGNORECASE)[0]
    return s.strip(":.,- ")


def clean_amount_in_words(val: str) -> Optional[str]:
    """Cleans amount in words string, verifying it contains valid currency text."""
    if not val:
        return None
    s = re.sub(r"^(?:amount\s*in\s*words|in\s*words|amount\s*words|rupees\s*in\s*words)\s*[:.\-]?\s*", "", str(val), flags=re.IGNORECASE)
    s = s.strip(":.,- ")
    if s.lower() in ("taxable amount", "total amount", "net taxable", "grand total", "net amount", "amount"):
        return None
    num_keywords = ["rupees", "only", "thousand", "hundred", "lakh", "crore", "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "twenty", "thirty", "forty", "fifty"]
    if not any(kw in s.lower() for kw in num_keywords):
        return None
    return s


def clean_invoice_number(val: str) -> str:
    """Cleans invoice number prefix words and stops before dates or other field anchors."""
    s = re.sub(r"^(?:invoice|inv|bill|voucher|challan)\s*(?:no\.?|#|num|number)?\s*[:.\-]?\s*", "", val, flags=re.IGNORECASE)
    s = re.sub(r"^[:\s\-#]+", "", s)
    # Stop before date patterns
    s = re.split(r"\s+(?:\d{1,2}[-/.](?:\d{1,2}|[A-Za-z]{3})[-/.]\d{2,4}|\d{4}[-/.])", s)[0]
    # Stop before subsequent field anchors
    s = re.split(r"\s+(?:date|dated|due|gstin|pan|po|place|to|bill|amount|total|customer|e-com|order|generated|cashier|name|phone|mobile)\b", s, flags=re.IGNORECASE)[0]
    tokens = [t.strip(":.,- ") for t in s.split() if t.strip(":.,- ")]
    if tokens:
        first = tokens[0]
        if len(first) >= 2 and any(c.isdigit() or c.isalpha() for c in first):
            return first
    return s.strip(":.,- ")


def clean_date_str(val: str) -> str:
    """Extract standard date string."""
    if not val:
        return ""
    s = val.strip()
    # Check for DD-Mon-YYYY or DD Mon YYYY e.g. 10-Aug-2026
    m_alpha = re.search(r"\b(\d{1,2})[-/\s]([A-Za-z]{3,9})[-/\s](\d{2,4})\b", s)
    if m_alpha:
        d = m_alpha.group(1).zfill(2)
        m_name = m_alpha.group(2).lower()[:3]
        m = MONTH_MAP.get(m_name, "01")
        y = m_alpha.group(3)
        if len(y) == 2:
            y = f"20{y}"
        return f"{d}/{m}/{y}"
    
    # Check standard numeric DD/MM/YYYY or DD-MM-YYYY
    m_num = DATE_RE.search(s)
    if m_num:
        return m_num.group(0).strip()
    return s


@dataclass
class FieldExtractionResult:
    field_name: str
    value: Optional[str]
    confidence: float
    strategy_used: str
    bbox: Optional[list[int]] = None
    raw_tokens: list[WordToken] = field(default_factory=list)


class TemplateExtractor:
    """
    TIE Field Extraction Engine executing rule strategies on a DocumentProfile.
    """

    def __init__(self):
        pass

    def extract(
        self,
        profile: DocumentProfile,
        field_rules: list[Any],
        template_version_id: Optional[str] = None,
        match_type: str = "exact_version",
    ) -> ExtractedInvoice:
        """
        Executes field extraction rules against DocumentProfile.
        Distinguishes between exact_version (deterministic rules) and family_anchor (adaptive candidate search).
        """
        extracted = ExtractedInvoice()
        field_confs: dict[str, float] = {}

        # Default rules if empty
        if not field_rules:
            field_rules = self._get_default_rules()

        is_family_match = (match_type == "family_anchor")

        for rule in field_rules:
            if isinstance(rule, dict):
                f_name = rule["field_name"]
                strat = rule.get("strategy", "anchor_relative")
                anchors = rule.get("anchors", [])
                search_reg = rule.get("search_region")
                rel_box = rule.get("relative_box") if not is_family_match else None  # Invalidate rigid offsets for family matches
                parser_spec = rule.get("parser_spec") or {}
                base_conf = rule.get("confidence_score", 0.95)
            else:
                f_name = rule.field_name
                strat = rule.strategy
                anchors = rule.anchors or []
                search_reg = rule.search_region
                rel_box = rule.relative_box if not is_family_match else None
                parser_spec = rule.parser_spec or {}
                base_conf = getattr(rule, "confidence_score", 0.95)

            if is_family_match:
                base_conf = min(0.90, base_conf * 0.92)

            res = self._extract_field(
                profile=profile,
                field_name=f_name,
                strategy=strat,
                anchors=anchors,
                search_region=search_reg,
                relative_box=rel_box,
                parser_spec=parser_spec,
                base_confidence=base_conf,
            )

            if res and res.value:
                ext_field = ExtractedField(
                    value=str(res.value),
                    confidence=res.confidence,
                    source="tie_template" if not is_family_match else "tie_family_anchor",
                )
                setattr(extracted, f_name, ext_field)
                field_confs[f_name] = res.confidence

                # Record spatial candidate evidence
                if res.bbox:
                    extracted.spatial_candidates.append(
                        SpatialCandidate(
                            field_name=f_name,
                            value=str(res.value),
                            bbox=res.bbox,
                            confidence=res.confidence,
                            region="template_rule",
                        )
                    )

        # Reconstruct line items if not extracted by rules
        if not extracted.line_items:
            extracted.line_items = self._reconstruct_table_items(profile)

        # Cross-field tax reconciliation: CGST == SGST when total tax == 2 * CGST
        if extracted.cgst and extracted.cgst.value and extracted.tax_amount and extracted.tax_amount.value:
            try:
                c_val = float(extracted.cgst.value)
                t_val = float(extracted.tax_amount.value)
                if c_val > 0 and abs((c_val * 2) - t_val) <= 0.05:
                    extracted.sgst = ExtractedField(value=f"{c_val:.2f}", confidence=0.98, source="tie_template")
                    field_confs["sgst"] = 0.98
            except (ValueError, TypeError):
                pass

        # Calibrate overall confidence
        weights = {
            "grand_total": 0.30,
            "invoice_number": 0.25,
            "invoice_date": 0.20,
            "vendor_gstin": 0.15,
            "subtotal": 0.10,
        }
        total_w = 0.0
        weighted_conf = 0.0
        for f, w in weights.items():
            if f in field_confs:
                c = field_confs[f]
                weighted_conf += c * w
                total_w += w

        extracted.overall_confidence = round(weighted_conf / total_w, 3) if total_w > 0 else 0.85
        return extracted

    def _extract_field(
        self,
        profile: DocumentProfile,
        field_name: str,
        strategy: str,
        anchors: list[str],
        search_region: Optional[list[int]],
        relative_box: Optional[list[int]],
        parser_spec: dict[str, Any],
        base_confidence: float,
    ) -> Optional[FieldExtractionResult]:
        """Dispatches to the appropriate extraction strategy."""
        if strategy == "anchor_relative":
            return self._extract_anchor_relative(
                profile, field_name, anchors, relative_box, search_region, parser_spec, base_confidence
            )
        elif strategy == "regex_pattern":
            return self._extract_regex(
                profile, field_name, parser_spec.get("pattern"), search_region, base_confidence
            )
        elif strategy == "semantic_numeric":
            return self._extract_semantic_numeric(
                profile, field_name, anchors, search_region, base_confidence
            )
        elif strategy == "text_region":
            return self._extract_text_region(
                profile, field_name, search_region, parser_spec, base_confidence
            )
        return None

    def _extract_anchor_relative(
        self,
        profile: DocumentProfile,
        field_name: str,
        anchors: list[str],
        relative_box: Optional[list[int]],
        search_region: Optional[list[int]],
        parser_spec: dict[str, Any],
        base_confidence: float,
    ) -> Optional[FieldExtractionResult]:
        """
        Extracts a field value located relative to a discovered text anchor token.
        Scores candidate anchor occurrences based on expected page layout.
        """
        # 1. Find and rank all anchor matches
        matched_anchor_candidates: list[tuple[float, list[WordToken]]] = []
        for anc in anchors:
            matches = profile.find_anchor_tokens(anc)
            for m in matches:
                if not m:
                    continue
                score = 1.0
                anc_y1 = min(w.bbox_norm[1] for w in m)
                anc_x1 = min(w.bbox_norm[0] for w in m)
                
                # Context scoring: header fields expect top 50%
                if field_name in ("invoice_number", "invoice_date", "vendor_name", "vendor_gstin", "buyer_name", "buyer_gstin"):
                    if anc_y1 <= 450:
                        score += 0.30
                    else:
                        score -= 0.20
                # Totals expect bottom 50%
                elif field_name in ("grand_total", "subtotal", "tax_amount", "cgst", "sgst", "igst", "discount", "round_off"):
                    if anc_y1 >= 400:
                        score += 0.30
                    else:
                        score -= 0.20

                # Buyer fields expect right or middle-left area
                if "buyer" in field_name:
                    if 150 <= anc_y1 <= 550:
                        score += 0.25

                matched_anchor_candidates.append((score, m))

        if not matched_anchor_candidates:
            # Fallback to search_region if provided
            if search_region:
                words = profile.find_words_in_box(search_region)
                if words:
                    val = " ".join(w.text for w in words).strip()
                    val = self._parse_field_value(field_name, val, parser_spec)
                    if val:
                        return FieldExtractionResult(
                            field_name=field_name,
                            value=val,
                            confidence=base_confidence * 0.85,
                            strategy_used="search_region_fallback",
                            bbox=search_region,
                            raw_tokens=words,
                        )
            return None

        # Pick best-scoring anchor occurrence
        matched_anchor_candidates.sort(key=lambda x: x[0], reverse=True)
        _, matched_anchor_seq = matched_anchor_candidates[0]

        # 2. Compute search box from anchor bounding box + relative offset
        anc_x1 = min(w.bbox_norm[0] for w in matched_anchor_seq)
        anc_y1 = min(w.bbox_norm[1] for w in matched_anchor_seq)
        anc_x2 = max(w.bbox_norm[2] for w in matched_anchor_seq)
        anc_y2 = max(w.bbox_norm[3] for w in matched_anchor_seq)
        anc_page = matched_anchor_seq[0].page

        # Default relative offset if not configured: to the right of anchor or directly below
        if relative_box and len(relative_box) == 4:
            dx1, dy1, dx2, dy2 = relative_box
            base_x = anc_x2 if dx1 >= 0 else anc_x1
            target_box = [
                max(0, min(1000, base_x + dx1)),
                max(0, min(1000, anc_y1 + dy1)),
                max(0, min(1000, anc_x2 + dx2)),
                max(0, min(1000, anc_y2 + dy2)),
            ]
        else:
            # Right side lookup default [0, -10, 350, 20]
            target_box = [
                anc_x2,
                max(0, anc_y1 - 10),
                min(1000, anc_x2 + 350),
                min(1000, anc_y2 + 15),
            ]

        # 3. Retrieve tokens in target box on the same page
        found_tokens = profile.find_words_in_box(target_box, page=anc_page)
        # Filter out the anchor tokens themselves
        anc_token_ids = {id(w) for w in matched_anchor_seq}
        target_tokens = [w for w in found_tokens if id(w) not in anc_token_ids]

        if not target_tokens:
            # Secondary attempt: check line immediately below anchor
            below_box = [
                anc_x1,
                anc_y2,
                min(1000, anc_x1 + 300),
                min(1000, anc_y2 + 35),
            ]
            below_tokens = [w for w in profile.find_words_in_box(below_box, page=anc_page) if id(w) not in anc_token_ids]
            if below_tokens:
                target_tokens = below_tokens
                target_box = below_box

        if not target_tokens:
            return None

        raw_val = " ".join(w.text for w in target_tokens).strip()
        parsed_val = self._parse_field_value(field_name, raw_val, parser_spec)
        if not parsed_val:
            return None

        avg_ocr_conf = sum(w.confidence for w in target_tokens) / max(1, len(target_tokens))
        conf = round(base_confidence * avg_ocr_conf, 3)

        return FieldExtractionResult(
            field_name=field_name,
            value=parsed_val,
            confidence=conf,
            strategy_used="anchor_relative",
            bbox=target_box,
            raw_tokens=target_tokens,
        )

    def _extract_regex(
        self,
        profile: DocumentProfile,
        field_name: str,
        custom_pattern: Optional[str],
        search_region: Optional[list[int]],
        base_confidence: float,
    ) -> Optional[FieldExtractionResult]:
        """Extracts fields matching regular expressions using field-aware candidate ranking."""
        pat = None
        if custom_pattern:
            pat = re.compile(custom_pattern)
        elif field_name in ("vendor_gstin", "buyer_gstin"):
            pat = GSTIN_RE
        elif field_name == "vendor_pan":
            pat = PAN_RE
        elif field_name == "ifsc_code":
            pat = IFSC_RE
        elif field_name == "vendor_email":
            pat = EMAIL_RE
        elif field_name in ("vendor_phone", "buyer_phone"):
            pat = PHONE_RE
        elif "date" in field_name:
            pat = DATE_RE

        if not pat:
            return None

        scope_words = profile.words
        if search_region:
            scope_words = profile.find_words_in_box(search_region)

        # Collect all pattern matches with spatial ranking
        candidates: list[tuple[float, str, WordToken]] = []

        for w in scope_words:
            m = pat.search(w.text)
            if not m:
                continue

            val = m.group(0).strip()
            score = 1.0 * w.confidence

            # ── GSTIN Field-Aware Candidate Ranking ───────────────────────────
            if field_name == "vendor_gstin":
                # Vendor GSTIN is expected in top 40% of page 1
                if w.page == 1 and w.bbox_norm[1] <= 400:
                    score += 0.35
                elif w.bbox_norm[1] > 600:
                    score -= 0.30

                # Check preceding lines / spatial context
                nearby_words = profile.find_words_in_box(
                    [max(0, w.bbox_norm[0] - 250), max(0, w.bbox_norm[1] - 30), w.bbox_norm[2], w.bbox_norm[3] + 10],
                    page=w.page
                )
                nearby_text = " ".join(nw.text.lower() for nw in nearby_words)
                if any(k in nearby_text for k in ("seller", "vendor", "supplier", "from", "sold by")):
                    score += 0.40
                if any(k in nearby_text for k in ("bill to", "buyer", "ship to", "consignee", "recipient")):
                    score -= 0.60  # Definitely buyer GSTIN

            elif field_name == "buyer_gstin":
                # Buyer GSTIN is expected near "Bill To" / "Buyer"
                nearby_words = profile.find_words_in_box(
                    [max(0, w.bbox_norm[0] - 250), max(0, w.bbox_norm[1] - 30), w.bbox_norm[2], w.bbox_norm[3] + 10],
                    page=w.page
                )
                nearby_text = " ".join(nw.text.lower() for nw in nearby_words)
                if any(k in nearby_text for k in ("bill to", "buyer", "ship to", "consignee", "recipient")):
                    score += 0.50
                if any(k in nearby_text for k in ("seller", "vendor", "supplier", "from")):
                    score -= 0.50

            # ── Date Field-Aware Candidate Ranking ───────────────────────────
            elif field_name == "invoice_date":
                nearby_words = profile.find_words_in_box(
                    [max(0, w.bbox_norm[0] - 200), max(0, w.bbox_norm[1] - 15), w.bbox_norm[2], w.bbox_norm[3] + 15],
                    page=w.page
                )
                nearby_text = " ".join(nw.text.lower() for nw in nearby_words)
                if any(k in nearby_text for k in ("inv date", "invoice date", "bill date", "dated", "date of issue")):
                    score += 0.50
                elif "due date" in nearby_text or "po date" in nearby_text or "delivery" in nearby_text:
                    score -= 0.40

            candidates.append((score, val, w))

        if candidates:
            candidates.sort(key=lambda c: c[0], reverse=True)
            best_score, best_val, best_token = candidates[0]

            calibrated_conf = base_confidence * best_token.confidence
            if len(candidates) > 1:
                second_score, second_val, _ = candidates[1]
                if second_val.upper() != best_val.upper():
                    margin = max(0.0, best_score - second_score)
                    if margin < 0.20:
                        # Ambiguous match (e.g. 2 competing candidates with low score separation)
                        calibrated_conf *= (0.70 + 1.5 * margin)

            return FieldExtractionResult(
                field_name=field_name,
                value=best_val,
                confidence=min(0.99, round(calibrated_conf, 3)),
                strategy_used="regex_ranked",
                bbox=best_token.bbox_norm,
                raw_tokens=[best_token],
            )

        # Check full text if tokenization split regex
        full_text = " ".join(w.text for w in scope_words)
        m = pat.search(full_text)
        if m:
            val = m.group(0).strip()
            return FieldExtractionResult(
                field_name=field_name,
                value=val,
                confidence=round(base_confidence * 0.90, 3),
                strategy_used="regex_full_text",
            )
        return None

    def _extract_semantic_numeric(
        self,
        profile: DocumentProfile,
        field_name: str,
        anchors: list[str],
        search_region: Optional[list[int]],
        base_confidence: float,
    ) -> Optional[FieldExtractionResult]:
        """
        Extracts financial amounts using candidate generation and multi-factor ranking.
        Scores distance, horizontal baseline alignment, currency prefix, format, and semantic type.
        """
        candidates: list[tuple[float, float, WordToken]] = []  # (rank_score, float_val, token)

        # 1. Search near each anchor
        for anc in anchors:
            matches = profile.find_anchor_tokens(anc)
            if not matches:
                continue

            for anchor_seq in matches:
                anc_x1 = min(w.bbox_norm[0] for w in anchor_seq)
                anc_y1 = min(w.bbox_norm[1] for w in anchor_seq)
                anc_x2 = max(w.bbox_norm[2] for w in anchor_seq)
                anc_y2 = max(w.bbox_norm[3] for w in anchor_seq)
                anc_cy = sum(w.center_norm[1] for w in anchor_seq) / len(anchor_seq)

                # Search horizontal strip to the right (inline)
                strip_box = [anc_x2, max(0, anc_y1 - 12), 1000, min(1000, anc_y2 + 15)]
                inline_words = profile.find_words_in_box(strip_box)

                # Also search below anchor (for vertical key-value pairs)
                below_box = [max(0, anc_x1 - 20), anc_y2, min(1000, anc_x2 + 250), min(1000, anc_y2 + 35)]
                below_words = profile.find_words_in_box(below_box)

                for w, is_inline in [(w, True) for w in inline_words] + [(w, False) for w in below_words]:
                    # Exclude anchor tokens
                    if any(id(w) == id(aw) for aw in anchor_seq):
                        continue

                    clean_f = clean_currency_str(w.text)
                    if clean_f is None or clean_f < 0:
                        continue

                    # Filter out HSN/SAC numbers (>= 100,000) for tax/discount components
                    if field_name in ("cgst", "sgst", "igst", "tax_amount", "discount", "round_off") and clean_f >= 100000.0:
                        continue

                    # Multi-Factor Score Calculation
                    score = 1.0

                    # A. Distance factor (closer is better)
                    dist = (w.bbox_norm[0] - anc_x2) if is_inline else (w.bbox_norm[1] - anc_y2)
                    dist_score = max(0.0, 1.0 - (max(0, dist) / 500.0))
                    score += dist_score * 0.40

                    # B. Baseline alignment factor
                    if is_inline:
                        y_diff = abs(w.center_norm[1] - anc_cy)
                        if y_diff <= 10:
                            score += 0.35
                    else:
                        score += 0.20

                    # C. Formatting factor (explicit decimal places .00, commas)
                    if "." in w.text:
                        score += 0.25
                    if "," in w.text or "rs" in w.text.lower() or "₹" in w.text:
                        score += 0.20

                    # D. Confidence of OCR token
                    score *= w.confidence

                    # E. Financial Plausibility Factor
                    if field_name == "grand_total":
                        if clean_f >= 1.0:
                            score += 0.15
                    elif field_name in ("cgst", "sgst", "igst", "tax_amount", "discount"):
                        if clean_f > 50000.0:
                            score -= 0.30

                    candidates.append((score, clean_f, w))

        # 2. Search fallback region
        if not candidates and search_region:
            reg_words = profile.find_words_in_box(search_region)
            for w in reg_words:
                clean_f = clean_currency_str(w.text)
                if clean_f is not None and clean_f > 0:
                    candidates.append((0.50 * w.confidence, clean_f, w))

        if not candidates:
            return None

        # Sort by multi-factor rank score descending
        candidates.sort(key=lambda c: c[0], reverse=True)
        best_score, best_val, best_token = candidates[0]

        calibrated_conf = base_confidence * best_token.confidence
        if len(candidates) > 1:
            second_score, second_val, _ = candidates[1]
            if abs(best_val - second_val) > 0.05:  # Distinct numeric amounts
                margin = max(0.0, best_score - second_score)
                if margin < 0.20:
                    # Low separation between competing financial amounts -> penalize confidence to trigger review
                    calibrated_conf *= (0.65 + 1.75 * margin)

        return FieldExtractionResult(
            field_name=field_name,
            value=f"{best_val:.2f}",
            confidence=min(0.99, round(calibrated_conf, 3)),
            strategy_used="semantic_numeric_ranked",
            bbox=best_token.bbox_norm,
            raw_tokens=[best_token],
        )

    def _extract_text_region(
        self,
        profile: DocumentProfile,
        field_name: str,
        search_region: Optional[list[int]],
        parser_spec: dict[str, Any],
        base_confidence: float,
    ) -> Optional[FieldExtractionResult]:
        """Extracts multiline entity text inside a bounding box."""
        if not search_region:
            return None

        words = profile.find_words_in_box(search_region)
        if not words:
            return None

        # Exclude anchor phrases if defined
        exclude_phrases = parser_spec.get("exclude_phrases", ["bill to", "ship to", "gstin", "pan", "phone"])
        filtered_words = []
        for w in words:
            w_clean = re.sub(r"[^a-z0-9]", "", w.text.lower())
            if not any(exc in w_clean for exc in exclude_phrases):
                filtered_words.append(w)

        if not filtered_words:
            return None

        raw_text = " ".join(w.text for w in filtered_words).strip()
        return FieldExtractionResult(
            field_name=field_name,
            value=raw_text,
            confidence=base_confidence * 0.90,
            strategy_used="text_region",
            bbox=search_region,
            raw_tokens=filtered_words,
        )

    def _parse_field_value(self, field_name: str, val: str, parser_spec: dict[str, Any]) -> Optional[str]:
        """Parses and sanitizes raw text into the appropriate field format."""
        if not val:
            return None

        if "amount_in_words" in field_name:
            return clean_amount_in_words(val)
        elif "amount" in field_name or "total" in field_name or field_name in ("subtotal", "cgst", "sgst", "igst", "discount", "round_off"):
            f_val = clean_currency_str(val)
            return f"{f_val:.2f}" if f_val is not None else None
        elif field_name == "invoice_number":
            return clean_invoice_number(val)
        elif "date" in field_name:
            return clean_date_str(val)
        elif field_name == "buyer_name":
            return clean_buyer_name(val)
        elif field_name == "vendor_name":
            return clean_vendor_name(val)
        elif field_name == "place_of_supply":
            return clean_place_of_supply(val)
        elif "gstin" in field_name:
            m = GSTIN_RE.search(val)
            return m.group(0) if m else val.strip().upper()
        elif "pan" in field_name:
            m = PAN_RE.search(val)
            return m.group(0) if m else val.strip().upper()
        elif "ifsc" in field_name:
            m = IFSC_RE.search(val)
            return m.group(0) if m else val.strip().upper()

        return val.strip()

    def _reconstruct_table_items(self, profile: DocumentProfile) -> list[dict[str, Any]]:
        """
        Spatial OCR reconstruction of line items table from word tokens.
        """
        # Find table header anchor keywords (Description, Qty, Rate, Amount)
        header_anchors = ["description", "particulars", "item", "qty", "quantity", "rate", "price", "amount", "taxable"]
        header_tokens = []
        for anc in header_anchors:
            m = profile.find_anchor_tokens(anc)
            if m:
                header_tokens.extend(m[0])

        if not header_tokens:
            return []

        header_y_min = min(w.bbox_norm[1] for w in header_tokens)
        header_y_max = max(w.bbox_norm[3] for w in header_tokens)

        # Find bottom boundary anchor (subtotal, total, grand total)
        total_anchors = ["subtotal", "total", "grand total", "total amount", "taxable value"]
        bottom_y = 950
        for anc in total_anchors:
            m = profile.find_anchor_tokens(anc)
            if m:
                anc_y = min(w.bbox_norm[1] for w in m[0])
                if anc_y > header_y_max:
                    bottom_y = min(bottom_y, anc_y - 10)

        # Find words within table vertical span
        table_words = [
            w for w in profile.words
            if header_y_max + 5 <= w.bbox_norm[1] <= bottom_y
        ]
        if not table_words:
            return []

        # Group words into line rows by vertical proximity
        rows: dict[int, list[WordToken]] = {}
        for w in sorted(table_words, key=lambda t: t.bbox_norm[1]):
            cy = int(w.center_norm[1])
            # Snap to 15-unit grid row
            grid_row = round(cy / 15.0) * 15
            if grid_row not in rows:
                rows[grid_row] = []
            rows[grid_row].append(w)

        items = []
        for r_y, r_words in sorted(rows.items()):
            row_tokens = sorted(r_words, key=lambda t: t.bbox_norm[0])
            desc_parts = []
            amounts = []

            for t in row_tokens:
                c_num = clean_currency_str(t.text)
                # Filter out HSN codes and require decimal or right-half column position for monetary values
                if c_num is not None and 0 < c_num < 100000.0 and (t.bbox_norm[0] >= 400 or "." in t.text):
                    amounts.append((c_num, t.bbox_norm[0]))
                else:
                    clean_w = re.sub(r"[^a-zA-Z0-9\s]", "", t.text).strip()
                    if clean_w:
                        desc_parts.append(clean_w)

            desc = " ".join(desc_parts).strip()
            desc_lower = desc.lower()
            if any(kw in desc_lower for kw in ["sub total", "subtotal", "+cgst", "+sgst", "+igst", "cgst", "sgst", "igst", "gst total", "net total", "net bill", "total bill", "tender", "tax summary"]):
                continue

            if desc and amounts:
                amounts_sorted = sorted(amounts, key=lambda a: a[1])
                final_amount = amounts_sorted[-1][0]
                rate = amounts_sorted[-2][0] if len(amounts_sorted) >= 2 else final_amount
                qty = amounts_sorted[0][0] if len(amounts_sorted) >= 3 and amounts_sorted[0][0] <= 100 else 1.0
                items.append({
                    "description": desc,
                    "quantity": qty,
                    "rate": rate,
                    "amount": final_amount,
                    "taxable_value": final_amount,
                })

        return items

    @staticmethod
    def _get_default_rules() -> list[dict[str, Any]]:
        """Canonical default field rules for standard Indian Tax Invoices."""
        return [
            {
                "field_name": "invoice_number",
                "strategy": "anchor_relative",
                "anchors": ["invoice no", "inv no", "bill no", "invoice #", "voucher no"],
                "relative_box": [0, -5, 300, 10],
                "confidence_score": 0.95,
            },
            {
                "field_name": "invoice_date",
                "strategy": "anchor_relative",
                "anchors": ["invoice date", "dated", "bill date", "date of issue", "date:"],
                "relative_box": [0, -5, 250, 10],
                "confidence_score": 0.95,
            },
            {
                "field_name": "due_date",
                "strategy": "anchor_relative",
                "anchors": ["due date", "payment due", "pay by"],
                "relative_box": [0, -5, 250, 10],
                "confidence_score": 0.90,
            },
            {
                "field_name": "vendor_gstin",
                "strategy": "regex_pattern",
                "anchors": ["gstin", "gst no", "vendor gstin"],
                "confidence_score": 0.98,
            },
            {
                "field_name": "vendor_pan",
                "strategy": "regex_pattern",
                "anchors": ["pan", "pan no"],
                "confidence_score": 0.95,
            },
            {
                "field_name": "vendor_name",
                "strategy": "text_region",
                "search_region": [0, 0, 600, 200],
                "parser_spec": {"exclude_phrases": ["tax invoice", "invoice", "gstin", "pan"]},
                "confidence_score": 0.90,
            },
            {
                "field_name": "buyer_name",
                "strategy": "anchor_relative",
                "anchors": ["bill to", "billed to", "consignee", "buyer"],
                "relative_box": [0, 10, 400, 80],
                "confidence_score": 0.90,
            },
            {
                "field_name": "place_of_supply",
                "strategy": "anchor_relative",
                "anchors": ["place of supply", "state/ut code", "supply state"],
                "relative_box": [0, -5, 250, 10],
                "confidence_score": 0.90,
            },
            {
                "field_name": "subtotal",
                "strategy": "semantic_numeric",
                "anchors": ["sub total", "subtotal", "taxable value", "taxable amount", "total before tax"],
                "confidence_score": 0.95,
            },
            {
                "field_name": "cgst",
                "strategy": "semantic_numeric",
                "anchors": ["cgst", "central tax", "cgst amt"],
                "confidence_score": 0.92,
            },
            {
                "field_name": "sgst",
                "strategy": "semantic_numeric",
                "anchors": ["sgst", "state tax", "sgst amt", "utgst"],
                "confidence_score": 0.92,
            },
            {
                "field_name": "igst",
                "strategy": "semantic_numeric",
                "anchors": ["igst", "integrated tax", "igst amt"],
                "confidence_score": 0.92,
            },
            {
                "field_name": "tax_amount",
                "strategy": "semantic_numeric",
                "anchors": ["total tax", "tax amount", "gst amount", "tax total"],
                "confidence_score": 0.92,
            },
            {
                "field_name": "discount",
                "strategy": "semantic_numeric",
                "anchors": ["discount", "less discount", "trade discount", "global discount"],
                "confidence_score": 0.90,
            },
            {
                "field_name": "round_off",
                "strategy": "semantic_numeric",
                "anchors": ["round off", "roundoff", "rounding"],
                "confidence_score": 0.90,
            },
            {
                "field_name": "grand_total",
                "strategy": "semantic_numeric",
                "anchors": ["grand total", "total payable", "net amount", "total amount", "total:", "net payable"],
                "confidence_score": 0.98,
            },
            {
                "field_name": "ifsc_code",
                "strategy": "regex_pattern",
                "anchors": ["ifsc", "ifsc code"],
                "confidence_score": 0.95,
            },
            {
                "field_name": "account_number",
                "strategy": "anchor_relative",
                "anchors": ["account no", "ac no", "a/c no", "bank a/c"],
                "relative_box": [0, -10, 300, 20],
                "confidence_score": 0.92,
            },
        ]
