"""
Stage 4a: LayoutLMv3 document understanding + Enhanced Heuristic Extraction Engine.

Uses Microsoft's LayoutLMv3 model (fine-tuned on invoices) to understand
the spatial relationship between text blocks and extract structured fields.

Falls back to comprehensive regex and geometric heuristic extraction across
commercial invoices, government vouchers, HRMS bill annexures, and challans.
"""

import re
import numpy as np
from pathlib import Path
from loguru import logger
from dataclasses import dataclass, field
from typing import Optional, Any


from ocr.extractor import OCRWord, decompose_line_into_words


@dataclass
class ExtractedField:
    value: str
    confidence: float
    source: str   # "native_pdf", "layoutlm", "heuristic", or "llm"


# Quality multiplier per source — used by _merge_invoices to pick the
# best value when multiple sources disagree.
SOURCE_WEIGHTS: dict[str, float] = {
    "tie_template":     1.10,   # deterministic template extraction — highest priority
    "tie_fast_path":    1.10,
    "native_pdf":       1.00,   # direct from PDF text layer
    "layoutlm":         0.85,   # model inference
    "heuristic":        0.65,   # regex fallback
    "heuristic_reconciled": 0.70,
    "paddleocr":        0.75,   # OCR on clean image
    "easyocr":          0.65,   # fallback OCR
    "llm":              0.60,   # LLM guess
    "llm_unavailable":  0.00,
    "llm_error":        0.00,
}

# Major Indian Bank IFSC 4-letter prefix mapping
IFSC_BANK_MAP: dict[str, str] = {
    "BARB": "Bank of Baroda",
    "SBIN": "State Bank of India",
    "PUNB": "Punjab National Bank",
    "HDFC": "HDFC Bank",
    "ICIC": "ICICI Bank",
    "UTIB": "Axis Bank",
    "CBIN": "Central Bank of India",
    "UBIN": "Union Bank of India",
    "CNRB": "Canara Bank",
    "IOBA": "Indian Overseas Bank",
    "IDIB": "Indian Bank",
    "BKID": "Bank of India",
    "YESB": "Yes Bank",
    "KKBK": "Kotak Mahindra Bank",
    "MAHB": "Bank of Maharashtra",
    "PSIB": "Punjab & Sind Bank",
    "UCOB": "UCO Bank",
    "BDBL": "Bandhan Bank",
    "FDRL": "Federal Bank",
    "IDFB": "IDFC First Bank",
    "INDB": "IndusInd Bank",
    "KVBL": "Karur Vysya Bank",
    "RATN": "RBL Bank",
    "SIBL": "South Indian Bank",
}


@dataclass
class SpatialCandidate:
    field_name: str
    value: str
    bbox: list          # [x1, y1, x2, y2]
    nearby_label: Optional[str] = None
    label_bbox: Optional[list] = None
    distance_px: float = 0.0
    confidence: float = 0.85
    region: str = "full_page"


@dataclass
class ExtractedInvoice:
    # Header
    invoice_number: Optional[ExtractedField] = None
    invoice_date: Optional[ExtractedField] = None
    due_date: Optional[ExtractedField] = None
    po_number: Optional[ExtractedField] = None
    place_of_supply: Optional[ExtractedField] = None
    category: Optional[ExtractedField] = None
    subcategory: Optional[ExtractedField] = None

    # Vendor
    vendor_name: Optional[ExtractedField] = None
    vendor_address: Optional[ExtractedField] = None
    vendor_address_line1: Optional[ExtractedField] = None
    vendor_address_line2: Optional[ExtractedField] = None
    vendor_gstin: Optional[ExtractedField] = None
    vendor_pan: Optional[ExtractedField] = None
    vendor_email: Optional[ExtractedField] = None
    vendor_phone: Optional[ExtractedField] = None

    # Buyer / Beneficiary
    buyer_name: Optional[ExtractedField] = None
    buyer_address: Optional[ExtractedField] = None
    buyer_address_line1: Optional[ExtractedField] = None
    buyer_address_line2: Optional[ExtractedField] = None
    buyer_gstin: Optional[ExtractedField] = None
    buyer_phone: Optional[ExtractedField] = None
    sls_code: Optional[ExtractedField] = None

    # Line items
    line_items: list[dict] = field(default_factory=list)

    # Totals
    subtotal: Optional[ExtractedField] = None
    tax_rate: Optional[ExtractedField] = None
    tax_amount: Optional[ExtractedField] = None
    discount: Optional[ExtractedField] = None
    round_off: Optional[ExtractedField] = None
    grand_total: Optional[ExtractedField] = None
    amount_in_words: Optional[ExtractedField] = None
    currency: Optional[ExtractedField] = None

    # Tax details
    cgst: Optional[ExtractedField] = None
    sgst: Optional[ExtractedField] = None
    igst: Optional[ExtractedField] = None

    # Payment / Bank
    bank_name: Optional[ExtractedField] = None
    branch_name: Optional[ExtractedField] = None
    account_name: Optional[ExtractedField] = None
    account_number: Optional[ExtractedField] = None
    ifsc_code: Optional[ExtractedField] = None
    payment_terms: Optional[ExtractedField] = None
    remarks: Optional[ExtractedField] = None
    certified_remarks: list[str] = field(default_factory=list)

    # Spatial Candidates (label <-> value evidence for LLM / verification)
    spatial_candidates: list[SpatialCandidate] = field(default_factory=list)

    # Meta
    overall_confidence: float = 0.0
    low_confidence_fields: list[str] = field(default_factory=list)


def calculate_realistic_confidence(inv: ExtractedInvoice) -> float:
    """
    Calculates a realistic, block-filling weighted percentage score (0.0 to 1.0).
    
    Evaluates:
    1. Field coverage across standard invoice blocks (Header, Parties, Items, Totals, Bank).
    2. Quality / confidence of each populated block.
    """
    total_score = 0.0

    # 1. Header & Meta (20%)
    if inv.invoice_number and str(inv.invoice_number.value).strip():
        total_score += 0.10 * max(0.2, min(1.0, inv.invoice_number.confidence))
    if inv.invoice_date and str(inv.invoice_date.value).strip():
        total_score += 0.10 * max(0.2, min(1.0, inv.invoice_date.confidence))

    # 2. Vendor / Biller (20%)
    if inv.vendor_name and len(str(inv.vendor_name.value).strip()) >= 2:
        total_score += 0.12 * max(0.2, min(1.0, inv.vendor_name.confidence))
    
    v_secondary = [
        inv.vendor_address, inv.vendor_address_line1, inv.vendor_gstin,
        inv.vendor_pan, inv.vendor_email, inv.vendor_phone
    ]
    v_sec_filled = [f for f in v_secondary if f and str(f.value).strip()]
    if v_sec_filled:
        avg_v_sec = sum(f.confidence for f in v_sec_filled) / len(v_sec_filled)
        total_score += 0.08 * max(0.2, min(1.0, avg_v_sec))

    # 3. Buyer / Client / Beneficiary (20%)
    if inv.buyer_name and len(str(inv.buyer_name.value).strip()) >= 2:
        total_score += 0.12 * max(0.2, min(1.0, inv.buyer_name.confidence))
    
    b_secondary = [
        inv.buyer_address, inv.buyer_address_line1, inv.buyer_gstin,
        inv.buyer_phone, inv.sls_code
    ]
    b_sec_filled = [f for f in b_secondary if f and str(f.value).strip()]
    if b_sec_filled:
        avg_b_sec = sum(f.confidence for f in b_sec_filled) / len(b_sec_filled)
        total_score += 0.08 * max(0.2, min(1.0, avg_b_sec))

    # 4. Line Items (20%)
    if inv.line_items and len(inv.line_items) > 0:
        valid_desc_items = [
            it for it in inv.line_items
            if bool(str(it.get("description", "")).strip()) and len(str(it.get("description", "")).strip()) >= 2
        ]
        valid_amt_items = [
            it for it in inv.line_items
            if float(it.get("amount", 0) or it.get("rate", 0) or 0) > 0
        ]
        
        # Calculate item confidences
        item_confs = [
            float(it.get("confidence", 0.85)) for it in inv.line_items if "confidence" in it
        ]
        avg_item_conf = (sum(item_confs) / len(item_confs)) if item_confs else 0.85

        if valid_desc_items:
            desc_ratio = min(1.0, len(valid_desc_items) / len(inv.line_items))
            total_score += 0.10 * desc_ratio * max(0.2, min(1.0, avg_item_conf))
        if valid_amt_items:
            amt_ratio = min(1.0, len(valid_amt_items) / len(inv.line_items))
            total_score += 0.10 * amt_ratio * max(0.2, min(1.0, avg_item_conf))

    # 5. Totals & Financials (15%)
    if inv.grand_total and str(inv.grand_total.value).strip():
        try:
            val = float(str(inv.grand_total.value).replace(",", ""))
            if val > 0:
                total_score += 0.10 * max(0.2, min(1.0, inv.grand_total.confidence))
        except (ValueError, TypeError):
            pass

    other_totals = [inv.subtotal, inv.tax_amount, inv.cgst, inv.sgst, inv.igst]
    t_filled = [f for f in other_totals if f and str(f.value).strip()]
    if t_filled:
        avg_t = sum(f.confidence for f in t_filled) / len(t_filled)
        total_score += 0.05 * max(0.2, min(1.0, avg_t))

    # 6. Bank Details (5%)
    bank_fields = [inv.account_number, inv.ifsc_code, inv.bank_name, inv.account_name]
    bk_filled = [f for f in bank_fields if f and str(f.value).strip()]
    if bk_filled:
        avg_bk = sum(f.confidence for f in bk_filled) / len(bk_filled)
        total_score += 0.05 * max(0.2, min(1.0, avg_bk))

    return round(max(0.0, min(1.0, total_score)), 4)


class LayoutLMExtractor:
    """
    LayoutLMv3-based invoice field extractor with comprehensive heuristic fallback engine.
    Loads fine-tuned model if available; falls back to robust pattern and geometric extraction.
    """

    GSTIN_PATTERN = re.compile(
        r"\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}\b"
    )
    PAN_PATTERN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]{1}\b")
    IFSC_PATTERN = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")
    DATE_PATTERNS = [
        re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b"),
        re.compile(r"\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+(\d{2,4})\b", re.IGNORECASE),
        re.compile(r"\b(\d{1,2})[-](Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*[-](\d{2,4})\b", re.IGNORECASE),
        re.compile(r"\b(\d{2})(\d{2})/(\d{4})\b"),  # e.g. 0307/2026 -> 03/07/2026
    ]
    AMOUNT_PATTERN = re.compile(r"(?:\u20b9|Rs\.?|INR|USD|\$)?\s*(\d[\d,]*\.\d{1,2}|\d{2,}[\d,]*)")
    EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b")
    PHONE_PATTERN = re.compile(r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b")
    INVOICE_NUM_PATTERN = re.compile(
        r"(?:invoice\s*(?:no|number|#)|inv\.?\s*(?:no\.?|#)|bill\s*no\.?\s*(?:&|and)?\s*bd\s*date|bill\s*(?:no|number|#)|reference\s*no|ref\s*no\.?|sanction\s*no|memo\s*no|voucher\s*no|token\s*no|challan\s*no)[\s:=]*([A-Z0-9/_-]{2,30})",
        re.IGNORECASE
    )

    MONTH_MAP = {
        "jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06",
        "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12"
    }

    @staticmethod
    def words_to_number(text: str) -> float:
        """Converts Indian and international English amount in words to a float number."""
        if not text:
            return 0.0
        cleaned = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
        tokens = cleaned.split()

        units = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
            'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
            'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
            'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90
        }
        multipliers = {
            'hundred': 100, 'thousand': 1000, 'lakh': 100000, 'lakhs': 100000,
            'lac': 100000, 'lacs': 100000, 'crore': 10000000, 'crores': 10000000,
            'million': 1000000, 'billion': 1000000000
        }

        total = 0
        current_segment = 0

        for token in tokens:
            if token in ['rupees', 'rupee', 'only', 'inr', 'rs', 'and', 'paise', 'paisa']:
                continue
            if token in units:
                current_segment += units[token]
            elif token == 'hundred':
                if current_segment == 0:
                    current_segment = 1
                current_segment *= 100
            elif token in multipliers:
                mult = multipliers[token]
                if current_segment == 0:
                    current_segment = 1
                total += current_segment * mult
                current_segment = 0

        total += current_segment
        return float(total)

    @staticmethod
    def normalize_gstin_candidate(cand: str) -> Optional[str]:
        """Auto-repairs common OCR character confusions (O/0, I/1, S/5, Z/2) in 15-char GSTINs."""
        if not cand:
            return None
        c = cand.strip().upper().replace(' ', '').replace('-', '').replace('/', '').replace(':', '')
        if len(c) < 14 or len(c) > 16:
            return None
        if len(c) == 16 and c.startswith(('I', '1', '0')):
            c = c[1:]
        if len(c) == 14:
            c = c[:13] + 'Z' + c[13:]
        if len(c) != 15:
            return None

        chars = list(c)
        d_map = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8'}
        l_map = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '2': 'Z'}

        # 0, 1 -> 2 digits state code
        for i in [0, 1]:
            if chars[i] in d_map:
                chars[i] = d_map[chars[i]]
        # 2..6 -> 5 letters (PAN)
        for i in range(2, 7):
            if chars[i] in l_map:
                chars[i] = l_map[chars[i]]
        # 7..10 -> 4 digits (PAN number)
        for i in range(7, 11):
            if chars[i] in d_map:
                chars[i] = d_map[chars[i]]
        # 11 -> 1 letter (PAN check)
        if chars[11] in l_map:
            chars[11] = l_map[chars[11]]
        # 12 -> 1 digit / entity
        if chars[12] in d_map:
            chars[12] = d_map[chars[12]]
        # 13 -> default Z
        if chars[13] in ['2', 'S']:
            chars[13] = 'Z'

        res = ''.join(chars)
        if re.match(r'^\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z0-9]{1}[A-Z0-9]{1}[A-Z0-9]{1}$', res):
            return res
        return None

    def __init__(self, model_path: Optional[str] = None, base_model: str = "microsoft/layoutlmv3-base"):
        self.model_path = model_path
        self.base_model = base_model
        self.model = None
        self.processor = None
        self._try_load_model()

    def _try_load_model(self):
        if not self.model_path:
            logger.warning("No LayoutLMv3 model path — using heuristic extraction")
            return
        path = Path(self.model_path)
        if not path.exists():
            logger.warning(f"LayoutLMv3 model not found at {path} — using heuristic extraction")
            return
        try:
            from transformers import (
                LayoutLMv3ForTokenClassification,
                LayoutLMv3Processor,
            )
            self.processor = LayoutLMv3Processor.from_pretrained(str(path))
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(str(path))
            self.model.eval()
            logger.info(f"LayoutLMv3 loaded from {path}")
        except Exception as e:
            logger.error(f"LayoutLMv3 load failed: {e} — falling back to heuristic")

    def extract(
        self,
        ocr_results: dict,      # {region_label: OCRResult}
        image=None,             # PIL Image (required for LayoutLMv3)
    ) -> ExtractedInvoice:
        """
        Extract structured invoice fields from OCR results.
        Uses LayoutLMv3 if available, otherwise heuristic regex pipeline.
        """
        if self.model is not None and image is not None:
            return self._extract_layoutlm(ocr_results, image)
        return self._extract_heuristic(ocr_results)

    def _extract_layoutlm(self, ocr_results: dict, image) -> ExtractedInvoice:
        """LayoutLMv3 token classification extraction with subword alignment and heuristic fallback fusion."""
        import torch

        all_words, all_boxes = [], []
        # Prefer full_page OCR so token order is preserved and not duplicated by region crops
        full_page_results = [v for k, v in ocr_results.items() if "full_page" in k]
        target_ocr_list = full_page_results if full_page_results else list(ocr_results.values())

        w, h = image.size
        seen_spans = set()
        for ocr_result in target_ocr_list:
            for block in ocr_result.text_blocks:
                word_objs = block.words if block.words else decompose_line_into_words(block.text, block.bbox, block.confidence)
                for w_obj in word_objs:
                    w_text = w_obj.text.strip()
                    if not w_text:
                        continue
                    w_bbox = w_obj.bbox
                    if len(w_bbox) == 4 and isinstance(w_bbox[0], (int, float)):
                        x1, y1, x2, y2 = w_bbox
                    else:
                        xs = [p[0] for p in w_bbox]
                        ys = [p[1] for p in w_bbox]
                        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    norm_bbox = [
                        max(0, min(1000, int(1000 * x1 / max(1, w)))),
                        max(0, min(1000, int(1000 * y1 / max(1, h)))),
                        max(0, min(1000, int(1000 * x2 / max(1, w)))),
                        max(0, min(1000, int(1000 * y2 / max(1, h)))),
                    ]
                    span_key = (w_text, norm_bbox[0], norm_bbox[1], norm_bbox[2], norm_bbox[3])
                    if span_key in seen_spans:
                        continue
                    seen_spans.add(span_key)
                    all_words.append(w_text)
                    all_boxes.append(norm_bbox)

        if not all_words:
            return self._extract_heuristic(ocr_results)

        try:
            import torch
            max_chunk = 512
            stride = 256
            word_labels = ["O"] * len(all_words)
            word_confs = [0.0] * len(all_words)

            chunks = []
            if len(all_words) <= max_chunk:
                chunks.append((0, len(all_words)))
            else:
                for start in range(0, len(all_words), stride):
                    end = min(len(all_words), start + max_chunk)
                    chunks.append((start, end))
                    if end == len(all_words):
                        break

            for start_idx, end_idx in chunks:
                chunk_words = all_words[start_idx:end_idx]
                chunk_boxes = all_boxes[start_idx:end_idx]
                try:
                    encoding = self.processor(
                        image,
                        chunk_words,
                        boxes=chunk_boxes,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    )

                    with torch.no_grad():
                        outputs = self.model(**encoding)

                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)
                    predictions = logits.argmax(-1).squeeze().tolist()
                    confidences = probs.max(-1).values.squeeze().tolist()

                    id2label = self.model.config.id2label
                    token_labels = [id2label.get(p, "O") for p in predictions] if isinstance(predictions, list) else [id2label.get(predictions, "O")]
                    token_confs = confidences if isinstance(confidences, list) else [confidences]

                    word_ids = encoding.word_ids(0) if hasattr(encoding, "word_ids") else None

                    if word_ids:
                        for token_idx, local_word_idx in enumerate(word_ids):
                            if local_word_idx is None or local_word_idx >= len(chunk_words):
                                continue
                            global_idx = start_idx + local_word_idx
                            lbl = token_labels[token_idx] if token_idx < len(token_labels) else "O"
                            cnf = float(token_confs[token_idx]) if token_idx < len(token_confs) else 0.0
                            if lbl != "O":
                                if word_labels[global_idx] == "O" or cnf > word_confs[global_idx]:
                                    word_labels[global_idx] = lbl
                                    word_confs[global_idx] = cnf
                            elif word_labels[global_idx] == "O":
                                word_confs[global_idx] = max(word_confs[global_idx], cnf)
                    else:
                        for i in range(min(len(chunk_words), len(token_labels))):
                            global_idx = start_idx + i
                            lbl = token_labels[i]
                            cnf = float(token_confs[i])
                            if lbl != "O" and (word_labels[global_idx] == "O" or cnf > word_confs[global_idx]):
                                word_labels[global_idx] = lbl
                                word_confs[global_idx] = cnf
                except Exception as chunk_err:
                    logger.debug(f"LayoutLM chunk [{start_idx}:{end_idx}] error: {chunk_err}")

            fields = self._group_token_labels(all_words, word_labels, word_confs)
            layoutlm_inv = self._fields_to_invoice(fields, source="layoutlm")
        except Exception as e:
            logger.warning(f"LayoutLM inference warning: {e} — relying on heuristic extraction")
            layoutlm_inv = ExtractedInvoice()

        # Fusion: Merge with comprehensive heuristic extraction
        heuristic_inv = self._extract_heuristic(ocr_results)
        merged = self._merge_invoices(layoutlm_inv, heuristic_inv)
        merged.spatial_candidates = self.generate_spatial_candidates(ocr_results)
        return merged

    def _merge_invoices(self, primary: ExtractedInvoice, secondary: ExtractedInvoice) -> ExtractedInvoice:
        """
        Merge primary with secondary extraction, picking the value with the
        highest effective confidence = raw_confidence × SOURCE_WEIGHTS[source].
        """
        merged = ExtractedInvoice()
        all_field_names = [
            "invoice_number", "invoice_date", "due_date", "po_number", "place_of_supply", "category", "subcategory",
            "vendor_name", "vendor_address", "vendor_address_line1", "vendor_address_line2", "vendor_gstin", "vendor_pan", "vendor_email", "vendor_phone",
            "buyer_name", "buyer_address", "buyer_address_line1", "buyer_address_line2", "buyer_gstin", "buyer_phone", "sls_code",
            "subtotal", "tax_rate", "tax_amount", "discount", "round_off", "grand_total", "amount_in_words", "currency",
            "cgst", "sgst", "igst",
            "bank_name", "branch_name", "account_name", "account_number", "ifsc_code", "payment_terms", "remarks"
        ]

        for fname in all_field_names:
            p_val = getattr(primary, fname, None)
            s_val = getattr(secondary, fname, None)
            p_valid = p_val is not None and getattr(p_val, "value", None) and len(str(p_val.value).strip()) > 0
            s_valid = s_val is not None and getattr(s_val, "value", None) and len(str(s_val.value).strip()) > 0

            if p_valid and s_valid:
                if fname == "buyer_name" and len(str(p_val.value).strip()) <= 3 and len(str(s_val.value).strip()) > 3:
                    chosen = s_val
                else:
                    p_eff = p_val.confidence * SOURCE_WEIGHTS.get(p_val.source, 0.5)
                    s_eff = s_val.confidence * SOURCE_WEIGHTS.get(s_val.source, 0.5)
                    chosen = p_val if p_eff >= s_eff else s_val
                setattr(merged, fname, chosen)
            elif p_valid and p_val.confidence >= 0.30:
                setattr(merged, fname, p_val)
            elif s_valid:
                setattr(merged, fname, s_val)

        # Certified remarks
        if primary.certified_remarks and len(primary.certified_remarks) > 0:
            merged.certified_remarks = primary.certified_remarks
        else:
            merged.certified_remarks = secondary.certified_remarks

        # Line items
        if primary.line_items and len(primary.line_items) > 0:
            merged.line_items = primary.line_items
        else:
            merged.line_items = secondary.line_items

        # Recalculate realistic confidence score
        merged.overall_confidence = calculate_realistic_confidence(merged)
        return merged

    def _group_token_labels(self, words, labels, confidences) -> dict:
        """Convert BIO token labels to grouped field values."""
        fields = {}
        current_label = None
        current_words = []
        current_confs = []

        for word, label, conf in zip(words, labels, confidences):
            if conf < 0.40:
                label = "O"

            if label.startswith("B-"):
                if current_label and current_words:
                    key = current_label[2:]
                    fields[key] = (" ".join(current_words), float(np.mean(current_confs)))
                current_label = label
                current_words = [word]
                current_confs = [conf]
            elif label.startswith("I-") and current_label and label[2:] == current_label[2:]:
                current_words.append(word)
                current_confs.append(conf)
            else:
                if current_label and current_words:
                    key = current_label[2:]
                    fields[key] = (" ".join(current_words), float(np.mean(current_confs)))
                current_label = None
                current_words = []
                current_confs = []

        if current_label and current_words:
            key = current_label[2:]
            fields[key] = (" ".join(current_words), float(np.mean(current_confs)))

        return fields

    def _fields_to_invoice(self, fields: dict, source: str) -> ExtractedInvoice:
        inv = ExtractedInvoice()
        mapping = {
            "INVOICE_NUMBER": "invoice_number",
            "INVOICE_DATE": "invoice_date",
            "DUE_DATE": "due_date",
            "PO_NUMBER": "po_number",
            "PLACE_OF_SUPPLY": "place_of_supply",
            "CATEGORY": "category",
            "SUBCATEGORY": "subcategory",
            "VENDOR_NAME": "vendor_name",
            "BILLER_NAME": "vendor_name",
            "VENDOR_ADDRESS": "vendor_address",
            "BILLER_ADDRESS": "vendor_address",
            "VENDOR_GSTIN": "vendor_gstin",
            "VENDOR_PAN": "vendor_pan",
            "VENDOR_EMAIL": "vendor_email",
            "VENDOR_PHONE": "vendor_phone",
            "GST": "vendor_gstin",
            "BUYER_NAME": "buyer_name",
            "BUYER_ADDRESS": "buyer_address",
            "BUYER_GSTIN": "buyer_gstin",
            "BUYER_PHONE": "buyer_phone",
            "SLS_CODE": "sls_code",
            "SUBTOTAL": "subtotal",
            "TAX_AMOUNT": "tax_amount",
            "CGST": "cgst",
            "SGST": "sgst",
            "IGST": "igst",
            "DISCOUNT": "discount",
            "ROUND_OFF": "round_off",
            "GRAND_TOTAL": "grand_total",
            "TOTAL": "grand_total",
            "AMOUNT_IN_WORDS": "amount_in_words",
            "CURRENCY": "currency",
            "BANK_NAME": "bank_name",
            "BRANCH_NAME": "branch_name",
            "ACCOUNT_NAME": "account_name",
            "ACCOUNT_NUMBER": "account_number",
            "IFSC_CODE": "ifsc_code",
            "PAYMENT_TERMS": "payment_terms",
            "REMARKS": "remarks",
        }
        for label_key, field_name in mapping.items():
            if label_key in fields:
                if getattr(inv, field_name) is None:
                    value, conf = fields[label_key]
                    setattr(inv, field_name, ExtractedField(value=value, confidence=conf, source=source))
        return inv

    def _extract_heuristic(self, ocr_results: dict) -> ExtractedInvoice:
        """
        Comprehensive heuristic extraction engine.
        Parses standard invoices, government bill details, HRMS annexures, vouchers, and challans.
        """
        inv = ExtractedInvoice()
        region_texts = {k: v.full_text for k, v in ocr_results.items() if "full_page" not in k}
        full_page_texts = [v.full_text for k, v in sorted(ocr_results.items()) if "full_page" in k]
        
        if full_page_texts:
            all_text = "\n".join(full_page_texts)
        else:
            all_text = "\n".join(v.full_text for v in ocr_results.values())

        # Clean noise characters
        clean_text = all_text.replace("\r", "\n")

        # --- 1. Invoice / Bill / Reference Number ---
        header_text = region_texts.get("header", "")
        search_scope = (header_text + "\n" + clean_text).strip()

        # Try bill no / invoice no patterns
        inv_num_match = self.INVOICE_NUM_PATTERN.search(search_scope)
        if inv_num_match:
            candidate_num = inv_num_match.group(1).strip()
            # If candidate was '0307/2026' or just date digits, check if preceded by bill code
            if len(candidate_num) >= 2 and not re.match(r"^\d{4,8}$", candidate_num) or len(candidate_num) <= 20:
                inv.invoice_number = ExtractedField(candidate_num, 0.85, "heuristic")

        if not inv.invoice_number:
            ref_match = re.search(r"(?:reference\s*no|ref\s*no|sanction\s*no|memo\s*no|voucher\s*no)[\s:=]*([A-Z0-9/_-]{2,30})", clean_text, re.I)
            if ref_match:
                inv.invoice_number = ExtractedField(ref_match.group(1).strip(), 0.82, "heuristic")

        # --- 1b. Purchase Order (PO) / Work Order Number ---
        po_match = re.search(
            r"(?:P\.?O\.?\s*(?:No\.?|Number|#)|Purchase\s*Order\s*(?:No\.?|#)?|Work\s*Order\s*(?:No\.?|#)?|WO\s*(?:No\.?|#))[\s:=]*([A-Z0-9/_-]{2,30})",
            clean_text, re.IGNORECASE
        )
        if po_match:
            inv.po_number = ExtractedField(po_match.group(1).strip(), 0.85, "heuristic")

        # --- 2. Invoice / Bill Date ---
        # Look for explicit date keywords first (BD Date, Invoice Date, Bill Date, Date:)
        date_explicit = re.search(
            r"(?:invoice\s*date|bill\s*date|bd\s*date|user\s*date|bill\s*no\s*&\s*bd\s*date\s+[A-Za-z0-9_-]+\s+|date)[\s:=]*(\d{2})(\d{2})[/-](\d{4})",
            clean_text, re.IGNORECASE
        )
        if date_explicit:
            formatted_date = f"{date_explicit.group(1)}/{date_explicit.group(2)}/{date_explicit.group(3)}"
            inv.invoice_date = ExtractedField(formatted_date, 0.88, "heuristic")

        if not inv.invoice_date:
            for pat in self.DATE_PATTERNS:
                m = pat.search(search_scope)
                if m:
                    g = m.groups()
                    if len(g) == 3:
                        d_str = str(g[0]).zfill(2)
                        m_raw = str(g[1]).lower()
                        m_str = self.MONTH_MAP.get(m_raw[:3], m_raw.zfill(2))
                        y_raw = str(g[2])
                        y_str = f"20{y_raw}" if len(y_raw) == 2 and int(y_raw) <= 50 else y_raw
                        inv.invoice_date = ExtractedField(f"{d_str}/{m_str}/{y_str}", 0.88, "heuristic")
                    else:
                        inv.invoice_date = ExtractedField(m.group(0).strip(), 0.80, "heuristic")
                    break

        # Due Date
        due_match = re.search(r"(?:due\s*date)[\s:=]*([0-9A-Za-z/ -]+)", search_scope, re.IGNORECASE)
        if due_match:
            for pat in self.DATE_PATTERNS:
                dm = pat.search(due_match.group(0))
                if dm:
                    inv.due_date = ExtractedField(dm.group(0).strip(), 0.75, "heuristic")
                    break

        # Place of supply
        pos_match = re.search(r"(?:place\s*of\s*supply|pos)[\s:=]*([A-Za-z0-9\s()-]+)", search_scope, re.IGNORECASE)
        if pos_match:
            pos_val = pos_match.group(1).split("\n")[0].strip()
            if len(pos_val) < 60:
                inv.place_of_supply = ExtractedField(pos_val, 0.85, "heuristic")

        # --- 3. Category & Subcategory & SLS Code ---
        cat_match = re.search(r"\bcategory[\s:=]+([A-Za-z0-9\s&/_-]+)", search_scope, re.IGNORECASE)
        if cat_match:
            cat_val = cat_match.group(1).split("\n")[0].strip()
            if cat_val and not any(kw in cat_val.lower() for kw in ["sub category", "subcategory"]):
                inv.category = ExtractedField(cat_val, 0.88, "heuristic")
        elif "hrms" in clean_text.lower() or "annexure" in clean_text.lower() or "employee bill" in clean_text.lower():
            inv.category = ExtractedField("Employee Bill / HRMS Details", 0.85, "heuristic")

        subcat_match = re.search(r"\bsub\s*category[\s:=]+([A-Za-z0-9\s&/_-]+)", search_scope, re.IGNORECASE)
        if subcat_match:
            subcat_val = subcat_match.group(1).split("\n")[0].strip()
            if subcat_val:
                inv.subcategory = ExtractedField(subcat_val, 0.88, "heuristic")
        elif "gpf" in clean_text.lower():
            inv.subcategory = ExtractedField("GPF Advance / Withdrawal", 0.85, "heuristic")
        elif "amc" in clean_text.lower():
            inv.subcategory = ExtractedField("AMC Charges", 0.85, "heuristic")
        elif "tr-50" in clean_text.lower() or "tr-5o" in clean_text.lower():
            inv.subcategory = ExtractedField("TR-50", 0.85, "heuristic")

        # SLS / Scheme / DDO Code / Head of Account
        ddo_match = re.search(r"DDO\s*(?:Code|Codo)[\s:=]*([A-Z0-9]+)", clean_text, re.I)
        sls_match = re.search(r"(?:sls\s*code|scheme\s*code)[\s:=]*([A-Za-z0-9_-]+)", clean_text, re.IGNORECASE)
        if ddo_match:
            clean_ddo = ddo_match.group(1).replace("OO", "00").strip()
            inv.sls_code = ExtractedField(f"DDO: {clean_ddo}", 0.90, "heuristic")
        elif sls_match:
            inv.sls_code = ExtractedField(sls_match.group(1).strip(), 0.90, "heuristic")
        else:
            scheme_match = re.search(r"\(([A-Z]{4}\d{8}|SCH\d+)\)", clean_text)
            if scheme_match:
                inv.sls_code = ExtractedField(scheme_match.group(1).strip(), 0.85, "heuristic")

        # --- 4. Vendor / Authority / Biller Block ---
        vendor_text = region_texts.get("vendor_block", "")
        if vendor_text:
            lines = [l.strip() for l in vendor_text.split("\n") if l.strip()]
            if lines:
                inv.vendor_name = ExtractedField(lines[0], 0.78, "heuristic")
            if len(lines) > 1:
                v_addr_lines = [
                    l for l in lines[1:]
                    if not any(k in l.lower() for k in ["gstin", "pan", "email", "phone", "invoice", "tax", "http", "url", "page", "e-signed"])
                ]
                if v_addr_lines:
                    inv.vendor_address = ExtractedField(" ".join(v_addr_lines[:3]), 0.75, "heuristic")
                    inv.vendor_address_line1 = ExtractedField(v_addr_lines[0], 0.75, "heuristic")
                    if len(v_addr_lines) > 1:
                        inv.vendor_address_line2 = ExtractedField(v_addr_lines[1], 0.70, "heuristic")

        # Government / Department / DDO Authority Vendor Scanner
        if not inv.vendor_name:
            gov_m = re.search(
                r"((?:REGISTRAR\s*&\s*DDO|GOVT[.:; ]+OF|DEPARTMENT\s+OF|OFFICE\s+OF)[A-Z\s,;]+(?:WEST\s+BENGAL|[A-Z]{3,}))",
                clean_text, re.IGNORECASE
            )
            if gov_m:
                v_name = gov_m.group(1).strip().replace(";", ",").replace("  ", " ")
                inv.vendor_name = ExtractedField(v_name[:80], 0.88, "heuristic")
                inv.vendor_address_line1 = ExtractedField("Govt of West Bengal", 0.80, "heuristic")
            else:
                dept_m = re.search(r"(?:Gov[.:; ]+of\s+[A-Za-z\s]+|Government\s+of\s+[A-Za-z\s]+)", clean_text, re.IGNORECASE)
                if dept_m:
                    inv.vendor_name = ExtractedField(dept_m.group(0).strip()[:80], 0.85, "heuristic")
                elif header_text:
                    lines = [
                        l.strip() for l in header_text.split("\n")
                        if l.strip() and not any(k in l.lower() for k in ["invoice", "tax", "gstin", "date", "bill", "issue", "page", "url", "original"])
                    ]
                    if lines:
                        inv.vendor_name = ExtractedField(lines[0], 0.70, "heuristic")

        # Vendor GSTIN, PAN, Email, Phone
        all_gstins = self.GSTIN_PATTERN.findall(clean_text)
        if len(all_gstins) < 2:
            # Fuzzy GSTIN search (detects OCR typos like O->0, I->1, GSTINIUIN prefix)
            fuzzy_cands = re.findall(r"(?:GSTIN|GST|UIN|GSTN|INUIN)[:\sI/_-]*([A-Za-z0-9]{14,16})", clean_text, re.IGNORECASE)
            fuzzy_cands += re.findall(r"\b([0-9A-Za-z]{15})\b", clean_text)
            for cand in fuzzy_cands:
                norm_g = self.normalize_gstin_candidate(cand)
                if norm_g and norm_g not in all_gstins:
                    all_gstins.append(norm_g)

        if all_gstins:
            inv.vendor_gstin = ExtractedField(all_gstins[0], 0.92, "heuristic")
            if len(all_gstins) > 1:
                inv.buyer_gstin = ExtractedField(all_gstins[1], 0.90, "heuristic")

        pan = self.PAN_PATTERN.search(vendor_text + " " + clean_text)
        if pan:
            inv.vendor_pan = ExtractedField(pan.group(0), 0.88, "heuristic")

        email = self.EMAIL_PATTERN.search(clean_text)
        if email:
            inv.vendor_email = ExtractedField(email.group(0), 0.92, "heuristic")

        phone = self.PHONE_PATTERN.search(clean_text)
        if phone:
            inv.vendor_phone = ExtractedField(phone.group(0), 0.88, "heuristic")

        # --- 5. Buyer / Beneficiary / Client Block ---
        buyer_text = region_texts.get("buyer_block", "")
        _BUYER_SKIP = {
            "bill to", "billed to", "buyer", "customer", "ship to",
            "taxable", "taxable value", "taxable amount", "amount", "total", "value", "tax",
            "description", "particulars", "item", "rate", "qty", "quantity",
            "unit", "uom", "disc", "disc.", "discount", "hsn", "sac", "sls", "sls code",
        }
        is_table_crop = any(k in buyer_text.lower() for k in ["description", "hsn", "qty", "rate", "disc.", "taxable value"])

        if buyer_text and not is_table_crop:
            lines = [l.strip() for l in buyer_text.split("\n") if l.strip()]
            candidate_lines = [
                l for l in lines
                if l.lower() not in _BUYER_SKIP
                and not any(kw in l.lower() for kw in ["taxable", "amount", "total", "value", "hsn", "rate", "qty", "unit", "disc", "invoice"])
                and len(l) > 3
                and not all(c in "#.-_/|* " for c in l)
            ]
            if candidate_lines:
                inv.buyer_name = ExtractedField(candidate_lines[0], 0.75, "heuristic")
            if len(candidate_lines) > 1:
                b_addr_lines = [
                    l for l in candidate_lines[1:]
                    if not any(kw in l.lower() for kw in ["sls", "phone", "gstin", "invoice", "date", "due", "supply", "category", "charges"])
                ]
                if b_addr_lines:
                    inv.buyer_address = ExtractedField(" ".join(b_addr_lines[:3]), 0.70, "heuristic")
                    inv.buyer_address_line1 = ExtractedField(b_addr_lines[0], 0.70, "heuristic")
                    if len(b_addr_lines) > 1:
                        inv.buyer_address_line2 = ExtractedField(b_addr_lines[1], 0.65, "heuristic")

        # Multi-line 'Bill To' scanner
        all_lines = [l.strip() for l in clean_text.split("\n") if l.strip()]
        for i, line in enumerate(all_lines):
            if re.match(r"^(?:bill\s*to|billed\s*to|buyer)[:\s]*$", line, re.I):
                for next_line in all_lines[i+1:i+6]:
                    if not any(kw in next_line.lower() for kw in ["invoice", "date", "gstin", "pan", "sls", "place of supply", "taxable", "phone", "url", "category", "charges"]):
                        if len(next_line) > 3 and not re.match(r"^\d+$", next_line):
                            inv.buyer_name = ExtractedField(next_line, 0.90, "heuristic")
                            break
                if inv.buyer_name:
                    addr_cands = []
                    for addr_line in all_lines[i+2:i+7]:
                        if "phone:" in addr_line.lower() or "mobile:" in addr_line.lower():
                            ph_m = re.search(r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b", addr_line)
                            if ph_m:
                                inv.buyer_phone = ExtractedField(ph_m.group(0), 0.88, "heuristic")
                        elif not any(kw in addr_line.lower() for kw in ["sls", "invoice", "date", "due", "place of supply", "category", "charges", "gstin", "pan", "#", "description", "taxable", "rate"]):
                            if len(addr_line) > 2 and not re.match(r"^\d+$", addr_line):
                                addr_cands.append(addr_line)
                    if addr_cands and not inv.buyer_address_line1:
                        inv.buyer_address_line1 = ExtractedField(addr_cands[0], 0.75, "heuristic")
                        if len(addr_cands) > 1:
                            inv.buyer_address_line2 = ExtractedField(addr_cands[1], 0.70, "heuristic")
                    break

        # Beneficiary / Employee / Payee Name Scanner (HRMS / Vouchers)
        if not inv.buyer_name:
            ben_m = re.search(
                r"\b((?:SRI|SMT|DR|MR|MS|MD)\s+[A-Z\s]{4,35}?)(?=\s+\d{10,18}|\s+@|\s+[A-Z]{4}0|\s+Total|\s+ECS|\s+NEFT|\s+RTGS|\s+Phone|\s+GSTIN|\n|$)",
                clean_text
            )
            if ben_m:
                b_cand = ben_m.group(1).strip()
                inv.buyer_name = ExtractedField(b_cand, 0.88, "heuristic")
            else:
                payee_m = re.search(
                    r"(?:Beneficiary\s*(?:Id)?\s*Name|Employee\s*Name|Payee\s*Name|In\s*favour\s*of|Paid\s*to)[\s:=]*([A-Za-z\s.]{3,40})",
                    clean_text, re.IGNORECASE
                )
                if payee_m:
                    cand = payee_m.group(1).split("\n")[0].strip()
                    if len(cand) > 3 and not any(kw in cand.lower() for kw in ["account", "ifsc", "mode", "amount"]):
                        inv.buyer_name = ExtractedField(cand, 0.85, "heuristic")

        if not inv.buyer_phone:
            buyer_ph_m = re.search(r"(?:bill\s*to[\s\S]{1,300}?)phone[\s:=]*(\+?\d[\d\s-]{8,14})", clean_text, re.IGNORECASE)
            if buyer_ph_m:
                inv.buyer_phone = ExtractedField(buyer_ph_m.group(1).strip(), 0.88, "heuristic")

        # --- 6. Totals & Financials ---
        totals_text = region_texts.get("totals_block", "") + " " + region_texts.get("tax_block", "") + " " + clean_text
        grand_total = self._find_amount_near_keyword(
            totals_text, ["grand total", "total amount", "net amount", "nal amount", "bill gross", "amount due", "invoice total", "total value", "total"]
        )
        if grand_total:
            inv.grand_total = ExtractedField(grand_total, 0.85, "heuristic")

        subtotal = self._find_amount_near_keyword(
            totals_text, ["subtotal", "sub-total", "bill gross", "taxable value", "taxable amount", "total taxable", "net total"]
        )
        if subtotal:
            inv.subtotal = ExtractedField(subtotal, 0.80, "heuristic")
        elif grand_total and not inv.subtotal:
            inv.subtotal = ExtractedField(grand_total, 0.75, "heuristic")

        round_off = self._find_amount_near_keyword(
            totals_text, ["round off", "roundoff", "rounding"]
        )
        if round_off:
            inv.round_off = ExtractedField(round_off, 0.70, "heuristic")

        num_words_kw = ["rupees", "only", "thousand", "hundred", "lakh", "crore", "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "twenty", "thirty", "forty", "fifty"]
        words_match = re.search(r"(?:amount\s*in\s*words|in\s*words|rupees\s*in\s*words)[\s:=]*([A-Za-z\s/-]+(?:only)?)", clean_text, re.IGNORECASE)
        if words_match:
            candidate_words = words_match.group(1).split("\n")[0].strip()
            candidate_words = re.sub(r"^(?:amount\s*in\s*words|in\s*words)\s*[:.\-]?\s*", "", candidate_words, flags=re.I).strip()
            if len(candidate_words) > 5 and any(kw in candidate_words.lower() for kw in num_words_kw):
                inv.amount_in_words = ExtractedField(candidate_words, 0.88, "heuristic")

        # Tax
        tax_text = region_texts.get("tax_block", totals_text)
        cgst = self._find_amount_near_keyword(tax_text, ["+cgst :", "+cgst", "cgst amt", "cgst amount", "central tax", "cgst"])
        if cgst:
            inv.cgst = ExtractedField(cgst, 0.78, "heuristic")

        igst = self._find_amount_near_keyword(tax_text, ["+igst :", "+igst", "igst amt", "igst amount", "integrated tax", "igst"])
        if igst:
            inv.igst = ExtractedField(igst, 0.78, "heuristic")

        tax_amount = self._find_amount_near_keyword(
            tax_text, ["gst total", "total tax", "tax total", "total gst", "gst amount", "vat total"]
        )
        def safe_float(v) -> float:
            if v is None:
                return 0.0
            if isinstance(v, (int, float)):
                return float(v)
            try:
                clean_v = str(v).replace(",", "").replace("₹", "").replace("Rs", "").strip()
                return float(clean_v)
            except (ValueError, TypeError):
                return 0.0

        if tax_amount:
            inv.tax_amount = ExtractedField(safe_float(tax_amount), 0.70, "heuristic")
        elif inv.cgst or inv.sgst or inv.igst:
            cgst_v = safe_float(inv.cgst.value if inv.cgst else 0.0)
            sgst_v = safe_float(inv.sgst.value if inv.sgst else 0.0)
            igst_v = safe_float(inv.igst.value if inv.igst else 0.0)
            calc_tax = cgst_v + sgst_v + igst_v
            if calc_tax > 0:
                inv.tax_amount = ExtractedField(round(calc_tax, 2), 0.85, "heuristic_calculated")

        # Intra-state dual GST reconciliation: CGST == SGST when total tax == 2 * CGST
        if inv.cgst and inv.cgst.value and inv.tax_amount and inv.tax_amount.value:
            cgst_v = safe_float(inv.cgst.value)
            tax_v = safe_float(inv.tax_amount.value)
            if cgst_v > 0 and abs((cgst_v * 2) - tax_v) <= 0.05:
                if not inv.sgst or safe_float(inv.sgst.value) != cgst_v:
                    inv.sgst = ExtractedField(cgst_v, 0.95, "heuristic_reconciled")

        # --- Math Auto-Reconciliation & Inversion Guard ---
        total_tax_val = safe_float(inv.tax_amount.value if inv.tax_amount else 0.0)
        subtotal_val = safe_float(inv.subtotal.value if inv.subtotal else 0.0)
        grand_total_val = safe_float(inv.grand_total.value if inv.grand_total else 0.0)

        if inv.amount_in_words:
            num_from_words = self.words_to_number(inv.amount_in_words.value)
            if num_from_words > 0:
                if not inv.grand_total or abs(grand_total_val - num_from_words) > 1.0:
                    if subtotal_val > 0 and abs((subtotal_val + total_tax_val) - num_from_words) <= 2.0:
                        inv.grand_total = ExtractedField(num_from_words, 0.95, "heuristic_reconciled")
                    elif grand_total_val > 0 and grand_total_val < num_from_words:
                        inv.subtotal = ExtractedField(grand_total_val, 0.90, "heuristic_reconciled")
                        inv.grand_total = ExtractedField(num_from_words, 0.95, "heuristic_reconciled")
                    elif not inv.grand_total:
                        inv.grand_total = ExtractedField(num_from_words, 0.90, "heuristic_reconciled")

        # Subtotal sanity: if subtotal > grand_total, reconstruct subtotal = grand_total - tax
        subtotal_val = safe_float(inv.subtotal.value if inv.subtotal else 0.0)
        grand_total_val = safe_float(inv.grand_total.value if inv.grand_total else 0.0)
        if subtotal_val > 0 and grand_total_val > 0 and subtotal_val > grand_total_val:
            if grand_total_val > total_tax_val and total_tax_val > 0:
                inv.subtotal = ExtractedField(round(grand_total_val - total_tax_val, 2), 0.90, "heuristic_reconciled")

        # Currency
        if "₹" in clean_text or "INR" in clean_text or "Rs" in clean_text or "Rupees" in clean_text:
            inv.currency = ExtractedField("INR", 0.95, "heuristic")
        elif "$" in clean_text or "USD" in clean_text:
            inv.currency = ExtractedField("USD", 0.95, "heuristic")
        else:
            inv.currency = ExtractedField("INR", 0.70, "heuristic")

        # --- 7. Bank Details & Settlement ---
        # Account Number
        ac_m = re.search(r"(?:AC\s*(?:No\.?|Codo)?|A/C\s*(?:No\.?|Codo)?|Account\s*No\.?)[\s:#.]*(\d{9,18})", clean_text, re.I)
        if ac_m:
            inv.account_number = ExtractedField(ac_m.group(1).strip(), 0.90, "heuristic")
        else:
            # Check after beneficiary name or document-wide account digits
            if inv.buyer_name:
                idx_b = clean_text.find(inv.buyer_name.value)
                if idx_b != -1:
                    post_text = clean_text[idx_b:]
                    ac_post = re.search(r"\b(\d{11,18})\b", post_text)
                    if ac_post:
                        inv.account_number = ExtractedField(ac_post.group(1), 0.88, "heuristic")

        if not inv.account_number:
            acc_match = re.search(r"\b00\d{10,16}\b|\b\d{11,18}\b", clean_text)
            if acc_match and (not inv.invoice_number or acc_match.group(0) != inv.invoice_number.value):
                inv.account_number = ExtractedField(acc_match.group(0), 0.80, "heuristic")

        # IFSC Code & OCR Glitch Repair (@ -> B, O -> 0)
        def parse_ifsc_token(raw_str: str) -> Optional[str]:
            if not raw_str:
                return None
            cand = raw_str.strip().upper().replace("@", "B")
            if len(cand) == 11:
                # 5th char must be 0 (convert OCR 'O', 'D', 'Q')
                if cand[4] in ["0", "O", "D", "Q"]:
                    cand = cand[:4] + "0" + cand[5:]
                if re.match(r"^[A-Z]{4}0[A-Z0-9]{6}$", cand):
                    # Avoid false positives from OCR English words
                    if not any(skip in cand for skip in ["EMPL", "BENE", "DEPT", "ANNE", "OFFI", "PAYE"]):
                        return cand
            return None

        # 1. Search after IFSC keyword
        ifsc_kw_m = re.search(r"(?:IFSC\s*(?:Codo|Code)?)[\s:=]*([@A-Za-z0-9]{11})", clean_text, re.IGNORECASE)
        if ifsc_kw_m:
            clean_ifsc = parse_ifsc_token(ifsc_kw_m.group(1))
            if clean_ifsc:
                inv.ifsc_code = ExtractedField(clean_ifsc, 0.95, "heuristic")

        # 2. Search after account number or in document
        if not inv.ifsc_code:
            search_area = clean_text
            if inv.account_number:
                idx_acc = clean_text.find(inv.account_number.value)
                if idx_acc != -1:
                    search_area = clean_text[idx_acc:]
            
            for token in re.findall(r"[@A-Za-z0-9]{11}", search_area):
                clean_ifsc = parse_ifsc_token(token)
                if clean_ifsc:
                    inv.ifsc_code = ExtractedField(clean_ifsc, 0.92, "heuristic")
                    break

        if not inv.ifsc_code:
            for token in re.findall(r"[@A-Za-z0-9]{11}", clean_text):
                clean_ifsc = parse_ifsc_token(token)
                if clean_ifsc:
                    inv.ifsc_code = ExtractedField(clean_ifsc, 0.88, "heuristic")
                    break

        # Bank Name
        if inv.ifsc_code and inv.ifsc_code.value:
            ifsc_pfx = inv.ifsc_code.value[:4]
            if ifsc_pfx in IFSC_BANK_MAP:
                inv.bank_name = ExtractedField(IFSC_BANK_MAP[ifsc_pfx], 0.92, "heuristic")

        if not inv.bank_name:
            bank_match = re.search(r"\bbank[\s:=]+(?!details\b)([A-Za-z\s&]+(?:bank|ltd|limited)?)", clean_text, re.IGNORECASE)
            if bank_match:
                b_val = bank_match.group(1).split("\n")[0].strip()
                if len(b_val) < 50 and b_val.lower() not in ["details"]:
                    inv.bank_name = ExtractedField(b_val, 0.88, "heuristic")

        # Account Name & Branch Name
        if inv.buyer_name:
            inv.account_name = ExtractedField(inv.buyer_name.value, 0.85, "heuristic")
        elif inv.vendor_name:
            inv.account_name = ExtractedField(inv.vendor_name.value, 0.80, "heuristic")

        branch_match = re.search(r"\bbranch[\s:=]+([A-Za-z0-9\s,-]+)", clean_text, re.IGNORECASE)
        if branch_match:
            br_val = branch_match.group(1).split("\n")[0].strip()
            if len(br_val) < 50:
                inv.branch_name = ExtractedField(br_val, 0.85, "heuristic")
        else:
            micr_m = re.search(r"(?:MICR|MKCR)[\s:=]*(?:No\.?)?[\s:=]*(\d{9})", clean_text, re.I)
            if micr_m:
                inv.branch_name = ExtractedField(f"MICR: {micr_m.group(1)}", 0.85, "heuristic")

        # Payment Terms
        pt_m = re.search(r"(?:Payment\s*Terms?|Terms\s*of\s*Payment)[\s:=]*([A-Za-z0-9\s,./-]{3,50})", clean_text, re.IGNORECASE)
        if pt_m and not inv.payment_terms:
            inv.payment_terms = ExtractedField(pt_m.group(1).strip(), 0.82, "heuristic")

        # Remarks & Declarations
        rem_m = re.search(r"(?:Remarks?|Notes?)[\s:=]*([^\n]{5,150})", clean_text, re.IGNORECASE)
        if rem_m and not inv.remarks:
            inv.remarks = ExtractedField(rem_m.group(1).strip(), 0.80, "heuristic")

        # Dynamic Extraction of Certified Remarks / Declarations from Invoice
        extracted_certs = []
        for pattern in [
            r"(?:Certified\s+that[^\n]{10,250})",
            r"(?:Declaration[\s:=]+[^\n]{10,250})",
            r"(?:We\s+(?:hereby\s+)?declare\s+that[^\n]{10,250})",
            r"(?:It\s+is\s+certified\s+that[^\n]{10,250})",
            r"(?:Subject\s+to\s+[A-Za-z\s]+\s+Jurisdiction[^\n]*)",
            r"(?:Stock\s+Entry\s+No\.?[^\n]{5,120})",
            r"(?:The\s+rates\s+charged\s+are\s+as\s+per[^\n]{10,150})",
            r"(?:The\s+claim\s+has\s+not\s+been\s+paid[^\n]{10,150})",
            r"(?:Goods\s+once\s+sold\s+will\s+not\s+be\s+taken\s+back[^\n]*)",
        ]:
            for m in re.finditer(pattern, clean_text, re.IGNORECASE):
                c_text = re.sub(r"\s+", " ", m.group(0).strip())
                if len(c_text) >= 15 and c_text not in extracted_certs:
                    extracted_certs.append(c_text)
        inv.certified_remarks = extracted_certs

        # --- 8. Line Items ---
        region_items = self._extract_line_items(region_texts.get("line_items", ""))
        if region_items:
            inv.line_items = region_items
        else:
            # Full-page line items fallback
            inv.line_items = self._extract_full_page_line_items(clean_text, inv)

        # Generate spatial candidates (label <-> value pairs with coordinates)
        inv.spatial_candidates = self.generate_spatial_candidates(ocr_results)

        # Calculate realistic overall confidence score
        inv.overall_confidence = calculate_realistic_confidence(inv)
        return inv

    def _find_amount_near_keyword(self, text: str, keywords: list[str]) -> Optional[str]:
        """Search for a currency amount near a keyword."""
        text_lower = text.lower()
        for kw in keywords:
            idx = text_lower.find(kw.lower())
            if idx == -1:
                continue
            snippet = text[idx:idx + 80]
            for match in self.AMOUNT_PATTERN.finditer(snippet):
                raw = match.group(1).replace(",", "")
                try:
                    val = float(raw)
                    # For tax components or general amounts, filter out HSN/PIN codes >= 100,000
                    if val >= 100000.0 and any(t in kw.lower() for t in ["cgst", "sgst", "igst", "tax", "disc", "round"]):
                        continue
                    return raw
                except ValueError:
                    continue
        return None

    def _extract_line_items(self, line_items_text: str) -> list[dict]:
        """Simple line item extractor for regional crops."""
        if not line_items_text:
            return []

        items = []
        lines = [l.strip() for l in line_items_text.split("\n") if l.strip()]

        for line in lines:
            if any(h in line.lower() for h in ["description", "item", "s.no", "sr.", "qty", "rate", "amount"]):
                continue
            amounts = self.AMOUNT_PATTERN.findall(line)
            if len(amounts) >= 1:
                amount = amounts[-1].replace(",", "")
                qty = amounts[0].replace(",", "") if len(amounts) >= 2 else "1"
                rate = amounts[-2].replace(",", "") if len(amounts) >= 3 else amount
                desc_match = re.match(r"^([\w\s,.\-/()]+?)(?:\s+\d)", line)
                desc = desc_match.group(1).strip() if desc_match else line[:50]
                if desc and amount:
                    items.append({
                        "description": desc,
                        "quantity": float(qty) if str(qty).replace(".", "", 1).isdigit() else 1.0,
                        "unit": "NOS",
                        "rate": float(rate) if str(rate).replace(".", "", 1).isdigit() else float(amount),
                        "amount": float(amount),
                        "taxable_value": float(amount),
                        "discount": 0.0,
                        "cgst_rate": 0.0,
                        "cgst_amount": 0.0,
                        "sgst_rate": 0.0,
                        "sgst_amount": 0.0,
                        "igst_rate": 0.0,
                        "igst_amount": 0.0,
                    })

        return items

    def _extract_full_page_line_items(self, all_text: str, inv: ExtractedInvoice) -> list[dict]:
        """
        Fallback table extractor for government vouchers, employee bills, and single-item invoices.
        Populates line items so the review form table is never left completely blank when data exists.
        """
        items = []
        # Check if HRMS or Employee Bill table format
        if "hrms" in all_text.lower() or "annexure" in all_text.lower() or "employee bill" in all_text.lower():
            emp_name = inv.buyer_name.value if inv.buyer_name else "Beneficiary"
            total_val = float(inv.grand_total.value) if inv.grand_total and str(inv.grand_total.value).replace(".", "", 1).isdigit() else 0.0
            
            head_m = re.search(r"Head\s*of\s*Account[\s:=]*([0-9-]+)", all_text, re.I)
            head = head_m.group(1) if head_m else (inv.subcategory.value if inv.subcategory else "HRMS Bill")
            
            desc = f"{inv.subcategory.value if inv.subcategory else 'Bill Payment'} ({head}) - {emp_name}"
            items.append({
                "description": desc,
                "quantity": 1.0,
                "unit": "NOS",
                "rate": total_val,
                "amount": total_val,
                "taxable_value": total_val,
                "discount": 0.0,
                "cgst_rate": 0.0,
                "cgst_amount": 0.0,
                "sgst_rate": 0.0,
                "sgst_amount": 0.0,
                "igst_rate": 0.0,
                "igst_amount": 0.0,
            })
            return items

        # For commercial invoices with a detected grand total
        if inv.grand_total and str(inv.grand_total.value).strip():
            try:
                g_val = float(str(inv.grand_total.value).replace(",", ""))
                if g_val > 0:
                    desc = f"Supply / Service Charges"
                    if inv.category and inv.category.value:
                        desc = f"{inv.category.value} - {inv.subcategory.value if inv.subcategory else 'Services'}"
                    elif inv.vendor_name and inv.vendor_name.value:
                        desc = f"Invoice Item - {inv.vendor_name.value}"
                    
                    sub_val = float(str(inv.subtotal.value).replace(",", "")) if inv.subtotal else g_val
                    items.append({
                        "description": desc,
                        "quantity": 1.0,
                        "unit": "NOS",
                        "rate": sub_val,
                        "amount": sub_val,
                        "taxable_value": sub_val,
                        "discount": 0.0,
                        "cgst_rate": 0.0,
                        "cgst_amount": float(inv.cgst.value) if inv.cgst else 0.0,
                        "sgst_rate": 0.0,
                        "sgst_amount": float(inv.sgst.value) if inv.sgst else 0.0,
                        "igst_rate": 0.0,
                        "igst_amount": float(inv.igst.value) if inv.igst else 0.0,
                    })
            except (ValueError, TypeError):
                pass

        return items

    def generate_spatial_candidates(self, ocr_results: dict) -> list[SpatialCandidate]:
        """
        Extracts spatial candidates (invoice numbers, dates, GSTINs, totals, IFSC)
        paired with their nearest spatial key labels and coordinates.
        """
        candidates: list[SpatialCandidate] = []
        blocks = []
        for region_label, ocr_res in ocr_results.items():
            for b in ocr_res.text_blocks:
                blocks.append((b, region_label))

        def bbox_center(box):
            if len(box) == 4 and isinstance(box[0], (int, float)):
                return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            return (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0

        def calc_dist(b1, b2):
            c1 = bbox_center(b1)
            c2 = bbox_center(b2)
            return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5

        # 1. Invoice Number Candidates
        label_blocks = [b for b, r in blocks if any(k in b.text.lower() for k in ["invoice no", "inv no", "bill no", "reference no", "invoice #", "bill #", "invoice"])]
        for b, r in blocks:
            val = None
            m = self.INVOICE_NUM_PATTERN.search(b.text)
            if m:
                val = m.group(1).strip()
            elif label_blocks and not any(k in b.text.lower() for k in ["invoice no", "inv no", "bill no", "date", "gstin", "total", "amount"]):
                clean_t = b.text.strip()
                if re.match(r"^[A-Z0-9/_-]{2,30}$", clean_t, re.IGNORECASE) and not clean_t.lower().startswith("gst"):
                    val = clean_t

            if val:
                nearest_label = None
                label_box = None
                min_d = 9999.0
                for lb in label_blocks:
                    if lb.bbox and b.bbox and lb is not b:
                        d = calc_dist(b.bbox, lb.bbox)
                        if d < min_d:
                            min_d = d
                            nearest_label = lb.text.strip()
                            label_box = lb.to_xyxy()
                candidates.append(SpatialCandidate(
                    field_name="invoice_number",
                    value=val,
                    bbox=b.to_xyxy(),
                    nearby_label=nearest_label,
                    label_bbox=label_box,
                    distance_px=min_d if min_d < 9999 else 0.0,
                    confidence=b.confidence,
                    region=r,
                ))

        # 2. Date Candidates
        date_label_blocks = [b for b, r in blocks if any(k in b.text.lower() for k in ["date", "inv date", "bill date", "dated"])]
        for b, r in blocks:
            for pat in self.DATE_PATTERNS:
                m = pat.search(b.text)
                if m:
                    val = m.group(0).strip()
                    min_d = 9999.0
                    nearest_label = None
                    label_box = None
                    for lb in date_label_blocks:
                        if lb.bbox and b.bbox:
                            d = calc_dist(b.bbox, lb.bbox)
                            if d < min_d:
                                min_d = d
                                nearest_label = lb.text.strip()
                                label_box = lb.to_xyxy()
                    candidates.append(SpatialCandidate(
                        field_name="invoice_date",
                        value=val,
                        bbox=b.to_xyxy(),
                        nearby_label=nearest_label,
                        label_bbox=label_box,
                        distance_px=min_d if min_d < 9999 else 0.0,
                        confidence=b.confidence,
                        region=r,
                    ))
                    break

        # 3. GSTIN Candidates
        gst_label_blocks = [b for b, r in blocks if any(k in b.text.lower() for k in ["gstin", "gst no", "gstin/uin"])]
        for b, r in blocks:
            m = self.GSTIN_PATTERN.search(b.text)
            if m:
                val = m.group(0).strip()
                min_d = 9999.0
                nearest_label = None
                label_box = None
                for lb in gst_label_blocks:
                    if lb.bbox and b.bbox:
                        d = calc_dist(b.bbox, lb.bbox)
                        if d < min_d:
                            min_d = d
                            nearest_label = lb.text.strip()
                            label_box = lb.to_xyxy()
                candidates.append(SpatialCandidate(
                    field_name="vendor_gstin",
                    value=val,
                    bbox=b.to_xyxy(),
                    nearby_label=nearest_label,
                    label_bbox=label_box,
                    distance_px=min_d if min_d < 9999 else 0.0,
                    confidence=b.confidence,
                    region=r,
                ))

        # 4. Grand Total / Financial Candidates
        total_label_blocks = [b for b, r in blocks if any(k in b.text.lower() for k in ["grand total", "total amount", "total", "net amount"])]
        for b, r in blocks:
            if any(kw in b.text.lower() for kw in ["total", "amount", "rs", "inr", "₹"]):
                m = self.AMOUNT_PATTERN.search(b.text)
                if m:
                    val = m.group(1).strip().replace(",", "")
                    try:
                        if float(val) > 0:
                            min_d = 9999.0
                            nearest_label = None
                            label_box = None
                            for lb in total_label_blocks:
                                if lb.bbox and b.bbox:
                                    d = calc_dist(b.bbox, lb.bbox)
                                    if d < min_d:
                                        min_d = d
                                        nearest_label = lb.text.strip()
                                        label_box = lb.to_xyxy()
                            candidates.append(SpatialCandidate(
                                field_name="grand_total",
                                value=val,
                                bbox=b.to_xyxy(),
                                nearby_label=nearest_label,
                                label_bbox=label_box,
                                distance_px=min_d if min_d < 9999 else 0.0,
                                confidence=b.confidence,
                                region=r,
                            ))
                    except ValueError:
                        pass

        # 5. Bank / IFSC Candidates
        ifsc_label_blocks = [b for b, r in blocks if "ifsc" in b.text.lower()]
        for b, r in blocks:
            m = self.IFSC_PATTERN.search(b.text.upper())
            if m:
                val = m.group(0).strip()
                min_d = 9999.0
                nearest_label = None
                label_box = None
                for lb in ifsc_label_blocks:
                    if lb.bbox and b.bbox:
                        d = calc_dist(b.bbox, lb.bbox)
                        if d < min_d:
                            min_d = d
                            nearest_label = lb.text.strip()
                            label_box = lb.to_xyxy()
                candidates.append(SpatialCandidate(
                    field_name="ifsc_code",
                    value=val,
                    bbox=b.to_xyxy(),
                    nearby_label=nearest_label,
                    label_bbox=label_box,
                    distance_px=min_d if min_d < 9999 else 0.0,
                    confidence=b.confidence,
                    region=r,
                ))

        # Deduplicate candidates across regions by (field_name, normalized_value)
        # Retains the best candidate with closer nearby label distance or higher confidence
        import math
        deduped: dict[tuple[str, str], SpatialCandidate] = {}
        for cand in candidates:
            norm_val = str(cand.value).strip().upper().replace(" ", "")
            key = (cand.field_name, norm_val)
            if key not in deduped:
                deduped[key] = cand
            else:
                existing = deduped[key]
                existing_prox = math.exp(-existing.distance_px / 60.0) if existing.nearby_label else 0.0
                cand_prox = math.exp(-cand.distance_px / 60.0) if cand.nearby_label else 0.0
                existing_score = existing.confidence * 0.6 + existing_prox * 0.4
                cand_score = cand.confidence * 0.6 + cand_prox * 0.4
                if cand_score > existing_score:
                    deduped[key] = cand

        return list(deduped.values())
