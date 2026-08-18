"""
Stage 4a: LayoutLMv3 document understanding.

Uses Microsoft's LayoutLMv3 model (fine-tuned on invoices) to understand
the spatial relationship between text blocks and extract structured fields.

Falls back to regex-based heuristic extraction if the model is not available.
"""

import re
import numpy as np
from pathlib import Path
from loguru import logger
from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class ExtractedField:
    value: str
    confidence: float
    source: str   # "layoutlm", "heuristic", or "llm"


@dataclass
class ExtractedInvoice:
    # Header
    invoice_number: Optional[ExtractedField] = None
    invoice_date: Optional[ExtractedField] = None
    due_date: Optional[ExtractedField] = None
    po_number: Optional[ExtractedField] = None
    place_of_supply: Optional[ExtractedField] = None

    # Vendor
    vendor_name: Optional[ExtractedField] = None
    vendor_address: Optional[ExtractedField] = None
    vendor_gstin: Optional[ExtractedField] = None
    vendor_pan: Optional[ExtractedField] = None
    vendor_email: Optional[ExtractedField] = None
    vendor_phone: Optional[ExtractedField] = None

    # Buyer
    buyer_name: Optional[ExtractedField] = None
    buyer_address: Optional[ExtractedField] = None
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

    # Meta
    overall_confidence: float = 0.0
    low_confidence_fields: list[str] = field(default_factory=list)


class LayoutLMExtractor:
    """
    LayoutLMv3-based invoice field extractor.
    Loads fine-tuned model if available; falls back to heuristic NLP extraction.
    """

    GSTIN_PATTERN = re.compile(
        r"\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}\b"
    )
    PAN_PATTERN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]{1}\b")
    IFSC_PATTERN = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")
    DATE_PATTERNS = [
        re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b"),
        re.compile(r"\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+(\d{4})\b", re.IGNORECASE),
    ]
    AMOUNT_PATTERN = re.compile(r"(?:₹|Rs\.?|INR|USD|\$)?\s*(\d[\d,]*(?:\.\d{1,2})?)")
    EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b")
    PHONE_PATTERN = re.compile(r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b")
    INVOICE_NUM_PATTERN = re.compile(
        r"(?:invoice\s*(?:no|number|#)|inv\.?\s*(?:no\.?|#))[\s:]*([A-Z0-9/_-]{3,20})",
        re.IGNORECASE
    )

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
        for region_label, ocr_result in ocr_results.items():
            for block in ocr_result.text_blocks:
                text_clean = block.text.strip()
                if not text_clean:
                    continue
                all_words.append(text_clean)
                # Normalise bbox to 0-1000 for LayoutLMv3
                bbox = block.bbox
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                w, h = image.size
                norm_bbox = [
                    max(0, min(1000, int(1000 * x1 / max(1, w)))),
                    max(0, min(1000, int(1000 * y1 / max(1, h)))),
                    max(0, min(1000, int(1000 * x2 / max(1, w)))),
                    max(0, min(1000, int(1000 * y2 / max(1, h)))),
                ]
                all_boxes.append(norm_bbox)

        # If no words detected, fall back to heuristic
        if not all_words:
            return self._extract_heuristic(ocr_results)

        try:
            encoding = self.processor(
                image,
                all_words,
                boxes=all_boxes,
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

            # Map BPE subword tokens back to original words using word_ids
            word_ids = encoding.word_ids(0) if hasattr(encoding, "word_ids") else None
            word_labels = ["O"] * len(all_words)
            word_confs = [0.0] * len(all_words)

            if word_ids:
                for token_idx, word_idx in enumerate(word_ids):
                    if word_idx is None or word_idx >= len(all_words):
                        continue
                    lbl = token_labels[token_idx] if token_idx < len(token_labels) else "O"
                    cnf = float(token_confs[token_idx]) if token_idx < len(token_confs) else 0.0
                    if lbl != "O":
                        if word_labels[word_idx] == "O" or cnf > word_confs[word_idx]:
                            word_labels[word_idx] = lbl
                            word_confs[word_idx] = cnf
                    elif word_labels[word_idx] == "O":
                        word_confs[word_idx] = cnf
            else:
                for i in range(min(len(all_words), len(token_labels))):
                    word_labels[i] = token_labels[i]
                    word_confs[i] = float(token_confs[i])

            # Group tokens by label to build field values
            fields = self._group_token_labels(all_words, word_labels, word_confs)
            layoutlm_inv = self._fields_to_invoice(fields, source="layoutlm")
        except Exception as e:
            logger.warning(f"LayoutLM inference warning: {e} — relying on heuristic extraction")
            layoutlm_inv = ExtractedInvoice()

        # FUSION: Merge with heuristic regex & geometric extraction
        heuristic_inv = self._extract_heuristic(ocr_results)
        return self._merge_invoices(layoutlm_inv, heuristic_inv)

    def _merge_invoices(self, primary: ExtractedInvoice, secondary: ExtractedInvoice) -> ExtractedInvoice:
        """Merge primary (LayoutLM) with secondary (Heuristic/Regex) extraction, picking higher confidence."""
        merged = ExtractedInvoice()
        all_field_names = [
            "invoice_number", "invoice_date", "due_date", "po_number", "place_of_supply",
            "vendor_name", "vendor_address", "vendor_gstin", "vendor_pan", "vendor_email", "vendor_phone",
            "buyer_name", "buyer_address", "buyer_gstin", "buyer_phone", "sls_code",
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
                # Pick the higher confidence prediction
                chosen = p_val if p_val.confidence >= s_val.confidence else s_val
                setattr(merged, fname, chosen)
            elif p_valid and p_val.confidence >= 0.35:
                setattr(merged, fname, p_val)
            elif s_valid:
                setattr(merged, fname, s_val)

        # Line items: use primary if available and non-empty, else secondary
        if primary.line_items and len(primary.line_items) > 0:
            merged.line_items = primary.line_items
        else:
            merged.line_items = secondary.line_items

        # Overall confidence calculation
        core_fields = [
            merged.invoice_number, merged.invoice_date, merged.vendor_name,
            merged.buyer_name, merged.grand_total, merged.subtotal,
            merged.vendor_gstin
        ]
        filled = [f for f in core_fields if f is not None and getattr(f, "value", None)]
        merged.overall_confidence = (
            sum(f.confidence for f in filled) / len(filled) if filled else 0.0
        )
        return merged

    def _group_token_labels(self, words, labels, confidences) -> dict:
        """Convert BIO token labels to grouped field values, ignoring low-confidence noise."""
        fields = {}
        current_label = None
        current_words = []
        current_confs = []

        for word, label, conf in zip(words, labels, confidences):
            # Ignore ultra-low confidence raw predictions
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
            "VENDOR_NAME": "vendor_name",
            "BILLER_NAME": "vendor_name",
            "VENDOR_ADDRESS": "vendor_address",
            "BILLER_ADDRESS": "vendor_address",
            "VENDOR_GSTIN": "vendor_gstin",
            "GST": "vendor_gstin",
            "BUYER_NAME": "buyer_name",
            "BUYER_ADDRESS": "buyer_address",
            "BUYER_GSTIN": "buyer_gstin",
            "SUBTOTAL": "subtotal",
            "TAX_AMOUNT": "tax_amount",
            "CGST": "cgst",
            "SGST": "sgst",
            "IGST": "igst",
            "GRAND_TOTAL": "grand_total",
            "TOTAL": "grand_total",
            "BANK_NAME": "bank_name",
            "ACCOUNT_NUMBER": "account_number",
            "IFSC_CODE": "ifsc_code",
        }
        for label_key, field_name in mapping.items():
            if label_key in fields:
                if getattr(inv, field_name) is None:
                    value, conf = fields[label_key]
                    setattr(inv, field_name, ExtractedField(value=value, confidence=conf, source=source))
        return inv

    def _extract_heuristic(self, ocr_results: dict) -> ExtractedInvoice:
        """
        Regex + keyword heuristic extraction.
        Used when LayoutLMv3 model is not available.
        """
        inv = ExtractedInvoice()
        region_texts = {k: v.full_text for k, v in ocr_results.items()}
        all_text = " ".join(region_texts.values())

        # --- Header region ---
        header_text = region_texts.get("header", "")
        search_scope = (header_text + "\n" + all_text).strip()

        inv_num = self.INVOICE_NUM_PATTERN.search(search_scope)
        if inv_num:
            inv.invoice_number = ExtractedField(inv_num.group(1).strip(), 0.75, "heuristic")

        for pat in self.DATE_PATTERNS:
            m = pat.search(search_scope)
            if m:
                inv.invoice_date = ExtractedField(m.group(0).strip(), 0.70, "heuristic")
                break

        pos_match = re.search(r"(?:place\s*of\s*supply|pos)[\s:]*([A-Za-z0-9\s-]+)", search_scope, re.IGNORECASE)
        if pos_match:
            pos_val = pos_match.group(1).split("\n")[0].strip()
            if len(pos_val) < 40:
                inv.place_of_supply = ExtractedField(pos_val, 0.75, "heuristic")

        due_match = re.search(r"(?:due\s*date)[\s:]*([0-9A-Za-z/ -]+)", search_scope, re.IGNORECASE)
        if due_match:
            for pat in self.DATE_PATTERNS:
                dm = pat.search(due_match.group(0))
                if dm:
                    inv.due_date = ExtractedField(dm.group(0).strip(), 0.70, "heuristic")
                    break

        # --- Vendor block ---
        vendor_text = region_texts.get("vendor_block", "")
        if vendor_text:
            lines = [l.strip() for l in vendor_text.split("\n") if l.strip()]
            if lines:
                inv.vendor_name = ExtractedField(lines[0], 0.68, "heuristic")
            if len(lines) > 1:
                inv.vendor_address = ExtractedField(" ".join(lines[1:4]), 0.60, "heuristic")
        elif header_text:
            lines = [l.strip() for l in header_text.split("\n") if l.strip() and not any(k in l.lower() for k in ["invoice", "tax", "gstin", "date", "bill"])]
            if lines:
                inv.vendor_name = ExtractedField(lines[0], 0.60, "heuristic")

        all_gstins = self.GSTIN_PATTERN.findall(all_text)
        if all_gstins:
            inv.vendor_gstin = ExtractedField(all_gstins[0], 0.92, "heuristic")
            if len(all_gstins) > 1:
                inv.buyer_gstin = ExtractedField(all_gstins[1], 0.90, "heuristic")

        pan = self.PAN_PATTERN.search(vendor_text + " " + all_text)
        if pan:
            inv.vendor_pan = ExtractedField(pan.group(0), 0.88, "heuristic")

        email = self.EMAIL_PATTERN.search(all_text)
        if email:
            inv.vendor_email = ExtractedField(email.group(0), 0.92, "heuristic")

        phone = self.PHONE_PATTERN.search(all_text)
        if phone:
            inv.vendor_phone = ExtractedField(phone.group(0), 0.88, "heuristic")

        # --- Buyer block ---
        buyer_text = region_texts.get("buyer_block", "")
        if buyer_text:
            lines = [l.strip() for l in buyer_text.split("\n") if l.strip()]
            if lines:
                inv.buyer_name = ExtractedField(lines[0], 0.65, "heuristic")
            if len(lines) > 1:
                inv.buyer_address = ExtractedField(" ".join(lines[1:4]), 0.58, "heuristic")

        # --- Totals block ---
        totals_text = region_texts.get("totals_block", "") + " " + region_texts.get("tax_block", "") + " " + all_text
        grand_total = self._find_amount_near_keyword(
            totals_text, ["grand total", "total amount", "amount due", "net amount", "invoice total", "total value", "total"]
        )
        if grand_total:
            inv.grand_total = ExtractedField(grand_total, 0.80, "heuristic")

        subtotal = self._find_amount_near_keyword(
            totals_text, ["subtotal", "sub-total", "taxable value", "taxable amount", "total taxable", "net total"]
        )
        if subtotal:
            inv.subtotal = ExtractedField(subtotal, 0.78, "heuristic")

        round_off = self._find_amount_near_keyword(
            totals_text, ["round off", "roundoff", "rounding"]
        )
        if round_off:
            inv.round_off = ExtractedField(round_off, 0.70, "heuristic")

        words_match = re.search(r"(?:amount\s*in\s*words|in\s*words|rupees)[\s:]*([A-Za-z\s/-]+(?:only)?)", totals_text, re.IGNORECASE)
        if words_match:
            words_val = words_match.group(1).split("\n")[0].strip()
            if len(words_val) > 5 and len(words_val) < 150:
                inv.amount_in_words = ExtractedField(words_val, 0.80, "heuristic")

        # --- Tax block ---
        tax_text = region_texts.get("tax_block", totals_text)
        cgst = self._find_amount_near_keyword(tax_text, ["cgst"])
        if cgst:
            inv.cgst = ExtractedField(cgst, 0.78, "heuristic")

        sgst = self._find_amount_near_keyword(tax_text, ["sgst"])
        if sgst:
            inv.sgst = ExtractedField(sgst, 0.78, "heuristic")

        igst = self._find_amount_near_keyword(tax_text, ["igst"])
        if igst:
            inv.igst = ExtractedField(igst, 0.78, "heuristic")

        igst = self._find_amount_near_keyword(tax_text, ["igst"])
        if igst:
            inv.igst = ExtractedField(igst, 0.78, "heuristic")

        tax_amount = self._find_amount_near_keyword(
            tax_text, ["total tax", "tax total", "gst", "vat"]
        )
        if tax_amount:
            inv.tax_amount = ExtractedField(tax_amount, 0.70, "heuristic")

        # --- Currency ---
        if "₹" in all_text or "INR" in all_text or "Rs" in all_text:
            inv.currency = ExtractedField("INR", 0.95, "heuristic")
        elif "$" in all_text or "USD" in all_text:
            inv.currency = ExtractedField("USD", 0.95, "heuristic")
        else:
            inv.currency = ExtractedField("INR", 0.50, "heuristic")

        # --- Payment / Bank block ---
        pay_text = region_texts.get("payment_terms", "")
        ifsc = self.IFSC_PATTERN.search(pay_text + " " + all_text)
        if ifsc:
            inv.ifsc_code = ExtractedField(ifsc.group(0), 0.90, "heuristic")

        acc_match = re.search(r"(?:a/c|account|acct|acc\s*no)[\s:#.]*(\d{9,18})", pay_text + " " + all_text, re.IGNORECASE)
        if acc_match:
            inv.account_number = ExtractedField(acc_match.group(1), 0.82, "heuristic")

        bank_match = re.search(r"(?:bank\s*name|bank)[\s:]*([A-Za-z\s&]+(?:bank|ltd|limited)?)", pay_text, re.IGNORECASE)
        if bank_match:
            b_val = bank_match.group(1).split("\n")[0].strip()
            if len(b_val) < 40:
                inv.bank_name = ExtractedField(b_val, 0.75, "heuristic")

        branch_match = re.search(r"(?:branch\s*name|branch)[\s:]*([A-Za-z0-9\s,-]+)", pay_text, re.IGNORECASE)
        if branch_match:
            br_val = branch_match.group(1).split("\n")[0].strip()
            if len(br_val) < 40:
                inv.branch_name = ExtractedField(br_val, 0.70, "heuristic")

        # --- Line items (simplified) ---
        inv.line_items = self._extract_line_items(region_texts.get("line_items", ""))

        # --- Overall confidence ---
        all_fields = [
            inv.invoice_number, inv.invoice_date, inv.vendor_name,
            inv.buyer_name, inv.grand_total, inv.subtotal,
        ]
        filled = [f for f in all_fields if f is not None]
        inv.overall_confidence = (
            sum(f.confidence for f in filled) / len(filled) if filled else 0.0
        )

        return inv

    def _find_amount_near_keyword(self, text: str, keywords: list[str]) -> Optional[str]:
        """Search for a currency amount near a keyword."""
        text_lower = text.lower()
        for kw in keywords:
            idx = text_lower.find(kw)
            if idx == -1:
                continue
            # Look at the 80 characters after the keyword
            snippet = text[idx:idx + 80]
            match = self.AMOUNT_PATTERN.search(snippet)
            if match:
                # Keep the raw value including decimal point; only strip commas
                raw = match.group(1).replace(",", "")
                return raw
        return None

    def _extract_line_items(self, line_items_text: str) -> list[dict]:
        """
        Simple line item extractor.
        Looks for rows with: description, qty, rate, amount pattern.
        """
        if not line_items_text:
            return []

        items = []
        lines = [l.strip() for l in line_items_text.split("\n") if l.strip()]

        for line in lines:
            # Skip header rows
            if any(h in line.lower() for h in ["description", "item", "s.no", "sr.", "qty", "rate", "amount"]):
                continue
            # A line item usually ends with a number (amount)
            amounts = self.AMOUNT_PATTERN.findall(line)
            if len(amounts) >= 1:
                amount = amounts[-1].replace(",", "")
                qty = amounts[0].replace(",", "") if len(amounts) >= 2 else "1"
                rate = amounts[-2].replace(",", "") if len(amounts) >= 3 else amount
                # Extract description (text before the numbers)
                desc_match = re.match(r"^([\w\s,.\-/()]+?)(?:\s+\d)", line)
                desc = desc_match.group(1).strip() if desc_match else line[:50]
                if desc and amount:
                    items.append({
                        "description": desc,
                        "quantity": qty,
                        "rate": rate,
                        "amount": amount,
                    })

        return items
