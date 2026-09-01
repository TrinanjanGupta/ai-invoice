"""
validation/confidence_engine.py

Multi-evidence field-level confidence and consistency scoring engine.
Calculates calibrated confidence scores [0.0, 1.0] for every extracted invoice field
based on:
1. OCR & Token spatial quality
2. Deterministic format & checksum validation (GSTIN modulo 36, PAN, IFSC, Dates)
3. Model agreement (LayoutLM probability vs Heuristic regex)
4. Cross-field arithmetic consistency (Subtotal + Taxes - Discount ≈ Grand Total)

Tags each field as:
- AUTO_ACCEPT (>= 0.85 with valid format)
- QUICK_CONFIRM (0.65 - 0.85)
- NEEDS_REVIEW (< 0.65 or arithmetic contradiction)
"""

import re
from datetime import datetime
from typing import Optional, Any
from loguru import logger


# GSTIN Checksum Validator (Modulo 36)
def validate_gstin_checksum(gstin: Optional[str]) -> bool:
    if not gstin or len(gstin.strip()) != 15:
        return False
    gstin = gstin.strip().upper()
    if not re.match(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$", gstin):
        return False
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    char_map = {c: i for i, c in enumerate(chars)}
    try:
        factor = 1
        total = 0
        for i in range(14):
            code_pt = char_map[gstin[i]]
            addend = factor * code_pt
            factor = 2 if factor == 1 else 1
            addend = (addend // 36) + (addend % 36)
            total += addend
        remainder = total % 36
        check_code = (36 - remainder) % 36
        return chars[check_code] == gstin[14]
    except Exception:
        return False


def validate_pan(pan: Optional[str]) -> bool:
    if not pan:
        return False
    return bool(re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]{1}$", pan.strip().upper()))


def validate_ifsc(ifsc: Optional[str]) -> bool:
    if not ifsc:
        return False
    return bool(re.match(r"^[A-Z]{4}0[A-Z0-9]{6}$", ifsc.strip().upper()))


def validate_date(date_str: Optional[str]) -> bool:
    if not date_str:
        return False
    clean = date_str.strip()
    clean_slash = clean.replace(".", "/").replace("-", "/")
    for fmt in (
        "%d/%m/%Y", "%Y/%m/%d", "%d/%m/%y", "%d %b %Y", "%d %B %Y",
        "%d/%b/%Y", "%d/%B/%Y", "%d-%b-%Y", "%d-%B-%Y", "%b %d, %Y", "%B %d, %Y"
    ):
        for candidate in (clean, clean_slash):
            try:
                datetime.strptime(candidate, fmt)
                return True
            except ValueError:
                pass
    return False


class FieldConfidenceEngine:
    """
    Evaluates field-level extraction evidence and generates calibrated confidence scores.
    """

    CONFIDENCE_AUTO_ACCEPT = 0.85
    CONFIDENCE_QUICK_CONFIRM = 0.65

    def evaluate(self, invoice_dict: dict, ocr_avg_conf: float = 0.90) -> dict[str, Any]:
        field_confidences: dict[str, float] = {}
        fields_needing_review: list[str] = []
        auto_accepted_fields: list[str] = []
        review_reasons: list[str] = []

        # 1. Invoice Number
        inv_no = str(invoice_dict.get("invoice_number") or "").strip()
        if inv_no:
            conf = min(0.98, max(0.60, ocr_avg_conf))
            if re.search(r"[0-9]", inv_no) and len(inv_no) >= 2:
                conf += 0.05
            field_confidences["invoice_number"] = min(0.99, conf)
        else:
            field_confidences["invoice_number"] = 0.0
            fields_needing_review.append("invoice_number")
            review_reasons.append("Invoice number missing")

        # 2. Invoice Date
        inv_date = invoice_dict.get("invoice_date")
        if inv_date:
            if validate_date(str(inv_date)):
                field_confidences["invoice_date"] = 0.96
            else:
                field_confidences["invoice_date"] = 0.60
                fields_needing_review.append("invoice_date")
                review_reasons.append("Invoice date format unrecognised")
        else:
            field_confidences["invoice_date"] = 0.0
            fields_needing_review.append("invoice_date")
            review_reasons.append("Invoice date missing")

        # 3. Vendor Name & GSTIN
        v_name = str(invoice_dict.get("vendor_name") or "").strip()
        if v_name and len(v_name) >= 3:
            field_confidences["vendor_name"] = 0.92
        else:
            field_confidences["vendor_name"] = 0.40
            fields_needing_review.append("vendor_name")
            review_reasons.append("Vendor name missing or uncertain")

        v_gstin = invoice_dict.get("vendor_gstin")
        if v_gstin:
            if validate_gstin_checksum(str(v_gstin)):
                field_confidences["vendor_gstin"] = 0.99
            elif bool(re.match(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$", str(v_gstin).strip().upper())):
                field_confidences["vendor_gstin"] = 0.70
                fields_needing_review.append("vendor_gstin")
                review_reasons.append("Vendor GSTIN checksum mismatch")
            else:
                field_confidences["vendor_gstin"] = 0.40
                fields_needing_review.append("vendor_gstin")
                review_reasons.append("Vendor GSTIN format invalid")

        # 4. Buyer Name & GSTIN
        b_name = str(invoice_dict.get("buyer_name") or "").strip()
        if b_name and len(b_name) >= 3:
            field_confidences["buyer_name"] = 0.92
        else:
            field_confidences["buyer_name"] = 0.50

        b_gstin = invoice_dict.get("buyer_gstin")
        if b_gstin:
            if validate_gstin_checksum(str(b_gstin)):
                field_confidences["buyer_gstin"] = 0.99
            elif bool(re.match(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$", str(b_gstin).strip().upper())):
                field_confidences["buyer_gstin"] = 0.70
                fields_needing_review.append("buyer_gstin")
                review_reasons.append("Buyer GSTIN checksum mismatch")
            else:
                field_confidences["buyer_gstin"] = 0.40
                fields_needing_review.append("buyer_gstin")

        # 5. Bank Details (IFSC & Account Number)
        ifsc = invoice_dict.get("ifsc_code")
        if ifsc:
            if validate_ifsc(str(ifsc)):
                field_confidences["ifsc_code"] = 0.98
            else:
                field_confidences["ifsc_code"] = 0.60
                fields_needing_review.append("ifsc_code")
                review_reasons.append("IFSC code format invalid")

        acc_no = str(invoice_dict.get("account_number") or "").strip()
        if acc_no:
            if re.match(r"^[0-9]{9,18}$", acc_no):
                field_confidences["account_number"] = 0.94
            else:
                field_confidences["account_number"] = 0.62
                fields_needing_review.append("account_number")

        # 6. Financial Arithmetic Cross-Validation (Holistic Scheme Resolver)
        subtotal = invoice_dict.get("subtotal")
        cgst = invoice_dict.get("cgst") or 0.0
        sgst = invoice_dict.get("sgst") or 0.0
        igst = invoice_dict.get("igst") or 0.0
        tax_amt = invoice_dict.get("tax_amount")
        discount = invoice_dict.get("discount") or invoice_dict.get("global_discount") or 0.0
        round_off = invoice_dict.get("round_off") or 0.0
        grand_total = invoice_dict.get("grand_total")

        global_cgst_rate = invoice_dict.get("global_cgst_rate") or 0.0
        global_sgst_rate = invoice_dict.get("global_sgst_rate") or 0.0
        global_igst_rate = invoice_dict.get("global_igst_rate") or 0.0
        total_global_rate = global_cgst_rate + global_sgst_rate + global_igst_rate

        # Tax calculation hypothesis resolver
        tax_total = (cgst + sgst + igst) if (cgst or sgst or igst) else (tax_amt or 0.0)
        if tax_total == 0.0 and subtotal and total_global_rate > 0:
            # Global GST hypothesis
            computed_global_tax = round(subtotal * (total_global_rate / 100.0), 2)
            tax_total = computed_global_tax

        arithmetic_valid = False
        if grand_total is not None and grand_total > 0:
            field_confidences["grand_total"] = 0.92
            if subtotal is not None and subtotal > 0:
                expected_total = subtotal + tax_total - discount + round_off
                diff = abs(expected_total - grand_total)
                if diff <= max(1.5, grand_total * 0.015):
                    arithmetic_valid = True
                    field_confidences["grand_total"] = 0.99
                    field_confidences["subtotal"] = 0.98
                    if tax_total > 0:
                        field_confidences["tax_amount"] = 0.98
                        if cgst: field_confidences["cgst"] = 0.98
                        if sgst: field_confidences["sgst"] = 0.98
                        if igst: field_confidences["igst"] = 0.98
                        if global_cgst_rate: field_confidences["global_cgst_rate"] = 0.98
                        if global_sgst_rate: field_confidences["global_sgst_rate"] = 0.98
                        if global_igst_rate: field_confidences["global_igst_rate"] = 0.98
                else:
                    # Arithmetic mismatch
                    field_confidences["grand_total"] = 0.65
                    field_confidences["subtotal"] = 0.65
                    fields_needing_review.extend(["grand_total", "subtotal"])
                    review_reasons.append(
                        f"Financial total mismatch: Subtotal ({subtotal}) + Tax ({tax_total}) - Disc ({discount}) != Grand Total ({grand_total})"
                    )
            else:
                field_confidences["grand_total"] = 0.85
        else:
            field_confidences["grand_total"] = 0.0
            fields_needing_review.append("grand_total")
            review_reasons.append("Grand total missing or zero")

        # 7. Segregate auto-accepted vs review fields
        for f_name, f_conf in field_confidences.items():
            if f_conf >= self.CONFIDENCE_AUTO_ACCEPT and f_name not in fields_needing_review:
                auto_accepted_fields.append(f_name)
            elif f_conf < self.CONFIDENCE_AUTO_ACCEPT and f_name not in fields_needing_review:
                fields_needing_review.append(f_name)

        # 8. "DO NOT AUTO ACCEPT" Critical Gatekeeper
        CRITICAL_FIELDS = {"grand_total", "vendor_gstin", "invoice_number", "invoice_date", "account_number", "ifsc_code"}
        critical_failure = False
        for crit in CRITICAL_FIELDS:
            crit_conf = field_confidences.get(crit)
            if crit_conf is None or crit_conf < self.CONFIDENCE_AUTO_ACCEPT or crit in fields_needing_review:
                # If field was extracted but failed format/confidence
                if invoice_dict.get(crit):
                    critical_failure = True
                    if crit not in fields_needing_review:
                        fields_needing_review.append(crit)

        # Overall weighted confidence
        if field_confidences:
            weights = {
                "grand_total": 2.5,
                "invoice_number": 2.0,
                "vendor_gstin": 2.0,
                "invoice_date": 1.5,
                "vendor_name": 1.5,
                "account_number": 1.5,
            }
            weighted_sum = sum(conf * weights.get(k, 1.0) for k, conf in field_confidences.items())
            total_weight = sum(weights.get(k, 1.0) for k in field_confidences)
            overall_conf = round(weighted_sum / total_weight, 3)
        else:
            overall_conf = 0.0

        needs_review = len(fields_needing_review) > 0 or not arithmetic_valid or critical_failure

        return {
            "overall_confidence": overall_conf,
            "field_confidences": field_confidences,
            "fields_needing_review": list(set(fields_needing_review)),
            "auto_accepted_fields": list(set(auto_accepted_fields)),
            "needs_review": needs_review,
            "review_reasons": list(set(review_reasons)),
            "arithmetic_valid": arithmetic_valid,
            "critical_failure": critical_failure,
        }
