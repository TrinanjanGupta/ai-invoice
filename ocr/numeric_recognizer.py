"""
ocr/numeric_recognizer.py

Domain-specialized digit and numeric recognizer for handwritten invoice financial fields:
- Quantities, Units, Rates, Line Amounts
- Subtotal, CGST, SGST, IGST, Round Off, Grand Total
- Account Numbers, HSN/SAC Codes, Invoice Numbers

Applies strict structural and field constraints to filter out character noise.
"""

from __future__ import annotations
import re
import cv2
import numpy as np
from typing import Optional
from loguru import logger

from validation.char_confusion import (
    correct_numeric_field,
    correct_gstin,
    correct_ifsc,
    correct_date,
    correct_phone,
)


class NumericRecognizer:
    """
    Specialized recognizer for handwritten digits and structured alphanumeric fields.
    """

    def recognize_numeric(
        self,
        raw_text: str,
        field_type: str = "amount",
        allow_decimal: bool = True,
    ) -> list[tuple[str, float]]:
        """
        Parses and corrects raw OCR text against field-type requirements.
        field_type: "amount" | "quantity" | "rate" | "gstin" | "ifsc" | "date" | "phone" | "account_number" | "invoice_number"
        """
        if not raw_text or not str(raw_text).strip():
            return []

        clean_raw = str(raw_text).strip()

        if field_type in ("amount", "subtotal", "grand_total", "tax_amount", "rate", "discount", "round_off"):
            return correct_numeric_field(clean_raw, allow_decimal=True)

        elif field_type in ("quantity", "qty", "hsn_code", "account_number"):
            return correct_numeric_field(clean_raw, allow_decimal=False)

        elif field_type in ("gstin", "vendor_gstin", "buyer_gstin"):
            return correct_gstin(clean_raw)

        elif field_type in ("ifsc", "ifsc_code"):
            return correct_ifsc(clean_raw)

        elif field_type in ("date", "invoice_date"):
            return correct_date(clean_raw)

        elif field_type in ("phone", "mobile"):
            return correct_phone(clean_raw)

        elif field_type in ("invoice_number", "inv_no"):
            # Invoice numbers can be alphanumeric e.g. "INV/2026/043" or "43"
            # Strip outer noise but preserve letters and slashes
            clean_inv = re.sub(r"[^\w/\-]", "", clean_raw).strip()
            if clean_inv:
                return [(clean_inv, 0.90)]
            return []

        # Default numeric fallback
        return correct_numeric_field(clean_raw, allow_decimal=allow_decimal)
