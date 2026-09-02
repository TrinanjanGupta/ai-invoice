"""
tests/test_numeric_recognizer.py

Unit tests for domain-specialized numeric and structured field recognition.
"""

import pytest
from ocr.numeric_recognizer import NumericRecognizer


def test_numeric_recognizer_amounts():
    rec = NumericRecognizer()
    # Currency with optical error: "Rs. 1,56O.50" -> "1560.50"
    res = rec.recognize_numeric("Rs. 1,56O.50", field_type="grand_total")
    assert len(res) > 0
    assert res[0][0] == "1560.50"
    assert res[0][1] >= 0.80

    # Subtotal with 'I' confusion: "I200.00" -> "1200.00"
    res_sub = rec.recognize_numeric("I200.00", field_type="subtotal")
    assert len(res_sub) > 0
    assert res_sub[0][0] == "1200.00"


def test_numeric_recognizer_quantities():
    rec = NumericRecognizer()
    # Integer quantity with 'O': "2O" -> "20"
    res_qty = rec.recognize_numeric("2O", field_type="quantity", allow_decimal=False)
    assert len(res_qty) > 0
    assert res_qty[0][0] == "20"


def test_numeric_recognizer_structured_fields():
    rec = NumericRecognizer()
    # GSTIN
    res_gstin = rec.recognize_numeric("27AABCU9603R1ZN", field_type="vendor_gstin")
    assert len(res_gstin) > 0
    assert res_gstin[0][0] == "27AABCU9603R1ZN"

    # IFSC
    res_ifsc = rec.recognize_numeric("SBINOOOO123", field_type="ifsc_code")
    assert len(res_ifsc) > 0
    assert res_ifsc[0][0] == "SBIN0000123"

    # Date
    res_date = rec.recognize_numeric("I5/O7/2O26", field_type="invoice_date")
    assert len(res_date) > 0
    assert res_date[0][0] == "15/07/2026"
