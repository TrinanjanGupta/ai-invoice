"""
tests/test_char_confusion.py

Unit tests for character confusion corrections across numeric, GSTIN, IFSC, date, and phone fields.
"""

import pytest
from validation.char_confusion import (
    correct_numeric_field,
    correct_gstin,
    correct_ifsc,
    correct_date,
    correct_phone,
    verify_gstin_checksum,
)


def test_numeric_field_corrections():
    # "15G" -> "156"
    cands = correct_numeric_field("15G", allow_decimal=True)
    assert len(cands) > 0
    assert cands[0][0] == "156"

    # "12O.5O" -> "120.50"
    cands2 = correct_numeric_field("12O.5O", allow_decimal=True)
    assert len(cands2) > 0
    assert cands2[0][0] == "120.50"

    # "I56" -> "156"
    cands3 = correct_numeric_field("I56", allow_decimal=False)
    assert len(cands3) > 0
    assert cands3[0][0] == "156"

    # Invalid non-numeric
    cands_inv = correct_numeric_field("XYZ##")
    assert len(cands_inv) == 0


def test_gstin_corrections():
    # Valid GSTIN should pass with 0.99 confidence
    valid_gstin = "27AABCU9603R1ZN"
    assert verify_gstin_checksum(valid_gstin)
    res = correct_gstin(valid_gstin)
    assert len(res) > 0
    assert res[0][0] == valid_gstin
    assert res[0][1] == 0.99

    # Confused GSTIN: 'O' instead of '0' in state code or PAN numbers: "27AABCU96O3R1ZN"
    confused = "27AABCU96O3R1ZN"
    corrected = correct_gstin(confused)
    assert len(corrected) > 0
    assert corrected[0][0] == valid_gstin
    assert verify_gstin_checksum(corrected[0][0])


def test_ifsc_corrections():
    # "SBINOOOO123" -> "SBIN0000123"
    cands = correct_ifsc("SBINOOOO123")
    assert len(cands) > 0
    assert cands[0][0] == "SBIN0000123"

    # Valid IFSC
    valid = "HDFC0001234"
    cands_valid = correct_ifsc(valid)
    assert len(cands_valid) > 0
    assert cands_valid[0][0] == valid


def test_date_corrections():
    # "I5/O7/2O26" -> "15/07/2026"
    cands = correct_date("I5/O7/2O26")
    assert len(cands) > 0
    assert cands[0][0] == "15/07/2026"

    # "22-Dec-25" -> should handle dash delimiters
    cands_slash = correct_date("15-07-26")
    assert len(cands_slash) > 0
    assert cands_slash[0][0] == "15/07/2026"


def test_phone_corrections():
    # "98765432IO" -> "9876543210"
    cands = correct_phone("98765432IO")
    assert len(cands) > 0
    assert cands[0][0] == "9876543210"
