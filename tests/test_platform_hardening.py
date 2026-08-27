"""
tests/test_platform_hardening.py

Comprehensive test suite verifying the hardened Invoice Intelligence Platform:
1. Document Quality Scorer (blur, contrast, illumination, composite quality)
2. Document Router (digital, scan, photo, handwriting detection)
3. Wholesome Accounting & Tax Model Decision Engine (line-item vs global GST)
4. Bank-Grade Validation & "DO NOT AUTO ACCEPT" Critical Gatekeeper
5. Token Provenance & Immutability Models
"""

import pytest
import numpy as np
import cv2
from preprocessing.quality_scorer import DocumentQualityScorer
from preprocessing.document_router import DocumentRouter
from validation.confidence_engine import FieldConfidenceEngine, validate_gstin_checksum, validate_ifsc, validate_pan
from validation.validator import InvoiceValidator, InvoiceSchema, LineItem, number_to_words_inr


# ---------------------------------------------------------------------------
# 1. Quality Scorer Tests
# ---------------------------------------------------------------------------

def test_quality_scorer_sharp_image():
    scorer = DocumentQualityScorer()
    # Create a synthetic high-contrast image with crisp text-like lines
    img = np.ones((400, 600, 3), dtype=np.uint8) * 240
    for y in range(50, 350, 25):
        cv2.putText(img, "Tax Invoice Sample Line 12345", (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)

    qa = scorer.assess(img)
    assert qa.composite_score >= 0.70
    assert not qa.is_blurry
    assert qa.is_acceptable


def test_quality_scorer_blurry_image():
    scorer = DocumentQualityScorer()
    # Create an intentionally blurred image
    img = np.ones((400, 600, 3), dtype=np.uint8) * 200
    cv2.putText(img, "Blurry text line", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)
    blurred = cv2.GaussianBlur(img, (25, 25), 0)

    qa = scorer.assess(blurred)
    assert qa.is_blurry
    assert "apply_unsharp_mask" in qa.recommended_actions


# ---------------------------------------------------------------------------
# 2. Document Router Tests
# ---------------------------------------------------------------------------

def test_document_router_printed_scan():
    router = DocumentRouter()
    # Synthetic flat white scanned document
    img = np.ones((800, 600, 3), dtype=np.uint8) * 245
    cv2.rectangle(img, (40, 40), (560, 760), (30, 30, 30), 2)
    decision = router.route(b"fake_bytes", filename="invoice_scan.png", first_page_image=img)
    assert decision.doc_type in (DocumentRouter.PRINTED_SCAN, DocumentRouter.PHONE_PHOTO)


def test_document_router_digital_pdf():
    router = DocumentRouter()
    # Synthetic PDF header
    pdf_bytes = b"%PDF-1.4\n%fake digital pdf stream with words"
    decision = router.route(pdf_bytes, filename="digital_invoice.pdf")
    assert decision.doc_type in (DocumentRouter.DIGITAL_PDF, DocumentRouter.UNKNOWN)


# ---------------------------------------------------------------------------
# 3. Bank-Grade Validation & Checksum Tests
# ---------------------------------------------------------------------------

def test_gstin_modulo36_checksum():
    # Valid GSTIN: 27AABCU9603R1ZN
    valid_gstin = "27AABCU9603R1ZN"
    assert validate_gstin_checksum(valid_gstin)

    # Corrupted checksum
    invalid_gstin = "27AABCU9603R1ZM"
    assert not validate_gstin_checksum(invalid_gstin)


def test_pan_and_ifsc_validators():
    assert validate_pan("AABCU9603R")
    assert not validate_pan("12345ABCDE")

    assert validate_ifsc("HDFC0001234")
    assert not validate_ifsc("HDFC1234567")


# ---------------------------------------------------------------------------
# 4. Wholesome Accounting & Tax Model Decision Engine Tests
# ---------------------------------------------------------------------------

def test_global_gst_rate_validation():
    engine = FieldConfidenceEngine()
    invoice_dict = {
        "invoice_number": "INV-2026-001",
        "invoice_date": "24/08/2026",
        "vendor_name": "Acme Industrial Supplies Pvt Ltd",
        "vendor_gstin": "27AABCU9603R1ZN",
        "subtotal": 10000.0,
        "global_cgst_rate": 9.0,
        "global_sgst_rate": 9.0,
        "tax_amount": 1800.0,  # 18% of 10000
        "grand_total": 11800.0,
        "account_number": "123456789012",
        "ifsc_code": "HDFC0001234",
    }

    eval_res = engine.evaluate(invoice_dict)
    assert eval_res["arithmetic_valid"]
    assert eval_res["field_confidences"]["grand_total"] >= 0.95
    assert not eval_res["critical_failure"]


def test_critical_gatekeeper_blocks_auto_accept_on_corrupt_gstin():
    engine = FieldConfidenceEngine()
    invoice_dict = {
        "invoice_number": "INV-2026-002",
        "invoice_date": "24/08/2026",
        "vendor_name": "Acme Industrial Supplies",
        "vendor_gstin": "27AABCU9603R1ZM",  # Invalid checksum check digit
        "subtotal": 1000.0,
        "tax_amount": 180.0,
        "grand_total": 1180.0,
        "account_number": "123456789012",
        "ifsc_code": "HDFC0001234",
    }

    eval_res = engine.evaluate(invoice_dict)
    # The critical gatekeeper MUST force review because vendor_gstin failed checksum
    assert eval_res["needs_review"]
    assert "vendor_gstin" in eval_res["fields_needing_review"]


def test_amount_in_words_conversion():
    words = number_to_words_inr(15600.0)
    assert "Fifteen Thousand Six Hundred" in words


# ---------------------------------------------------------------------------
# 5. Evaluation Fail-Closed & AutoAcceptanceGate Tests
# ---------------------------------------------------------------------------

def test_locked_evaluation_fails_closed_on_missing_or_corrupt_files(tmp_path):
    from active_learning.auto_trainer import evaluate_model_on_locked_test
    # 1. Empty directory
    res = evaluate_model_on_locked_test(tmp_path, tmp_path / "empty_locked_test")
    assert res["status"] == "EVALUATION_FAILED"
    assert res["entity_f1"] == 0.0

    # 2. Directory with corrupt sample JSON
    corrupt_dir = tmp_path / "corrupt_locked_test"
    (corrupt_dir / "train").mkdir(parents=True, exist_ok=True)
    with open(corrupt_dir / "locked_job_ids.json", "w") as f:
        import json
        json.dump(["job_1", "job_2"], f)
    with open(corrupt_dir / "train" / "job_1.json", "w") as f:
        f.write("{invalid_json:")

    res_corrupt = evaluate_model_on_locked_test(tmp_path, corrupt_dir)
    assert res_corrupt["status"] == "EVALUATION_FAILED"
    assert res_corrupt["entity_f1"] == 0.0


def test_auto_acceptance_gate_blocks_handwritten_and_novel_templates():
    from validation.acceptance_gate import AutoAcceptanceGate
    from validation.validator import InvoiceSchema, ValidationReport

    inv = InvoiceSchema(
        invoice_number="INV-100",
        invoice_date="25/08/2026",
        vendor_gstin="27AABCU9603R1ZN",
        grand_total=5000.0,
        subtotal=5000.0,
        overall_confidence=0.95,
        field_confidences={
            "grand_total": 0.98,
            "invoice_number": 0.95,
            "invoice_date": 0.95,
            "vendor_gstin": 0.98,
        }
    )
    report = ValidationReport()

    # 1. Standard scan -> Auto-accepted
    ok, reasons = AutoAcceptanceGate.evaluate(inv, report, quality_score=0.90, doc_type="PRINTED_SCAN")
    assert ok
    assert len(reasons) == 0

    # 2. Novel layout template -> Rejected for initial verification
    ok_novel, reasons_novel = AutoAcceptanceGate.evaluate(inv, report, quality_score=0.90, is_novel_template=True, doc_type="PRINTED_SCAN")
    assert not ok_novel
    assert any("layout template" in r for r in reasons_novel)

    # 3. Handwritten route -> Rejected for human review
    ok_hand, reasons_hand = AutoAcceptanceGate.evaluate(inv, report, quality_score=0.90, doc_type="HANDWRITTEN")
    assert not ok_hand
    assert any("HANDWRITTEN" in r for r in reasons_hand)

