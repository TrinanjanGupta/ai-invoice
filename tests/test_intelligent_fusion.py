"""
tests/test_intelligent_fusion.py

Unit tests for Phase H3: Intelligent Fusion & Confidence Calibration.
- Crop-level VLM ambiguity resolver
- Accounting hypothesis validator (arithmetic equilibrium)
- 6-factor composite confidence model
"""

import numpy as np
import pytest
from llm_fallback.ollama_client import OllamaClient
from validation.validator import InvoiceValidator, InvoiceSchema
from validation.confidence_engine import FieldConfidenceEngine


def test_crop_vlm_resolver_interface():
    client = OllamaClient(enabled=False, enable_vision=False)
    # When VLM is disabled / offline, gracefully returns empty string and 0.0 confidence
    dummy_crop = np.ones((40, 150, 3), dtype=np.uint8) * 255
    res_text, res_conf = client.resolve_crop_ambiguity(
        crop_image=dummy_crop,
        field_name="grand_total",
        candidates=["1560.50", "156O.50"],
    )
    assert res_text == ""
    assert res_conf == 0.0


def test_accounting_hypothesis_equilibrium_verification():
    validator = InvoiceValidator()
    # Candidate hypotheses with noise
    subtotal_candidates = [1000.0, 1000.5, 1050.0]
    tax_candidates = [180.0, 180.5, 190.0]
    total_candidates = [1200.0, 1180.0, 1180.5]

    winner = validator.reconcile_accounting_hypotheses(
        candidate_map={},
        subtotal_cands=subtotal_candidates,
        tax_cands=tax_candidates,
        total_cands=total_candidates,
    )

    assert winner is not None
    assert winner["subtotal"][0] == 1000.0
    assert winner["tax_amount"][0] == 180.0
    assert winner["grand_total"][0] == 1180.0
    assert winner["grand_total"][1] >= 0.95  # Boosted confidence on math equilibrium


def test_composite_confidence_model():
    engine = FieldConfidenceEngine()
    invoice_data = {
        "invoice_number": "INV/2026/088",
        "invoice_date": "15/07/2026",
        "vendor_name": "ABC Enterprises",
        "vendor_gstin": "27AABCU9603R1ZN",  # Valid checksum
        "buyer_name": "XYZ Corp",
        "subtotal": 1000.0,
        "tax_amount": 180.0,
        "grand_total": 1180.0,
    }

    # Evaluate printed vs handwritten documents
    res_printed = engine.evaluate(invoice_data, ocr_avg_conf=0.95, handwriting_level="NONE", doc_type="PRINTED_SCAN")
    res_hw = engine.evaluate(invoice_data, ocr_avg_conf=0.95, handwriting_level="MOSTLY_HANDWRITTEN", doc_type="HANDWRITTEN", handwriting_penalty=0.85)

    assert res_printed["field_confidences"]["vendor_gstin"] == 0.99
    assert res_printed["field_confidences"]["grand_total"] >= 0.95
    # Handwritten penalty properly calibrates confidence
    assert res_hw["field_confidences"]["vendor_name"] < res_printed["field_confidences"]["vendor_name"]
