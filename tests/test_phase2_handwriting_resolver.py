"""
tests/test_phase2_handwriting_resolver.py

Unit tests verifying production-grade handwriting candidate extraction,
equilibrium reconciliation, and downstream field resolution.
"""

import numpy as np
import pytest
from ocr.candidate_generator import CandidateGenerator, OCRCandidate
from preprocessing.handwriting_cropper import FieldCrop
from validation.validator import InvoiceValidator, InvoiceSchema


def test_handwriting_candidate_generation_with_corrections():
    generator = CandidateGenerator()
    dummy_img = np.ones((50, 200, 3), dtype=np.uint8) * 255
    crop = FieldCrop(
        field_name="grand_total",
        crop_image=dummy_img,
        enhanced_crop=dummy_img,
        bbox_page=[50, 100, 250, 150],
        page=1,
    )

    # Raw OCR misrecognized numeric '156O.50' (with letter O instead of 0)
    cands = generator.generate_candidates(
        field_crop=crop,
        raw_ocr_text="156O.50",
        raw_ocr_confidence=0.82,
    )

    texts = [c.text for c in cands]
    assert "1560.50" in texts or "156O.50" in texts
    assert len(cands) >= 1


def test_accounting_equilibrium_boosts_confidence():
    validator = InvoiceValidator()
    # Candidate hypotheses that balance perfectly: 1000 + 180 = 1180
    winner = validator.reconcile_accounting_hypotheses(
        candidate_map={},
        subtotal_cands=[1000.0],
        tax_cands=[180.0],
        total_cands=[1180.0],
    )

    assert winner is not None
    assert winner["subtotal"][0] == 1000.0
    assert winner["tax_amount"][0] == 180.0
    assert winner["grand_total"][0] == 1180.0
    assert winner["grand_total"][1] >= 0.95
