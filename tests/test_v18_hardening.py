"""
tests/test_v18_hardening.py

Unit tests for v18 Production Hardening & Architectural Optimization:
1. Single source of truth PDF preprocessing.
2. Mixed PDF page-level provenance.
3. Audit field provenance with real page, bbox, and OCR confidence.
4. Multi-invoice segments as processing units with early return.
5. Score-based document segmentation with boundary weights.
6. Centralized CandidateFusionEngine OCR hypothesis resolution.
7. Celery production default configuration.
"""

import pytest
import numpy as np
from PIL import Image

from preprocessing.document_router import DocumentRouter, PageRoutingDecision
from preprocessing.document_profile import DocumentProfile, WordToken
from preprocessing.document_segmenter import DocumentSegmenter, PageSignals
from understanding.layoutlm import ExtractedInvoice, ExtractedField
from validation.validator import InvoiceValidator, InvoiceSchema
from ocr.candidate_generator import OCRCandidate
from ocr.candidate_fusion import CandidateFusionEngine
from config.settings import get_settings


def test_single_source_of_truth_pdf_routing():
    """Verify that PageRoutingDecision carries all per-page metadata."""
    decision = PageRoutingDecision(
        page_num=1,
        doc_type=DocumentRouter.HANDWRITTEN,
        confidence=0.95,
        is_digital_native=False,
        requires_perspective_warp=False,
        requires_shadow_removal=False,
        requires_stroke_enhancement=True,
        reason="Handwritten test scan",
        handwriting_level="FULLY_HANDWRITTEN",
        handwriting_confidence=0.95,
    )
    assert decision.doc_type == DocumentRouter.HANDWRITTEN
    assert decision.handwriting_level == "FULLY_HANDWRITTEN"
    assert decision.page_num == 1


def test_mixed_pdf_page_level_provenance():
    """Verify that page_sources map enforces page-specific token provenance."""
    # Mock text blocks for page 1 and page 2
    class MockBlock:
        def __init__(self, text, page):
            self.text = text
            self.page = page
            self.bbox = [100, 100, 200, 150]
            self.confidence = 0.95

    class MockOCRRes:
        def __init__(self, blocks, page):
            self.text_blocks = blocks
            self.page = page
            self.engine = None

    p1_blocks = [MockBlock("TAX INVOICE", page=1)]
    p2_blocks = [MockBlock("TERMS AND CONDITIONS", page=2)]

    ocr_results = {
        "full_page_p1": MockOCRRes(p1_blocks, page=1),
        "full_page_p2": MockOCRRes(p2_blocks, page=2),
    }

    page_sources = {
        1: "native_pdf",
        2: "paddleocr",
    }

    profile = DocumentProfile.from_ocr_and_regions(
        ocr_results=ocr_results,
        regions=[],
        width=1000,
        height=1400,
        page_count=2,
        page_sources=page_sources,
    )

    tokens_p1 = [t for t in profile.words if t.page == 1]
    tokens_p2 = [t for t in profile.words if t.page == 2]

    assert len(tokens_p1) > 0
    assert len(tokens_p2) > 0
    assert tokens_p1[0].source == "native_pdf"
    assert tokens_p2[0].source == "paddleocr"


def test_audit_field_provenance_recording():
    """Verify that validator captures full field provenance including real page and bbox."""
    inv = ExtractedInvoice(
        invoice_number=ExtractedField(value="INV-2026-99", confidence=0.98, source="native_pdf", page=1, bbox=[50, 50, 200, 80]),
        grand_total=ExtractedField(value="45000.00", confidence=0.94, source="trocr", page=2, bbox=[600, 800, 750, 840]),
        vendor_name=ExtractedField(value="Acme Corp", confidence=0.95, source="paddleocr", page=1, bbox=[100, 120, 300, 150]),
    )

    validator = InvoiceValidator()
    schema, report = validator.validate(inv)

    assert "grand_total" in schema.field_provenance
    assert schema.field_provenance["grand_total"]["page"] == 2
    assert schema.field_provenance["grand_total"]["source"] == "trocr"
    assert schema.field_provenance["grand_total"]["bbox"] == [600, 800, 750, 840]
    assert schema.field_provenance["invoice_number"]["page"] == 1
    assert schema.field_provenance["invoice_number"]["source"] == "native_pdf"


def test_score_based_document_segmenter():
    """Verify score-based segmentation boundary weights and continuation suppression."""
    segmenter = DocumentSegmenter()

    p1_signals = PageSignals(
        page_num=1,
        invoice_number="INV-001",
        vendor_gstin="29ABCDE1234F1Z5",
        has_invoice_title=True,
        has_continuation_marker=False,
        has_totals_block=True,
    )

    # Page 2: New invoice title and different invoice number
    p2_new_inv = PageSignals(
        page_num=2,
        invoice_number="INV-002",
        vendor_gstin="29ABCDE1234F1Z5",
        has_invoice_title=True,
        has_continuation_marker=False,
        has_totals_block=False,
    )

    score_new, reasons_new = segmenter.compute_boundary_score(
        sig=p2_new_inv,
        prev_sig=p1_signals,
        current_inv_no="INV-001",
        current_gstin="29ABCDE1234F1Z5",
    )
    # +0.35 (inv_no) + +0.20 (title) + +0.10 (prev totals) = 0.65 >= 0.40
    assert score_new >= 0.40

    # Page 2 with explicit continuation marker
    p2_cont = PageSignals(
        page_num=2,
        invoice_number=None,
        vendor_gstin="29ABCDE1234F1Z5",
        has_invoice_title=False,
        has_continuation_marker=True,
        has_totals_block=False,
    )

    score_cont, reasons_cont = segmenter.compute_boundary_score(
        sig=p2_cont,
        prev_sig=p1_signals,
        current_inv_no="INV-001",
        current_gstin="29ABCDE1234F1Z5",
    )
    assert score_cont < 0.40


def test_candidate_fusion_engine_numeric_and_gstin():
    """Verify that CandidateFusionEngine resolves OCR ambiguities via type validation."""
    engine = CandidateFusionEngine()

    # 1. Invoice Number with O instead of 0 inside numeric sequence
    cands_inv = [
        OCRCandidate(text="INV8O21", confidence=0.88, source="paddleocr", field_name="invoice_number", bbox=[10, 10, 50, 30]),
        OCRCandidate(text="INV8021", confidence=0.86, source="trocr", field_name="invoice_number", bbox=[10, 10, 50, 30]),
    ]
    fused_inv = engine.fuse_candidates(cands_inv, field_name="invoice_number", expected_type="text")
    assert fused_inv.selected_text == "INV8021"

    # 2. Grand Total numeric resolution
    cands_num = [
        OCRCandidate(text="12,5OO.OO", confidence=0.90, source="paddleocr", field_name="grand_total", bbox=[20, 20, 60, 40]),
        OCRCandidate(text="12500.00", confidence=0.87, source="trocr", field_name="grand_total", bbox=[20, 20, 60, 40]),
    ]
    fused_num = engine.fuse_candidates(cands_num, field_name="grand_total", expected_type="numeric")
    assert float(fused_num.selected_text.replace(",", "")) == 12500.00

    # 3. Valid GSTIN checksum resolution
    valid_gstin = "29AAAAA0000A1ZY"  # Valid Modulo 36 GSTIN with calculated checksum 'Y'
    cands_gstin = [
        OCRCandidate(text="29AAAAAOOOOA1Z5", confidence=0.92, source="paddleocr", field_name="vendor_gstin", bbox=[0, 0, 10, 10]),
        OCRCandidate(text=valid_gstin, confidence=0.85, source="trocr", field_name="vendor_gstin", bbox=[0, 0, 10, 10]),
    ]
    fused_gstin = engine.fuse_candidates(cands_gstin, field_name="vendor_gstin", expected_type="gstin")
    assert fused_gstin.selected_text == valid_gstin


def test_celery_production_default_settings():
    """Verify that Celery is enabled by default in settings."""
    settings = get_settings()
    assert settings.use_celery is True
