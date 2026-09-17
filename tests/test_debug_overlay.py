"""
tests/test_debug_overlay.py

Tests for the Invoice Debug Mode overlay endpoint and payload generator:
- Validates 0-1000 coordinate space normalization.
- Validates YOLO region, OCR box, TIE anchor, and handwriting zone formatting.
- Validates Field Evidence Layer (selected value, source, candidates, validation status).
- Tests GET /api/invoices/{job_id}/debug-overlay endpoint.
"""

import pytest
from preprocessing.document_profile import DocumentProfile, WordToken, RegionBlock
from api.debug_router import normalize_to_1000, build_debug_overlay_payload


def test_normalize_to_1000():
    # Test raw pixels to 0-1000 with is_raw=True
    norm = normalize_to_1000([120, 160, 240, 320], width=1200, height=1600, is_raw=True)
    assert norm == [100, 100, 200, 200]

    # Test raw pixels exceeding 1000 auto-detected
    norm_large = normalize_to_1000([1200, 1600, 2400, 3200], width=2400, height=3200)
    assert norm_large == [500, 500, 1000, 1000]

    # Test already normalized 0-1000
    norm2 = normalize_to_1000([50, 60, 200, 300])
    assert norm2 == [50, 60, 200, 300]

    # Test normalized 0-1 float
    norm3 = normalize_to_1000([0.1, 0.2, 0.3, 0.4])
    assert norm3 == [100, 200, 300, 400]

    # Test invalid box
    assert normalize_to_1000([], 1000, 1000) is None
    assert normalize_to_1000(None, 1000, 1000) is None


def test_build_debug_overlay_payload():
    # Mock DocumentProfile
    words = [
        WordToken(
            text="INVOICE",
            bbox_norm=[50, 50, 150, 80],
            bbox_raw=[60, 80, 180, 128],
            confidence=0.99,
            page=1,
            block_no=0,
            line_no=0,
            source="native_pdf",
        ),
        WordToken(
            text="INV-2026-001",
            bbox_norm=[200, 50, 350, 80],
            bbox_raw=[240, 80, 420, 128],
            confidence=0.99,
            page=1,
            block_no=0,
            line_no=0,
            source="native_pdf",
        ),
    ]
    regions = [
        RegionBlock(
            label="header",
            bbox_norm=[40, 40, 960, 120],
            bbox_raw=[48, 64, 1152, 192],
            confidence=0.95,
            page=1,
        )
    ]

    doc_prof = DocumentProfile(
        page_count=1,
        width=1200,
        height=1600,
        aspect_ratio=1.33,
        words=words,
        regions=regions,
        anchor_occurrences={
            "invoice_no": [{
                "page": 1,
                "bbox_norm": [50, 50, 150, 80],
                "center_norm": [100.0, 65.0],
            }]
        }
    )

    mock_invoice = {
        "invoice_number": "INV-2026-001",
        "grand_total": "11800.00",
        "field_confidences": {
            "invoice_number": 0.99,
            "grand_total": 0.96,
        },
        "field_provenance": {
            "invoice_number": {
                "value": "INV-2026-001",
                "source": "tie_template",
                "confidence": 0.99,
                "page": 1,
                "bbox": [200, 50, 350, 80],
                "selection_reason": "TIE deterministic template rule match",
                "candidates": [
                    {"source": "tie_template", "value": "INV-2026-001", "confidence": 0.99, "bbox_norm": [200, 50, 350, 80]},
                    {"source": "paddleocr", "value": "INV-2026-001", "confidence": 0.95, "bbox_norm": [200, 50, 350, 80]},
                ],
                "disagreement_score": 0.0,
            }
        },
        "review_reasons": [],
    }

    payload = build_debug_overlay_payload(
        job_id="test-job-123",
        page_idx=0,
        doc_profile=doc_prof,
        invoice_dict=mock_invoice,
    )

    assert payload["job_id"] == "test-job-123"
    assert payload["page"] == 0
    assert payload["page_number"] == 1
    assert payload["total_pages"] == 1
    assert len(payload["yolo_regions"]) == 1
    assert payload["yolo_regions"][0]["label"] == "header"
    assert len(payload["tie_anchors"]) >= 1
    assert any(a["anchor"] == "invoice_no" for a in payload["tie_anchors"])

    # Field evidence checks
    evidence = payload["field_evidence"]
    assert "invoice_number" in evidence
    assert evidence["invoice_number"]["selected_value"] == "INV-2026-001"
    assert evidence["invoice_number"]["selected_source"] == "tie_template"
    assert len(evidence["invoice_number"]["candidates"]) == 2
    assert payload["summary"]["fields_located_count"] >= 1


def test_human_corrected_field_provenance_and_location_editing():
    """
    Tests that human-edited locations in field_provenance are preserved
    via InvoiceSchema.from_invoice_builder_json and rendered into the debug overlay.
    """
    from validation.validator import InvoiceSchema

    corrections_payload = {
        "meta": {
            "invoiceNo": "INV-MODIFIED-999",
            "date": "15/09/2026",
        },
        "company": {
            "name": "Acme Tools Pvt Ltd",
            "gstin": "19AAACA1234A1Z5",
        },
        "client": {
            "name": "State Forest Dept",
        },
        "totals": {
            "grandTotal": 45000.0,
            "taxableAmount": 40000.0,
        },
        "items": [
            {"description": "Industrial Pump", "quantity": 1, "rate": 40000.0, "taxableValue": 40000.0}
        ],
        "field_provenance": {
            "invoice_number": {
                "bbox": [250, 60, 480, 95],
                "page": 1,
                "source": "human_corrected",
                "confidence": 1.0,
                "value": "INV-MODIFIED-999",
            },
            "grand_total": {
                "bbox": [750, 850, 920, 890],
                "page": 1,
                "source": "human_pasted_snap",
                "confidence": 1.0,
                "value": "45000.0",
            },
        },
    }

    schema = InvoiceSchema.from_invoice_builder_json(corrections_payload)
    assert schema.invoice_number == "INV-MODIFIED-999"
    assert schema.grand_total == 45000.0
    assert "invoice_number" in schema.field_provenance
    assert schema.field_provenance["invoice_number"]["bbox"] == [250, 60, 480, 95]
    assert schema.field_provenance["invoice_number"]["source"] == "human_corrected"
    assert schema.field_provenance["grand_total"]["bbox"] == [750, 850, 920, 890]
    assert schema.field_provenance["grand_total"]["source"] == "human_pasted_snap"

    # Build debug overlay payload with this human-edited schema
    payload = build_debug_overlay_payload(
        job_id="job-corrected-123",
        page_idx=0,
        doc_profile=None,
        invoice_dict=schema.model_dump(),
    )

    fe = payload["field_evidence"]
    assert "invoice_number" in fe
    assert fe["invoice_number"]["bbox_norm"] == [250, 60, 480, 95]
    assert fe["invoice_number"]["selected_source"] == "human_corrected"
    assert fe["invoice_number"]["selected_value"] == "INV-MODIFIED-999"

    assert "grand_total" in fe
    assert fe["grand_total"]["bbox_norm"] == [750, 850, 920, 890]
    assert fe["grand_total"]["selected_source"] == "human_pasted_snap"
    assert fe["grand_total"]["selected_value"] == "45000.0"

