"""
tests/test_page_level_routing.py

Unit tests for page-level routing classification and adaptive per-page paths.
"""

import numpy as np
import cv2
import pytest
from preprocessing.document_router import DocumentRouter, PageRoutingDecision, HandwritingLevel


def test_page_routing_digital_native():
    router = DocumentRouter()
    img = np.ones((800, 600, 3), dtype=np.uint8) * 255
    res = router.route_page(img, is_native_pdf=True, page_num=1)

    assert isinstance(res, PageRoutingDecision)
    assert res.doc_type == DocumentRouter.DIGITAL_PDF
    assert res.is_digital_native is True
    assert res.confidence >= 0.95
    assert res.page_num == 1


def test_page_routing_scanned_printed():
    router = DocumentRouter()
    # Uniform white scanned page with printed horizontal lines
    img = np.ones((800, 600, 3), dtype=np.uint8) * 255
    cv2.line(img, (50, 100), (550, 100), (0, 0, 0), 2)
    cv2.line(img, (50, 200), (550, 200), (0, 0, 0), 2)
    cv2.putText(img, "TAX INVOICE", (150, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    res = router.route_page(img, is_native_pdf=False, page_num=2)
    assert isinstance(res, PageRoutingDecision)
    assert res.doc_type in (DocumentRouter.PRINTED_SCAN, DocumentRouter.MIXED)
    assert res.is_digital_native is False
    assert res.page_num == 2


def test_page_routing_handwritten_pad():
    router = DocumentRouter()
    # Handwritten pad with ruled lines and cursive ink
    img = np.ones((800, 600, 3), dtype=np.uint8) * 245
    for y in range(80, 750, 35):
        cv2.line(img, (30, y), (570, y), (210, 200, 200), 1)
    cv2.putText(img, "Shree Ganesh Hardware", (60, 120), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, (120, 30, 20), 2)
    cv2.putText(img, "Bill No 452  Date 12/03", (60, 160), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.7, (120, 30, 20), 2)

    res = router.route_page(img, is_native_pdf=False, page_num=3)
    assert isinstance(res, PageRoutingDecision)
    assert res.handwriting_level != HandwritingLevel.NONE.value
    assert res.has_ruled_lines is True
    assert res.page_num == 3
