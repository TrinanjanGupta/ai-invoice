"""
tests/test_handwriting_router.py

Unit tests for 5-level handwriting classification, ruled line detection, and routing decisions.
"""

import cv2
import numpy as np
import pytest
from preprocessing.document_router import DocumentRouter, HandwritingLevel, DocumentRoutingDecision


def test_router_clean_printed_scan():
    router = DocumentRouter()
    # Create synthetic printed scan image (clean white background with crisp straight table lines)
    img = np.ones((800, 600, 3), dtype=np.uint8) * 255
    # Add printed grid lines
    cv2.rectangle(img, (50, 50), (550, 750), (0, 0, 0), 2)
    for y in range(100, 700, 50):
        cv2.line(img, (50, y), (550, y), (0, 0, 0), 1)

    _, encoded = cv2.imencode(".png", img)
    decision = router.route(encoded.tobytes(), filename="printed_invoice.png", first_page_image=img)

    assert decision.doc_type in (DocumentRouter.PRINTED_SCAN, DocumentRouter.DIGITAL_PDF)
    assert decision.handwriting_level == HandwritingLevel.NONE.value
    assert decision.handwriting_confidence == 0.0
    assert not decision.has_ruled_lines


def test_router_handwritten_pad():
    router = DocumentRouter()
    # Create synthetic handwritten pad image (ruled lines + cursive stroke curves)
    img = np.ones((800, 600, 3), dtype=np.uint8) * 255
    # Ruled pad lines
    for y in range(80, 750, 40):
        cv2.line(img, (40, y), (560, y), (120, 120, 120), 1)

    # Add cursive handwritten strokes (ellipses and curved polylines)
    for i in range(12):
        center = (np.random.randint(100, 500), np.random.randint(100, 700))
        cv2.ellipse(img, center, (40, 20), np.random.randint(0, 180), 0, 360, (20, 20, 80), 2)
        pts = np.array([
            [center[0] - 30, center[1] + 10],
            [center[0] - 10, center[1] - 15],
            [center[0] + 15, center[1] + 20],
            [center[0] + 35, center[1] - 5],
        ], np.int32)
        cv2.polylines(img, [pts], isClosed=False, color=(20, 20, 80), thickness=2)

    _, encoded = cv2.imencode(".png", img)
    decision = router.route(encoded.tobytes(), filename="pad_invoice.png", first_page_image=img)

    assert decision.handwriting_level in (
        HandwritingLevel.MOSTLY_HANDWRITTEN.value,
        HandwritingLevel.FULLY_HANDWRITTEN.value,
        HandwritingLevel.MIXED.value,
        HandwritingLevel.FIELD_ONLY.value,
    )
    assert decision.handwriting_level != HandwritingLevel.NONE.value
    assert decision.requires_stroke_enhancement is True


def test_router_confidence_domain_separation():
    router = DocumentRouter()
    img = np.ones((600, 400, 3), dtype=np.uint8) * 255
    _, encoded = cv2.imencode(".png", img)
    decision = router.route(encoded.tobytes(), filename="test.png", first_page_image=img)

    # doc_type confidence and handwriting_confidence must be distinct attributes
    assert hasattr(decision, "confidence")
    assert hasattr(decision, "handwriting_confidence")
    assert isinstance(decision.confidence, float)
    assert isinstance(decision.handwriting_confidence, float)
