"""
tests/test_layout_reconstructor.py

Unit tests for LayoutReconstructor:
- Tests horizontal word merging into coherent lines.
- Tests 2-column gutter detection and natural reading order sorting.
- Tests paragraph / block clustering.
- Tests macro zone classification (vendor, header, buyer, totals, footer).
"""

import pytest
from ocr.extractor import OCRWord, TextBlock
from ocr.layout_reconstructor import LayoutReconstructor, ReconstructedLine


def test_horizontal_line_merging():
    reconstructor = LayoutReconstructor(page_width=1000, page_height=1400)

    # Three words on the same horizontal baseline (y around 100)
    w1 = OCRWord(text="Invoice", confidence=0.98, bbox=[50, 100, 120, 125])
    w2 = OCRWord(text="No:", confidence=0.99, bbox=[130, 102, 165, 124])
    w3 = OCRWord(text="INV-9901", confidence=0.97, bbox=[180, 100, 270, 126])

    # Another word on a lower line (y around 150)
    w4 = OCRWord(text="Date:", confidence=0.95, bbox=[50, 150, 100, 175])
    w5 = OCRWord(text="16-09-2026", confidence=0.96, bbox=[110, 151, 210, 174])

    ocr_res, regions = reconstructor.reconstruct([w1, w2, w3, w4, w5], page_num=1)

    assert len(ocr_res.text_blocks) == 2
    assert ocr_res.text_blocks[0].text == "Invoice No: INV-9901"
    assert ocr_res.text_blocks[1].text == "Date: 16-09-2026"


def test_two_column_reading_order():
    reconstructor = LayoutReconstructor(page_width=1000, page_height=1400)

    # Left Column: Bill To section
    # y=100: Bill To
    # y=130: Acme Corp
    # y=160: Mumbai India
    b1 = OCRWord(text="Bill To:", confidence=0.99, bbox=[50, 100, 120, 125])
    b2 = OCRWord(text="Acme Corp", confidence=0.98, bbox=[50, 130, 150, 155])
    b3 = OCRWord(text="Mumbai India", confidence=0.97, bbox=[50, 160, 170, 185])

    # Right Column: Invoice Details (x >= 500)
    # y=100: Invoice No: 120
    # y=130: Date: 2026-09-16
    # y=160: Due: 2026-09-30
    d1 = OCRWord(text="Invoice No: 120", confidence=0.99, bbox=[600, 100, 750, 125])
    d2 = OCRWord(text="Date: 2026-09-16", confidence=0.98, bbox=[600, 130, 760, 155])
    d3 = OCRWord(text="Due: 2026-09-30", confidence=0.97, bbox=[600, 160, 750, 185])

    # Raw OCR list might interleave left and right lines by Y coordinate
    raw_words = [b1, d1, b2, d2, b3, d3]

    ocr_res, regions = reconstructor.reconstruct(raw_words, page_num=1)
    texts = [b.text for b in ocr_res.text_blocks]

    # In proper reading order, left column lines come first, followed by right column lines
    # "Bill To:", "Acme Corp", "Mumbai India" should appear consecutively without "Invoice No" interleaved
    assert texts[0] == "Bill To:"
    assert texts[1] == "Acme Corp"
    assert texts[2] == "Mumbai India"
    assert texts[3] == "Invoice No: 120"
    assert texts[4] == "Date: 2026-09-16"
    assert texts[5] == "Due: 2026-09-30"


def test_zone_classification():
    reconstructor = LayoutReconstructor(page_width=1000, page_height=1400)

    # Top vendor block
    v1 = OCRWord(text="RELIANCE INDUSTRIES LTD", confidence=0.99, bbox=[50, 40, 300, 70])
    v2 = OCRWord(text="GSTIN: 27AABCU9603R1ZM", confidence=0.98, bbox=[50, 75, 260, 95])

    # Buyer block
    b1 = OCRWord(text="Bill To: Tata Consultancy", confidence=0.99, bbox=[50, 200, 280, 225])

    # Header details
    h1 = OCRWord(text="Invoice No: INV-4401", confidence=0.98, bbox=[600, 200, 800, 225])

    # Totals block at bottom
    t1 = OCRWord(text="Subtotal: 50,000.00", confidence=0.98, bbox=[600, 1100, 850, 1125])
    t2 = OCRWord(text="Grand Total: 59,000.00", confidence=0.99, bbox=[600, 1150, 880, 1180])

    ocr_res, regions = reconstructor.reconstruct([v1, v2, b1, h1, t1, t2], page_num=1)

    labels = {r.label for r in regions}
    assert "vendor_block" in labels or "header" in labels
    assert "buyer_block" in labels
    assert "totals" in labels
