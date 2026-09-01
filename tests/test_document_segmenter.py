"""
tests/test_document_segmenter.py

Unit tests for DocumentSegmenter and multi-page merged invoice boundary detection.
"""

import pytest
from preprocessing.document_segmenter import DocumentSegmenter, DocumentSegment


class TestDocumentSegmenter:
    def test_single_page_document(self):
        segmenter = DocumentSegmenter()
        pages_text = {
            1: "TAX INVOICE\nInvoice No: INV-2026-001\nDate: 01/08/2026\nGSTIN: 27AAAAA0000A1Z5\nGrand Total: 5000.00"
        }
        segments = segmenter.segment(pages_text)
        assert len(segments) == 1
        assert segments[0].page_indices == [1]
        assert segments[0].invoice_number_hint == "INV-2026-001"
        assert segments[0].vendor_gstin_hint == "27AAAAA0000A1Z5"
        assert segments[0].is_multi_page is False

    def test_multi_page_single_invoice_continuation(self):
        segmenter = DocumentSegmenter()
        pages_text = {
            1: "TAX INVOICE\nInvoice No: INV-999\nGSTIN: 27AAAAA0000A1Z5\nItem 1: 1000\nItem 2: 2000\nContinued on next page...",
            2: "Invoice No: INV-999\nGSTIN: 27AAAAA0000A1Z5\nItem 3: 3000\nSubtotal: 6000\nGrand Total: 7080.00",
        }
        segments = segmenter.segment(pages_text)
        assert len(segments) == 1
        assert segments[0].page_indices == [1, 2]
        assert segments[0].is_multi_page is True
        assert segments[0].invoice_number_hint == "INV-999"

    def test_merged_multi_invoice_detection(self):
        segmenter = DocumentSegmenter()
        pages_text = {
            # Invoice A (Page 1)
            1: "TAX INVOICE\nInvoice No: INV-A-101\nGSTIN: 27AAAAA0000A1Z5\nGrand Total: 1500.00",
            # Invoice B (Page 2 & 3)
            2: "TAX INVOICE\nInvoice No: INV-B-202\nGSTIN: 29BBBBB0000B1Z6\nPage 1 of 2\nItem 1: 500",
            3: "Invoice No: INV-B-202\nGSTIN: 29BBBBB0000B1Z6\nPage 2 of 2\nGrand Total: 590.00",
            # Invoice C (Page 4)
            4: "BILL OF SUPPLY\nInvoice No: INV-C-303\nGSTIN: 07CCCCC0000C1Z7\nGrand Total: 25000.00",
        }
        segments = segmenter.segment(pages_text)
        assert len(segments) == 3
        assert segments[0].page_indices == [1]
        assert segments[0].invoice_number_hint == "INV-A-101"

        assert segments[1].page_indices == [2, 3]
        assert segments[1].invoice_number_hint == "INV-B-202"
        assert segments[1].is_multi_page is True

        assert segments[2].page_indices == [4]
        assert segments[2].invoice_number_hint == "INV-C-303"

