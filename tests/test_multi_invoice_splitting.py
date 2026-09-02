"""
tests/test_multi_invoice_splitting.py

Unit tests for multi-invoice document segmentation and child sub-pipeline execution.
"""

import pytest
from preprocessing.document_segmenter import DocumentSegmenter, DocumentSegment


def test_document_segmenter_multi_invoice_splitting():
    segmenter = DocumentSegmenter()
    pages_text = {
        1: "TAX INVOICE\nVendor: Alpha Traders\nGSTIN: 27AABCU9603R1ZN\nInvoice No: INV-001\nItem 1: Rs. 1000\nTotal: Rs. 1000",
        2: "TAX INVOICE\nVendor: Beta Enterprises\nGSTIN: 29ABCDE1234F1Z5\nInvoice No: INV-999\nItem 1: Rs. 5000\nTotal: Rs. 5000",
    }

    segments = segmenter.segment(pages_text)
    assert len(segments) == 2
    assert segments[0].page_indices == [1]
    assert segments[0].invoice_number_hint == "INV-001"
    assert segments[1].page_indices == [2]
    assert segments[1].invoice_number_hint == "INV-999"


def test_single_invoice_multi_page_not_split():
    segmenter = DocumentSegmenter()
    pages_text = {
        1: "TAX INVOICE\nVendor: Alpha Traders\nInvoice No: INV-100\nPage 1 of 2\nItem 1: 500.00\nContinued...",
        2: "Invoice No: INV-100\nPage 2 of 2\nBrought forward: 500.00\nGrand Total: Rs. 1500.00",
    }

    segments = segmenter.segment(pages_text)
    assert len(segments) == 1
    assert segments[0].page_indices == [1, 2]
    assert segments[0].is_multi_page is True
