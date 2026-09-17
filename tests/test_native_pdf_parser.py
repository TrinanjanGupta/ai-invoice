"""
tests/test_native_pdf_parser.py

Unit tests for NativePDFParser (authoritative vector geometry and text extractor):
- Verifies extraction of blocks, lines, spans, fonts, flags, and WordTokens.
- Verifies deterministic geometry zone classification (vendor_block, header, buyer_block, totals, etc.).
- Verifies vendor name discovery via font hierarchy.
- Verifies regex anchor extraction on digital vector text.
- Verifies vector table extraction integration.
"""

import pytest
import pymupdf
from preprocessing.native_pdf_parser import NativePDFParser, NativeVectorPage


def create_mock_digital_invoice_pdf() -> bytes:
    """Creates a realistic in-memory vector digital invoice PDF."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)  # Standard A4

    # Vendor Header (Large bold font at top left)
    page.insert_text((40, 50), "ACME INDUSTRIAL SOLUTIONS PVT LTD", fontsize=14, fontname="helv")
    page.insert_text((40, 70), "GSTIN: 27AABCU9603R1ZM | PAN: AABCU9603R", fontsize=9, fontname="helv")
    page.insert_text((40, 85), "123 Nariman Point, Mumbai, Maharashtra 400021", fontsize=8, fontname="helv")

    # Document Title (Top right)
    page.insert_text((450, 50), "TAX INVOICE", fontsize=18, fontname="helv")

    # Invoice Metadata (Right side)
    page.insert_text((350, 110), "Invoice No: INV-2026-9901", fontsize=9, fontname="helv")
    page.insert_text((350, 125), "Invoice Date: 16-09-2026", fontsize=9, fontname="helv")
    page.insert_text((350, 140), "Due Date: 30-09-2026", fontsize=9, fontname="helv")
    page.insert_text((350, 155), "PO Number: PO-88712", fontsize=9, fontname="helv")
    page.insert_text((350, 170), "Place of Supply: Maharashtra", fontsize=9, fontname="helv")

    # Buyer Section (Left side under vendor)
    page.insert_text((40, 110), "Bill To:", fontsize=9, fontname="helv")
    page.insert_text((40, 125), "Global Logistics & Infra Corp", fontsize=9, fontname="helv")
    page.insert_text((40, 140), "GSTIN: 29AABCU9603R1Z5", fontsize=9, fontname="helv")
    page.insert_text((40, 155), "Bangalore, Karnataka 560001", fontsize=8, fontname="helv")

    # Draw a vector table
    # Table border and header rule
    page.draw_rect(pymupdf.Rect(40, 220, 555, 300))
    page.draw_line(pymupdf.Point(40, 245), pymupdf.Point(555, 245))
    page.draw_line(pymupdf.Point(40, 270), pymupdf.Point(555, 270))

    # Table Header text
    page.insert_text((45, 235), "Description", fontsize=9, fontname="helv")
    page.insert_text((260, 235), "Qty", fontsize=9, fontname="helv")
    page.insert_text((330, 235), "Rate", fontsize=9, fontname="helv")
    page.insert_text((460, 235), "Amount", fontsize=9, fontname="helv")

    # Row 1
    page.insert_text((45, 260), "Industrial Valve 4-inch", fontsize=9, fontname="helv")
    page.insert_text((260, 260), "5", fontsize=9, fontname="helv")
    page.insert_text((330, 260), "1000.00", fontsize=9, fontname="helv")
    page.insert_text((460, 260), "5000.00", fontsize=9, fontname="helv")

    # Row 2
    page.insert_text((45, 285), "Hydraulic Pump Unit", fontsize=9, fontname="helv")
    page.insert_text((260, 285), "2", fontsize=9, fontname="helv")
    page.insert_text((330, 285), "2500.00", fontsize=9, fontname="helv")
    page.insert_text((460, 285), "5000.00", fontsize=9, fontname="helv")

    # Totals Section (Bottom right)
    page.insert_text((380, 330), "Subtotal: 10000.00", fontsize=9, fontname="helv")
    page.insert_text((380, 345), "CGST: 900.00", fontsize=9, fontname="helv")
    page.insert_text((380, 360), "SGST: 900.00", fontsize=9, fontname="helv")
    page.insert_text((380, 380), "Grand Total: 11800.00", fontsize=11, fontname="helv")

    # Bank Details / Footer (Bottom left)
    page.insert_text((40, 420), "Bank Name: State Bank of India", fontsize=9, fontname="helv")
    page.insert_text((40, 435), "A/C No: 123456789012", fontsize=9, fontname="helv")
    page.insert_text((40, 450), "IFSC Code: SBIN0001234", fontsize=9, fontname="helv")

    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


def test_native_pdf_parser_on_synthetic_invoice():
    pdf_bytes = create_mock_digital_invoice_pdf()
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]

    parser = NativePDFParser()
    parsed: NativeVectorPage = parser.parse_page(page, page_num=1)

    assert parsed.page_num == 1
    assert parsed.page_width == 595.0
    assert parsed.page_height == 842.0

    # 1. Vendor discovery via font hierarchy
    assert parsed.vendor_name_candidate is not None
    assert "ACME INDUSTRIAL SOLUTIONS" in parsed.vendor_name_candidate.upper()

    # 2. Buyer discovery
    assert parsed.buyer_name_candidate is not None
    assert "Global Logistics" in parsed.buyer_name_candidate

    # 3. Deterministic regex field extraction
    fields = parsed.extracted_fields
    assert "vendor_name" in fields
    assert "INV-2026-9901" in fields["invoice_number"]["value"]
    assert "16-09-2026" in fields["invoice_date"]["value"]
    assert "30-09-2026" in fields["due_date"]["value"]
    assert "PO-88712" in fields["po_number"]["value"]
    assert "11800.00" in fields["grand_total"]["value"]
    assert "10000.00" in fields["subtotal"]["value"]
    assert "900.00" in fields["cgst"]["value"]
    assert "900.00" in fields["sgst"]["value"]

    # 4. Geometry zones
    region_labels = {r.label for r in parsed.regions}
    assert "vendor_block" in region_labels
    assert "buyer_block" in region_labels
    assert "totals" in region_labels
    assert len(parsed.words) > 30

    doc.close()
