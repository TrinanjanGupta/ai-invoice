"""
scripts/test_pipeline.py

Smoke-tests the full pipeline with a synthetic invoice image.
No GPU, no trained models required — uses heuristic fallbacks.

Usage:
    python scripts/test_pipeline.py
    python scripts/test_pipeline.py --image path/to/invoice.jpg
    python scripts/test_pipeline.py --pdf path/to/invoice.pdf
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def create_synthetic_invoice() -> bytes:
    """Create a simple synthetic invoice PNG for testing without a real file."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io

        img = Image.new("RGB", (794, 1123), color=(255, 255, 255))  # A4 at 96dpi
        draw = ImageDraw.Draw(img)

        def text(x, y, t, size=14, bold=False):
            draw.text((x, y), t, fill=(30, 30, 30))

        # Header
        draw.rectangle([40, 40, 754, 160], fill=(37, 99, 235))
        draw.text((50, 60), "INVOICE", fill="white")
        draw.text((50, 100), "Acme Technologies Pvt Ltd", fill="white")
        draw.text((50, 120), "GSTIN: 27AADCA1234A1Z5", fill=(200, 220, 255))

        # Invoice meta
        draw.text((500, 60), "Invoice No: INV-2024-0042", fill="white")
        draw.text((500, 80), "Date: 15/06/2024", fill="white")
        draw.text((500, 100), "Due Date: 15/07/2024", fill="white")

        # Vendor block
        draw.text((40, 180), "Bill From:", fill=(100, 100, 100))
        draw.text((40, 200), "Acme Technologies Pvt Ltd", fill=(30, 30, 30))
        draw.text((40, 218), "42, Tech Park, Pune, Maharashtra 411001", fill=(80, 80, 80))
        draw.text((40, 236), "Email: billing@acme.tech | Phone: +91 9876543210", fill=(80, 80, 80))
        draw.text((40, 254), "PAN: AADCA1234A", fill=(80, 80, 80))

        # Buyer block
        draw.text((420, 180), "Bill To:", fill=(100, 100, 100))
        draw.text((420, 200), "Global Exports Ltd", fill=(30, 30, 30))
        draw.text((420, 218), "18, Commerce House, Mumbai 400001", fill=(80, 80, 80))
        draw.text((420, 236), "GSTIN: 27AABCG5678B1Z3", fill=(80, 80, 80))

        # Table header
        draw.rectangle([40, 310, 754, 334], fill=(240, 244, 255))
        draw.text((50, 316), "Description", fill=(50, 50, 50))
        draw.text((420, 316), "Qty", fill=(50, 50, 50))
        draw.text((490, 316), "Rate", fill=(50, 50, 50))
        draw.text((620, 316), "Amount", fill=(50, 50, 50))

        # Line items
        items = [
            ("Web Development Services", "10", "5000.00", "50000.00"),
            ("Cloud Hosting (Annual)", "1",  "12000.00", "12000.00"),
            ("SSL Certificate", "2",  "1500.00", "3000.00"),
        ]
        y = 348
        for desc, qty, rate, amt in items:
            draw.text((50, y), desc, fill=(30, 30, 30))
            draw.text((420, y), qty, fill=(30, 30, 30))
            draw.text((490, y), rate, fill=(30, 30, 30))
            draw.text((620, y), amt, fill=(30, 30, 30))
            draw.line([(40, y + 18), (754, y + 18)], fill=(230, 230, 230))
            y += 28

        # Totals
        y = 500
        draw.text((490, y),       "Subtotal:",   fill=(80, 80, 80)); draw.text((640, y), "65000.00", fill=(30, 30, 30)); y += 24
        draw.text((490, y),       "CGST (9%):",  fill=(80, 80, 80)); draw.text((640, y),  "5850.00", fill=(30, 30, 30)); y += 24
        draw.text((490, y),       "SGST (9%):",  fill=(80, 80, 80)); draw.text((640, y),  "5850.00", fill=(30, 30, 30)); y += 24
        draw.line([(490, y), (750, y)], fill=(200, 200, 200)); y += 8
        draw.text((490, y), "Grand Total:", fill=(37, 99, 235)); draw.text((640, y), "Rs. 76700.00", fill=(37, 99, 235)); y += 8

        # Payment
        draw.text((40, 620), "Payment Details:", fill=(100, 100, 100))
        draw.text((40, 640), "Bank: HDFC Bank | A/C: 50100234567890 | IFSC: HDFC0001234", fill=(80, 80, 80))
        draw.text((40, 660), "Payment Terms: Net 30 days", fill=(80, 80, 80))

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    except Exception as e:
        print(f"Could not create synthetic image: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to invoice image")
    parser.add_argument("--pdf", help="Path to invoice PDF")
    parser.add_argument("--output_dir", default="data/outputs/test", help="Output directory for PDF")
    args = parser.parse_args()

    print("=" * 60)
    print("Invoice Digitizer — Pipeline Smoke Test")
    print("=" * 60)

    # Load settings
    from config.settings import get_settings
    settings = get_settings()

    # Get file bytes
    if args.image:
        path = Path(args.image)
        if not path.exists():
            print(f"ERROR: File not found: {path}")
            sys.exit(1)
        file_bytes = path.read_bytes()
        filename = path.name
        print(f"\nInput: {filename}")
    elif args.pdf:
        path = Path(args.pdf)
        if not path.exists():
            print(f"ERROR: File not found: {path}")
            sys.exit(1)
        file_bytes = path.read_bytes()
        filename = path.name
        print(f"\nInput: {filename}")
    else:
        print("\nNo file provided — using synthetic invoice image...")
        file_bytes = create_synthetic_invoice()
        if not file_bytes:
            print("ERROR: Could not create synthetic image (PIL not installed?)")
            sys.exit(1)
        filename = "synthetic_invoice.png"
        print(f"Generated: {filename}")

    # Run pipeline
    print("\nInitialising pipeline...")
    from api.pipeline_runner import InvoicePipeline
    pipeline = InvoicePipeline(settings)

    print("\nRunning pipeline stages...\n")
    result = pipeline.process(
        file_bytes=file_bytes,
        filename=filename,
        output_dir=args.output_dir,
    )

    # Print results
    inv = result.invoice
    print("\n" + "=" * 60)
    print("EXTRACTION RESULTS")
    print("=" * 60)
    print(f"Job ID:           {result.job_id}")
    print(f"Model used:       {result.model_used}")
    print(f"Pages:            {result.page_count}")
    print(f"Confidence:       {inv.overall_confidence * 100:.1f}%")
    print(f"Needs review:     {inv.needs_review}")
    print()
    print(f"Invoice Number:   {inv.invoice_number}")
    print(f"Invoice Date:     {inv.invoice_date}")
    print(f"Vendor Name:      {inv.vendor_name}")
    print(f"Vendor GSTIN:     {inv.vendor_gstin}")
    print(f"Buyer Name:       {inv.buyer_name}")
    print(f"Subtotal:         {inv.currency} {inv.subtotal}")
    print(f"Tax Amount:       {inv.currency} {inv.tax_amount}")
    print(f"Grand Total:      {inv.currency} {inv.grand_total}")
    print(f"Line Items:       {len(inv.line_items)}")
    for i, item in enumerate(inv.line_items, 1):
        print(f"  {i}. {item.description[:40]:40s}  qty={item.quantity}  amt={item.amount}")

    print()
    vr = result.validation_report
    print(f"Validation Errors:   {len(vr.errors)}")
    for e in vr.errors:
        print(f"  [X] {e}")
    print(f"Validation Warnings: {len(vr.warnings)}")
    for w in vr.warnings:
        print(f"  [!] {w}")
 
    if result.pdf_path:
        print(f"\nPDF output: {result.pdf_path}")
 
    html_preview = Path(args.output_dir) / f"{result.job_id}.html"
    html_preview.parent.mkdir(parents=True, exist_ok=True)
    html_preview.write_text(result.html_output, encoding="utf-8")
    print(f"HTML output: {html_preview}")
 
    print("\n[OK] Pipeline smoke test complete!")


if __name__ == "__main__":
    main()
