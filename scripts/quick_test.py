"""Quick end-to-end pipeline validation test."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings
from api.pipeline_runner import InvoicePipeline

def main():
    settings = get_settings()
    pipeline = InvoicePipeline(settings)

    test_files = [
        ("data/raw/ceddc6c6-9705-4cb1-99bb-f05dc4d0d094_Invoice_52.pdf", "Invoice_52"),
        ("data/raw/bbf5a93e-4c10-400f-97bb-fb835a40074c_b97835cc-126a-401f-b530-abd0623c7a4c.pdf", "bbf5a93e"),
        ("data/raw/VBMS_Bill_Details04_08_2026 .pdf", "VBMS"),
    ]

    for fpath, label in test_files:
        if not Path(fpath).exists():
            print(f"SKIP: {fpath}")
            continue
        with open(fpath, "rb") as f:
            data = f.read()
        t0 = time.time()
        result = pipeline.process(data, filename=Path(fpath).name)
        elapsed = time.time() - t0
        inv = result.invoice
        valid = result.validation_report.is_valid
        print()
        print(f"=== {label} ({elapsed:.1f}s | {result.model_used}) valid={valid} ===")
        print(f"  Invoice No:  {inv.invoice_number!r}")
        print(f"  Date:        {inv.invoice_date!r}")
        print(f"  Vendor:      {inv.vendor_name!r}")
        print(f"  Buyer:       {inv.buyer_name!r}")
        print(f"  GSTIN:       {inv.vendor_gstin!r}")
        print(f"  Grand Total: {inv.grand_total!r}")
        print(f"  Subtotal:    {inv.subtotal!r}")
        print(f"  Tax Amount:  {inv.tax_amount!r}")
        print(f"  CGST/SGST:   {inv.cgst!r} / {inv.sgst!r}")
        print(f"  IFSC:        {inv.ifsc_code!r}")
        print(f"  Confidence:  {inv.overall_confidence:.1%}")


if __name__ == "__main__":
    main()
