"""
Stage 5: Output renderer.
Generates a professional PDF and HTML invoice from the validated InvoiceSchema.
Uses WeasyPrint + Jinja2 — 100% free and offline.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from loguru import logger
from jinja2 import Environment, DictLoader
from typing import Optional
from validation.validator import InvoiceSchema


INVOICE_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 12px;
         color: #1a1a1a; background: #fff; padding: 40px; }
  .invoice-wrapper { max-width: 800px; margin: 0 auto; }

  /* Header */
  .inv-header { display: flex; justify-content: space-between; align-items: flex-start;
                margin-bottom: 32px; padding-bottom: 20px;
                border-bottom: 2px solid #2563eb; }
  .inv-title { font-size: 28px; font-weight: 700; color: #2563eb; letter-spacing: -0.5px; }
  .inv-meta { text-align: right; }
  .inv-meta .inv-number { font-size: 16px; font-weight: 600; color: #111; }
  .inv-meta .inv-date { color: #666; margin-top: 4px; }

  /* Address blocks */
  .address-row { display: flex; gap: 40px; margin-bottom: 28px; }
  .address-block { flex: 1; }
  .address-block h3 { font-size: 10px; font-weight: 700; text-transform: uppercase;
                      letter-spacing: 1px; color: #2563eb; margin-bottom: 8px; }
  .address-block p { line-height: 1.6; color: #333; }
  .address-block .gstin { font-size: 11px; color: #666; margin-top: 4px;
                           font-family: monospace; }

  /* Line items table */
  .items-table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
  .items-table thead tr { background: #2563eb; color: white; }
  .items-table thead th { padding: 10px 12px; text-align: left;
                           font-size: 11px; font-weight: 600; }
  .items-table thead th.num { text-align: right; }
  .items-table tbody tr:nth-child(even) { background: #f8faff; }
  .items-table tbody td { padding: 9px 12px; border-bottom: 1px solid #e5e7eb; }
  .items-table tbody td.num { text-align: right; font-family: monospace; }

  /* Totals */
  .totals-wrapper { display: flex; justify-content: flex-end; margin-bottom: 28px; }
  .totals-table { width: 300px; }
  .totals-table tr td { padding: 6px 0; }
  .totals-table tr td:last-child { text-align: right; font-family: monospace; }
  .totals-table .label { color: #555; }
  .totals-table .divider td { border-top: 1px solid #e5e7eb; padding-top: 8px; }
  .totals-table .grand td { font-size: 15px; font-weight: 700; color: #2563eb; }

  /* Payment + badges */
  .bottom-row { display: flex; gap: 40px; margin-top: 20px; }
  .payment-block { flex: 1; }
  .payment-block h3 { font-size: 10px; font-weight: 700; text-transform: uppercase;
                       letter-spacing: 1px; color: #2563eb; margin-bottom: 8px; }
  .payment-block p { line-height: 1.8; color: #333; font-size: 11px; }

  /* Status badge */
  .status-block { display: flex; flex-direction: column; align-items: flex-end; gap: 8px; }
  .badge { display: inline-block; padding: 4px 12px; border-radius: 20px;
           font-size: 10px; font-weight: 700; letter-spacing: 0.5px; }
  .badge-review { background: #fef3c7; color: #92400e; }
  .badge-ok { background: #d1fae5; color: #065f46; }
  .badge-currency { background: #eff6ff; color: #1e40af; }

  /* Footer */
  .footer { margin-top: 40px; padding-top: 16px; border-top: 1px solid #e5e7eb;
             font-size: 10px; color: #aaa; text-align: center; }

  /* Review warnings */
  .review-box { background: #fff7ed; border: 1px solid #f97316; border-radius: 6px;
                 padding: 12px 16px; margin-bottom: 20px; }
  .review-box h4 { color: #c2410c; font-size: 11px; margin-bottom: 6px; }
  .review-box ul { padding-left: 16px; color: #7c2d12; font-size: 11px; line-height: 1.8; }

  @media print {
    body { padding: 20px; }
    .review-box { display: none; }
  }
</style>
</head>
<body>
<div class="invoice-wrapper">

  {% if invoice.needs_review and invoice.review_reasons %}
  <div class="review-box">
    <h4>⚠ Needs Review — {{ invoice.review_reasons|length }} issue(s) found</h4>
    <ul>
      {% for reason in invoice.review_reasons %}<li>{{ reason }}</li>{% endfor %}
    </ul>
  </div>
  {% endif %}

  <!-- Header -->
  <div class="inv-header">
    <div>
      <div class="inv-title">INVOICE</div>
      {% if invoice.vendor_name %}
      <div style="font-size:14px; font-weight:600; margin-top:6px;">{{ invoice.vendor_name }}</div>
      {% endif %}
      {% if invoice.vendor_address %}
      <div style="color:#555; font-size:11px; margin-top:2px;">{{ invoice.vendor_address }}</div>
      {% endif %}
      {% if invoice.vendor_gstin %}
      <div class="gstin" style="margin-top:4px;">GSTIN: {{ invoice.vendor_gstin }}</div>
      {% endif %}
    </div>
    <div class="inv-meta">
      <div class="inv-number">{{ invoice.invoice_number or 'N/A' }}</div>
      <div class="inv-date">Date: {{ invoice.invoice_date or '—' }}</div>
      {% if invoice.due_date %}
      <div class="inv-date">Due: {{ invoice.due_date }}</div>
      {% endif %}
      {% if invoice.po_number %}
      <div class="inv-date">PO: {{ invoice.po_number }}</div>
      {% endif %}
    </div>
  </div>

  <!-- Address row -->
  <div class="address-row">
    <div class="address-block">
      <h3>Bill From</h3>
      <p>{{ invoice.vendor_name or '—' }}</p>
      {% if invoice.vendor_address %}<p>{{ invoice.vendor_address }}</p>{% endif %}
      {% if invoice.vendor_email %}<p>{{ invoice.vendor_email }}</p>{% endif %}
      {% if invoice.vendor_phone %}<p>{{ invoice.vendor_phone }}</p>{% endif %}
      {% if invoice.vendor_gstin %}<p class="gstin">GSTIN: {{ invoice.vendor_gstin }}</p>{% endif %}
      {% if invoice.vendor_pan %}<p class="gstin">PAN: {{ invoice.vendor_pan }}</p>{% endif %}
    </div>
    <div class="address-block">
      <h3>Bill To</h3>
      <p>{{ invoice.buyer_name or '—' }}</p>
      {% if invoice.buyer_address %}<p>{{ invoice.buyer_address }}</p>{% endif %}
      {% if invoice.buyer_gstin %}<p class="gstin">GSTIN: {{ invoice.buyer_gstin }}</p>{% endif %}
    </div>
  </div>

  <!-- Line items -->
  {% if invoice.line_items %}
  <table class="items-table">
    <thead>
      <tr>
        <th>#</th>
        <th>Description</th>
        <th class="num">Qty</th>
        <th class="num">Rate ({{ invoice.currency }})</th>
        <th class="num">Amount ({{ invoice.currency }})</th>
      </tr>
    </thead>
    <tbody>
      {% for item in invoice.line_items %}
      <tr>
        <td>{{ loop.index }}</td>
        <td>{{ item.description }}</td>
        <td class="num">{{ item.quantity }}</td>
        <td class="num">{{ "%.2f"|format(item.rate) }}</td>
        <td class="num">{{ "%.2f"|format(item.amount) }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
  {% else %}
  <div style="color:#999; font-style:italic; margin-bottom:20px; font-size:11px;">
    No line items extracted — see original invoice for details.
  </div>
  {% endif %}

  <!-- Totals -->
  <div class="totals-wrapper">
    <table class="totals-table">
      {% if invoice.subtotal is not none %}
      <tr><td class="label">Subtotal</td><td>{{ invoice.currency }} {{ "%.2f"|format(invoice.subtotal) }}</td></tr>
      {% endif %}
      {% if invoice.discount %}
      <tr><td class="label">Discount</td><td>- {{ invoice.currency }} {{ "%.2f"|format(invoice.discount) }}</td></tr>
      {% endif %}
      {% if invoice.cgst is not none %}
      <tr><td class="label">CGST</td><td>{{ invoice.currency }} {{ "%.2f"|format(invoice.cgst) }}</td></tr>
      {% endif %}
      {% if invoice.sgst is not none %}
      <tr><td class="label">SGST</td><td>{{ invoice.currency }} {{ "%.2f"|format(invoice.sgst) }}</td></tr>
      {% endif %}
      {% if invoice.igst is not none %}
      <tr><td class="label">IGST</td><td>{{ invoice.currency }} {{ "%.2f"|format(invoice.igst) }}</td></tr>
      {% endif %}
      {% if invoice.tax_amount is not none and not (invoice.cgst or invoice.sgst or invoice.igst) %}
      <tr><td class="label">Tax</td><td>{{ invoice.currency }} {{ "%.2f"|format(invoice.tax_amount) }}</td></tr>
      {% endif %}
      <tr class="divider"><td></td><td></td></tr>
      <tr class="grand">
        <td>Total Due</td>
        <td>{{ invoice.currency }} {{ "%.2f"|format(invoice.grand_total or 0) }}</td>
      </tr>
    </table>
  </div>

  <!-- Payment + status -->
  <div class="bottom-row">
    {% if invoice.bank_name or invoice.account_number or invoice.ifsc_code %}
    <div class="payment-block">
      <h3>Payment Details</h3>
      {% if invoice.bank_name %}<p>Bank: {{ invoice.bank_name }}</p>{% endif %}
      {% if invoice.account_number %}<p>Account: {{ invoice.account_number }}</p>{% endif %}
      {% if invoice.ifsc_code %}<p>IFSC: {{ invoice.ifsc_code }}</p>{% endif %}
      {% if invoice.payment_terms %}<p>Terms: {{ invoice.payment_terms }}</p>{% endif %}
    </div>
    {% endif %}
    <div class="status-block">
      <span class="badge badge-currency">{{ invoice.currency }}</span>
      {% if invoice.needs_review %}
      <span class="badge badge-review">⚠ NEEDS REVIEW</span>
      {% else %}
      <span class="badge badge-ok">✓ VALIDATED</span>
      {% endif %}
      <span style="font-size:10px; color:#aaa;">
        Confidence: {{ "%.0f"|format(invoice.overall_confidence * 100) }}%
      </span>
    </div>
  </div>

  <div class="footer">
    Generated by Invoice Digitizer · {{ generated_at }} · This is a digitised representation of the original invoice.
  </div>

</div>
</body>
</html>
"""


class InvoiceRenderer:
    """
    Renders a validated InvoiceSchema to HTML and PDF.
    """

    def __init__(self):
        self.env = Environment(loader=DictLoader({"invoice.html": INVOICE_HTML_TEMPLATE}))

    def to_html(self, invoice: InvoiceSchema) -> str:
        """Render invoice as HTML string."""
        template = self.env.get_template("invoice.html")
        return template.render(
            invoice=invoice,
            generated_at=datetime.now().strftime("%d %b %Y %H:%M"),
        )

    def to_pdf(self, invoice: InvoiceSchema, output_path: str | Path) -> Optional[Path]:
        """Render invoice as PDF using WeasyPrint."""
        try:
            from weasyprint import HTML
            html_content = self.to_html(invoice)
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            HTML(string=html_content).write_pdf(str(output_path))
            logger.info(f"PDF rendered: {output_path}")
            return output_path
        except Exception as e:
            logger.warning(f"WeasyPrint PDF rendering skipped or failed (missing GTK/system dependencies?): {e}")
            return None

    def to_json(self, invoice: InvoiceSchema) -> str:
        """Export invoice as canonical JSON."""
        return invoice.model_dump_json(indent=2)

    def to_dict(self, invoice: InvoiceSchema) -> dict:
        """Export invoice as dict."""
        return invoice.model_dump()
