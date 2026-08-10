"""
Stage 4c: Pydantic validation + business rules engine.

Validates extracted invoice data for:
- Mathematical consistency (totals add up, tax matches rate)
- Format correctness (GSTIN, PAN, IFSC patterns)
- Completeness (required fields present)
- Logical consistency (dates, positive amounts)

Produces a ValidationReport with pass/fail for each rule
and a list of fields that need human review.
"""

import re
import json
from decimal import Decimal, InvalidOperation
from datetime import datetime
from loguru import logger
from dataclasses import dataclass, field
from typing import Optional
from pydantic import BaseModel, Field, validator, model_validator
from understanding.layoutlm import ExtractedInvoice, ExtractedField


# -------------------------------------------------------------------
# Canonical output schema (what we export)
# -------------------------------------------------------------------

class LineItem(BaseModel):
    description: str
    quantity: float = 1.0
    rate: float = 0.0
    amount: float = 0.0
    hsn_code: Optional[str] = None


class InvoiceSchema(BaseModel):
    """Canonical standardised invoice format."""
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    po_number: Optional[str] = None

    vendor_name: Optional[str] = None
    vendor_address: Optional[str] = None
    vendor_gstin: Optional[str] = None
    vendor_pan: Optional[str] = None
    vendor_email: Optional[str] = None
    vendor_phone: Optional[str] = None

    buyer_name: Optional[str] = None
    buyer_address: Optional[str] = None
    buyer_gstin: Optional[str] = None

    line_items: list[LineItem] = Field(default_factory=list)

    subtotal: Optional[float] = None
    cgst: Optional[float] = None
    sgst: Optional[float] = None
    igst: Optional[float] = None
    tax_amount: Optional[float] = None
    discount: Optional[float] = None
    grand_total: Optional[float] = None
    currency: str = "INR"

    bank_name: Optional[str] = None
    account_number: Optional[str] = None
    ifsc_code: Optional[str] = None
    payment_terms: Optional[str] = None

    overall_confidence: float = 0.0
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)


# -------------------------------------------------------------------
# Validation report
# -------------------------------------------------------------------

@dataclass
class RuleResult:
    rule: str
    passed: bool
    message: str
    severity: str  # "error", "warning", "info"


@dataclass
class ValidationReport:
    results: list[RuleResult] = field(default_factory=list)
    is_valid: bool = True
    needs_review: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add(self, rule: str, passed: bool, message: str, severity: str = "error"):
        self.results.append(RuleResult(rule, passed, message, severity))
        if not passed:
            if severity == "error":
                self.is_valid = False
                self.errors.append(message)
                self.needs_review = True
            elif severity == "warning":
                self.warnings.append(message)
                self.needs_review = True


# -------------------------------------------------------------------
# Validator
# -------------------------------------------------------------------

GSTIN_RE = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d]$")
PAN_RE   = re.compile(r"^[A-Z]{5}\d{4}[A-Z]$")
IFSC_RE  = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")


class InvoiceValidator:
    """
    Applies business rules to an ExtractedInvoice and produces
    a canonicalised InvoiceSchema + ValidationReport.
    """

    TOLERANCE = 0.02   # 2% tolerance for floating-point rounding in OCR amounts

    def validate(self, invoice: ExtractedInvoice) -> tuple[InvoiceSchema, ValidationReport]:
        report = ValidationReport()
        schema = self._to_schema(invoice)

        self._check_required_fields(schema, report)
        self._check_formats(schema, report)
        self._check_math(schema, report)
        self._check_dates(schema, report)
        self._check_amounts(schema, report)

        schema.needs_review = report.needs_review
        schema.review_reasons = report.errors + report.warnings

        logger.info(
            f"Validation: valid={report.is_valid}, "
            f"errors={len(report.errors)}, warnings={len(report.warnings)}"
        )
        return schema, report

    # ------------------------------------------------------------------

    def _get_val(self, field: Optional[ExtractedField]) -> Optional[str]:
        return field.value if field else None

    def _get_float(self, field: Optional[ExtractedField]) -> Optional[float]:
        if not field or not field.value:
            return None
        try:
            # Keep leading minus sign for negative detection
            is_neg = field.value.strip().startswith("-")
            cleaned = re.sub(r"[^\d.]", "", field.value)
            if not cleaned:
                return None
            val = float(cleaned)
            return -val if is_neg else val
        except ValueError:
            return None

    def _to_schema(self, inv: ExtractedInvoice) -> InvoiceSchema:
        """Convert ExtractedInvoice to canonical InvoiceSchema."""
        line_items = []
        if inv.line_items:
            for item in inv.line_items:
                if isinstance(item, dict):
                    try:
                        line_items.append(LineItem(
                            description=item.get("description", ""),
                            quantity=float(item.get("quantity", 1) or 1),
                            rate=float(item.get("rate", 0) or 0),
                            amount=float(item.get("amount", 0) or 0),
                        ))
                    except (ValueError, TypeError):
                        pass

        return InvoiceSchema(
            invoice_number=self._get_val(inv.invoice_number),
            invoice_date=self._get_val(inv.invoice_date),
            due_date=self._get_val(inv.due_date),
            po_number=self._get_val(inv.po_number),
            vendor_name=self._get_val(inv.vendor_name),
            vendor_address=self._get_val(inv.vendor_address),
            vendor_gstin=self._get_val(inv.vendor_gstin),
            vendor_pan=self._get_val(inv.vendor_pan),
            vendor_email=self._get_val(inv.vendor_email),
            vendor_phone=self._get_val(inv.vendor_phone),
            buyer_name=self._get_val(inv.buyer_name),
            buyer_address=self._get_val(inv.buyer_address),
            buyer_gstin=self._get_val(inv.buyer_gstin),
            line_items=line_items,
            subtotal=self._get_float(inv.subtotal),
            cgst=self._get_float(inv.cgst),
            sgst=self._get_float(inv.sgst),
            igst=self._get_float(inv.igst),
            tax_amount=self._get_float(inv.tax_amount),
            discount=self._get_float(inv.discount),
            grand_total=self._get_float(inv.grand_total),
            currency=self._get_val(inv.currency) or "INR",
            bank_name=self._get_val(inv.bank_name),
            account_number=self._get_val(inv.account_number),
            ifsc_code=self._get_val(inv.ifsc_code),
            payment_terms=self._get_val(inv.payment_terms),
            overall_confidence=inv.overall_confidence,
        )

    def _check_required_fields(self, s: InvoiceSchema, r: ValidationReport):
        required = {
            "invoice_number": s.invoice_number,
            "vendor_name": s.vendor_name,
            "grand_total": s.grand_total,
        }
        for name, val in required.items():
            if not val:
                r.add(f"required_{name}", False,
                      f"Required field missing: {name}", "error")
            else:
                r.add(f"required_{name}", True, f"{name} present", "info")

        recommended = {
            "invoice_date": s.invoice_date,
            "buyer_name": s.buyer_name,
            "subtotal": s.subtotal,
        }
        for name, val in recommended.items():
            if not val:
                r.add(f"recommended_{name}", False,
                      f"Recommended field missing: {name}", "warning")

    def _check_formats(self, s: InvoiceSchema, r: ValidationReport):
        if s.vendor_gstin:
            clean = s.vendor_gstin.replace(" ", "").upper()
            ok = bool(GSTIN_RE.match(clean))
            r.add("gstin_format", ok,
                  "Vendor GSTIN format valid" if ok else f"Invalid GSTIN format: {s.vendor_gstin}",
                  "error" if not ok else "info")

        if s.buyer_gstin:
            clean = s.buyer_gstin.replace(" ", "").upper()
            ok = bool(GSTIN_RE.match(clean))
            r.add("buyer_gstin_format", ok,
                  "Buyer GSTIN format valid" if ok else f"Invalid buyer GSTIN: {s.buyer_gstin}",
                  "warning")

        if s.vendor_pan:
            ok = bool(PAN_RE.match(s.vendor_pan.upper()))
            r.add("pan_format", ok,
                  "PAN format valid" if ok else f"Invalid PAN format: {s.vendor_pan}",
                  "warning")

        if s.ifsc_code:
            ok = bool(IFSC_RE.match(s.ifsc_code.upper()))
            r.add("ifsc_format", ok,
                  "IFSC format valid" if ok else f"Invalid IFSC: {s.ifsc_code}",
                  "warning")

    def _check_math(self, s: InvoiceSchema, r: ValidationReport):
        """
        Core financial consistency checks.
        """
        tol = self.TOLERANCE

        # 1. Line items sum = subtotal
        if s.line_items and s.subtotal:
            items_sum = sum(item.amount for item in s.line_items)
            diff = abs(items_sum - s.subtotal)
            ok = diff <= max(s.subtotal * tol, 1.0)
            r.add("line_items_sum",
                  ok,
                  f"Line items sum {items_sum:.2f} matches subtotal {s.subtotal:.2f}" if ok
                  else f"Line items sum {items_sum:.2f} ≠ subtotal {s.subtotal:.2f} (diff={diff:.2f})",
                  "error" if not ok else "info")

        # 2. CGST + SGST = tax_amount (if using CGST/SGST)
        if s.cgst is not None and s.sgst is not None and s.tax_amount:
            computed = s.cgst + s.sgst
            diff = abs(computed - s.tax_amount)
            ok = diff <= max(s.tax_amount * tol, 1.0)
            r.add("cgst_sgst_sum", ok,
                  f"CGST+SGST={computed:.2f} matches tax={s.tax_amount:.2f}" if ok
                  else f"CGST({s.cgst})+SGST({s.sgst})={computed:.2f} ≠ tax={s.tax_amount:.2f}",
                  "error" if not ok else "info")

        # 3. subtotal + tax - discount = grand_total
        if s.subtotal and s.grand_total and s.tax_amount is not None:
            discount = s.discount or 0.0
            computed_total = s.subtotal + s.tax_amount - discount
            diff = abs(computed_total - s.grand_total)
            ok = diff <= max(s.grand_total * tol, 1.0)
            r.add("grand_total_math", ok,
                  f"Grand total math checks out ({computed_total:.2f} ≈ {s.grand_total:.2f})" if ok
                  else f"Grand total mismatch: {s.subtotal}+{s.tax_amount}-{discount}={computed_total:.2f} ≠ {s.grand_total:.2f}",
                  "error" if not ok else "info")

    def _check_dates(self, s: InvoiceSchema, r: ValidationReport):
        date_formats = ["%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d %b %Y", "%d %B %Y"]

        def parse_date(d: str) -> Optional[datetime]:
            for fmt in date_formats:
                try:
                    return datetime.strptime(d.strip(), fmt)
                except ValueError:
                    continue
            return None

        if s.invoice_date:
            dt = parse_date(s.invoice_date)
            r.add("invoice_date_parseable",
                  dt is not None,
                  f"Invoice date parseable: {s.invoice_date}" if dt else f"Cannot parse date: {s.invoice_date}",
                  "warning")

        if s.invoice_date and s.due_date:
            inv_dt = parse_date(s.invoice_date)
            due_dt = parse_date(s.due_date)
            if inv_dt and due_dt:
                ok = due_dt >= inv_dt
                r.add("due_date_after_invoice", ok,
                      "Due date is after invoice date" if ok
                      else f"Due date {s.due_date} is before invoice date {s.invoice_date}",
                      "warning")

    def _check_amounts(self, s: InvoiceSchema, r: ValidationReport):
        for field_name, val in [
            ("subtotal", s.subtotal),
            ("tax_amount", s.tax_amount),
            ("grand_total", s.grand_total),
        ]:
            if val is not None:
                ok = val >= 0
                r.add(f"{field_name}_positive", ok,
                      f"{field_name} is positive ({val:.2f})" if ok
                      else f"{field_name} is negative: {val:.2f}",
                      "error" if not ok else "info")

        if s.subtotal and s.grand_total:
            ok = s.grand_total >= s.subtotal
            r.add("grand_gte_subtotal", ok,
                  "Grand total >= subtotal" if ok
                  else f"Grand total {s.grand_total} < subtotal {s.subtotal}",
                  "warning")
