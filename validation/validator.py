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
    unit: Optional[str] = "NOS"
    rate: float = 0.0
    discount: Optional[float] = 0.0
    taxable_value: Optional[float] = None
    amount: float = 0.0
    hsn_code: Optional[str] = None
    cgst_rate: Optional[float] = 0.0
    cgst_amount: Optional[float] = 0.0
    sgst_rate: Optional[float] = 0.0
    sgst_amount: Optional[float] = 0.0
    igst_rate: Optional[float] = 0.0
    igst_amount: Optional[float] = 0.0


class InvoiceSchema(BaseModel):
    """Canonical standardised invoice format aligned with InvoiceBuilderComponent."""
    # Meta / Header
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    po_number: Optional[str] = None
    place_of_supply: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None

    # Company / Vendor (Biller)
    vendor_name: Optional[str] = None
    vendor_address: Optional[str] = None
    vendor_address_line1: Optional[str] = None
    vendor_address_line2: Optional[str] = None
    vendor_gstin: Optional[str] = None
    vendor_pan: Optional[str] = None
    vendor_email: Optional[str] = None
    vendor_phone: Optional[str] = None

    # Client / Buyer (Bill To)
    buyer_name: Optional[str] = None
    buyer_address: Optional[str] = None
    buyer_address_line1: Optional[str] = None
    buyer_address_line2: Optional[str] = None
    buyer_gstin: Optional[str] = None
    buyer_phone: Optional[str] = None
    sls_code: Optional[str] = None

    # Line items
    line_items: list[LineItem] = Field(default_factory=list)

    # Totals & Tax
    subtotal: Optional[float] = None
    cgst: Optional[float] = None
    sgst: Optional[float] = None
    igst: Optional[float] = None
    tax_amount: Optional[float] = None
    discount: Optional[float] = None
    round_off: Optional[float] = 0.0
    grand_total: Optional[float] = None
    amount_in_words: Optional[str] = None
    currency: str = "INR"

    # Bank Details
    bank_name: Optional[str] = None
    branch_name: Optional[str] = None
    account_name: Optional[str] = None
    account_number: Optional[str] = None
    ifsc_code: Optional[str] = None
    payment_terms: Optional[str] = None
    remarks: Optional[str] = None

    overall_confidence: float = 0.0
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)

    def to_invoice_builder_json(self) -> dict:
        """
        Transforms the extracted invoice data into the exact structure
        used by ui/src/app/pages/invoice/invoice-builder/invoice-builder.component.ts.
        Allows direct form patching via `this.invoiceForm.patchValue(data)`.
        """
        v_lines = (self.vendor_address or "").strip().split("\n")
        v_addr1 = self.vendor_address_line1 or (v_lines[0] if v_lines else "")
        v_addr2 = self.vendor_address_line2 or ("\n".join(v_lines[1:]) if len(v_lines) > 1 else "")

        b_lines = (self.buyer_address or "").strip().split("\n")
        b_addr1 = self.buyer_address_line1 or (b_lines[0] if b_lines else "")
        b_addr2 = self.buyer_address_line2 or ("\n".join(b_lines[1:]) if len(b_lines) > 1 else "")

        return {
            "company": {
                "name": self.vendor_name or "",
                "addressLine1": v_addr1,
                "addressLine2": v_addr2,
                "email": self.vendor_email or "",
                "phone": self.vendor_phone or "",
                "gstin": self.vendor_gstin or "",
                "pan": self.vendor_pan or "",
            },
            "client": {
                "slsCode": self.sls_code or "",
                "name": self.buyer_name or "",
                "addressLine1": b_addr1,
                "addressLine2": b_addr2,
                "gstin": self.buyer_gstin or "",
                "phone": self.buyer_phone or "",
            },
            "meta": {
                "invoiceNo": self.invoice_number or "",
                "category": self.category or "",
                "subcategory": self.subcategory or "",
                "date": self.invoice_date or "",
                "placeOfSupply": self.place_of_supply or "",
                "dueDate": self.due_date or "",
            },
            "items": [
                {
                    "description": item.description or "",
                    "hsnSac": item.hsn_code or "",
                    "quantity": item.quantity or 1,
                    "unit": item.unit or "NOS",
                    "rate": item.rate or 0,
                    "discount": item.discount or 0,
                    "taxableValue": item.taxable_value if item.taxable_value is not None else item.amount,
                    "cgstRate": item.cgst_rate or 0,
                    "cgstAmount": item.cgst_amount or 0,
                    "sgstRate": item.sgst_rate or 0,
                    "sgstAmount": item.sgst_amount or 0,
                    "igstRate": item.igst_rate or 0,
                    "igstAmount": item.igst_amount or 0,
                }
                for item in self.line_items
            ],
            "totals": {
                "taxableAmount": self.subtotal or 0,
                "totalDiscount": self.discount or 0,
                "netTaxable": (self.subtotal or 0) - (self.discount or 0),
                "globalDiscount": 0,
                "totalCgst": self.cgst or 0,
                "totalSgst": self.sgst or 0,
                "totalIgst": self.igst or 0,
                "globalCgstRate": 0,
                "globalSgstRate": 0,
                "globalIgstRate": 0,
                "roundOff": self.round_off or 0,
                "grandTotal": self.grand_total or 0,
                "amountInWords": self.amount_in_words or "",
            },
            "bankDetails": {
                "ifsc": self.ifsc_code or "",
                "branchName": self.branch_name or "",
                "bankName": self.bank_name or "",
                "accountName": self.account_name or self.vendor_name or "",
                "accountNumber": self.account_number or "",
                "confirmAccountNumber": self.account_number or "",
            },
            "remarks": self.remarks or "",
            "certifiedRemarks": [],
        }

    @classmethod
    def from_invoice_builder_json(cls, data: dict) -> "InvoiceSchema":
        """
        Parses nested JSON from the Angular Invoice Builder or Review UI
        back into an InvoiceSchema instance.
        """
        company = data.get("company") or {}
        client = data.get("client") or {}
        meta = data.get("meta") or {}
        totals = data.get("totals") or {}
        bank = data.get("bankDetails") or {}
        raw_items = data.get("items") or []

        line_items = []
        for it in raw_items:
            if isinstance(it, dict):
                line_items.append(LineItem(
                    description=it.get("description") or "",
                    hsn_code=it.get("hsnSac") or it.get("hsn_code"),
                    quantity=float(it.get("quantity", 1) or 1),
                    unit=str(it.get("unit") or "NOS"),
                    rate=float(it.get("rate", 0) or 0),
                    discount=float(it.get("discount", 0) or 0),
                    taxable_value=float(it.get("taxableValue", it.get("taxable_value", 0)) or 0),
                    amount=float(it.get("taxableValue", it.get("amount", 0)) or 0),
                    cgst_rate=float(it.get("cgstRate", it.get("cgst_rate", 0)) or 0),
                    cgst_amount=float(it.get("cgstAmount", it.get("cgst_amount", 0)) or 0),
                    sgst_rate=float(it.get("sgstRate", it.get("sgst_rate", 0)) or 0),
                    sgst_amount=float(it.get("sgstAmount", it.get("sgst_amount", 0)) or 0),
                    igst_rate=float(it.get("igstRate", it.get("igst_rate", 0)) or 0),
                    igst_amount=float(it.get("igstAmount", it.get("igst_amount", 0)) or 0),
                ))

        v_lines = [l for l in [company.get("addressLine1"), company.get("addressLine2")] if l]
        v_addr = "\n".join(v_lines) if v_lines else None

        b_lines = [l for l in [client.get("addressLine1"), client.get("addressLine2")] if l]
        b_addr = "\n".join(b_lines) if b_lines else None

        subtotal = float(totals.get("taxableAmount", 0) or 0)
        cgst = float(totals.get("totalCgst", 0) or 0)
        sgst = float(totals.get("totalSgst", 0) or 0)
        igst = float(totals.get("totalIgst", 0) or 0)
        tax_amount = cgst + sgst + igst
        discount = float(totals.get("totalDiscount", 0) or 0)
        round_off = float(totals.get("roundOff", 0) or 0)
        grand_total = float(totals.get("grandTotal", 0) or 0)

        return cls(
            invoice_number=meta.get("invoiceNo"),
            category=meta.get("category"),
            subcategory=meta.get("subcategory"),
            invoice_date=meta.get("date"),
            place_of_supply=meta.get("placeOfSupply"),
            due_date=meta.get("dueDate"),
            vendor_name=company.get("name"),
            vendor_address=v_addr,
            vendor_address_line1=company.get("addressLine1"),
            vendor_address_line2=company.get("addressLine2"),
            vendor_email=company.get("email"),
            vendor_phone=company.get("phone"),
            vendor_gstin=company.get("gstin"),
            vendor_pan=company.get("pan"),
            buyer_name=client.get("name"),
            buyer_address=b_addr,
            buyer_address_line1=client.get("addressLine1"),
            buyer_address_line2=client.get("addressLine2"),
            buyer_gstin=client.get("gstin"),
            buyer_phone=client.get("phone"),
            sls_code=client.get("slsCode"),
            line_items=line_items,
            subtotal=subtotal,
            cgst=cgst,
            sgst=sgst,
            igst=igst,
            tax_amount=tax_amount,
            discount=discount,
            round_off=round_off,
            grand_total=grand_total,
            amount_in_words=totals.get("amountInWords"),
            bank_name=bank.get("bankName"),
            branch_name=bank.get("branchName"),
            account_name=bank.get("accountName"),
            account_number=bank.get("accountNumber"),
            ifsc_code=bank.get("ifsc"),
            remarks=data.get("remarks"),
        )


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
                            unit=str(item.get("unit", "NOS") or "NOS"),
                            rate=float(item.get("rate", 0) or 0),
                            discount=float(item.get("discount", 0) or 0),
                            taxable_value=float(item.get("taxable_value", item.get("amount", 0)) or 0),
                            amount=float(item.get("amount", 0) or 0),
                            hsn_code=item.get("hsn_code") or item.get("hsnSac"),
                            cgst_rate=float(item.get("cgst_rate", item.get("cgstRate", 0)) or 0),
                            cgst_amount=float(item.get("cgst_amount", item.get("cgstAmount", 0)) or 0),
                            sgst_rate=float(item.get("sgst_rate", item.get("sgstRate", 0)) or 0),
                            sgst_amount=float(item.get("sgst_amount", item.get("sgstAmount", 0)) or 0),
                            igst_rate=float(item.get("igst_rate", item.get("igstRate", 0)) or 0),
                            igst_amount=float(item.get("igst_amount", item.get("igstAmount", 0)) or 0),
                        ))
                    except (ValueError, TypeError):
                        pass

        return InvoiceSchema(
            invoice_number=self._get_val(inv.invoice_number),
            invoice_date=self._get_val(inv.invoice_date),
            due_date=self._get_val(inv.due_date),
            po_number=self._get_val(inv.po_number),
            place_of_supply=self._get_val(getattr(inv, "place_of_supply", None)),
            vendor_name=self._get_val(inv.vendor_name),
            vendor_address=self._get_val(inv.vendor_address),
            vendor_gstin=self._get_val(inv.vendor_gstin),
            vendor_pan=self._get_val(inv.vendor_pan),
            vendor_email=self._get_val(inv.vendor_email),
            vendor_phone=self._get_val(inv.vendor_phone),
            buyer_name=self._get_val(inv.buyer_name),
            buyer_address=self._get_val(inv.buyer_address),
            buyer_gstin=self._get_val(inv.buyer_gstin),
            sls_code=self._get_val(getattr(inv, "sls_code", None)),
            line_items=line_items,
            subtotal=self._get_float(inv.subtotal),
            cgst=self._get_float(inv.cgst),
            sgst=self._get_float(inv.sgst),
            igst=self._get_float(inv.igst),
            tax_amount=self._get_float(inv.tax_amount),
            discount=self._get_float(inv.discount),
            round_off=self._get_float(getattr(inv, "round_off", None)) or 0.0,
            grand_total=self._get_float(inv.grand_total),
            amount_in_words=self._get_val(getattr(inv, "amount_in_words", None)),
            currency=self._get_val(inv.currency) or "INR",
            bank_name=self._get_val(inv.bank_name),
            branch_name=self._get_val(getattr(inv, "branch_name", None)),
            account_name=self._get_val(getattr(inv, "account_name", None)),
            account_number=self._get_val(inv.account_number),
            ifsc_code=self._get_val(inv.ifsc_code),
            payment_terms=self._get_val(inv.payment_terms),
            remarks=self._get_val(getattr(inv, "remarks", None)),
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
