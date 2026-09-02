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
    description: str = ""
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

    class Config:
        extra = "allow"


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
    columns: list[dict] = Field(default_factory=list)

    # Totals & Tax
    subtotal: Optional[float] = None
    cgst: Optional[float] = None
    sgst: Optional[float] = None
    igst: Optional[float] = None
    tax_amount: Optional[float] = None
    discount: Optional[float] = None
    global_discount: Optional[float] = 0.0
    global_cgst_rate: Optional[float] = 0.0
    global_sgst_rate: Optional[float] = 0.0
    global_igst_rate: Optional[float] = 0.0
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
    certified_remarks: list[str] = Field(default_factory=list)

    overall_confidence: float = 0.0
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    field_confidences: dict[str, float] = Field(default_factory=dict)
    fields_needing_review: list[str] = Field(default_factory=list)
    auto_accepted_fields: list[str] = Field(default_factory=list)
    template_id: Optional[str] = None
    template_family_id: Optional[str] = None
    template_version_id: Optional[str] = None
    disagreement_score: Optional[float] = 0.0
    is_novel_template: bool = False
    ground_truth_source: str = "auto_accepted"
    field_provenance: dict[str, dict] = Field(default_factory=dict)

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

        items = []
        for item in self.line_items:
            item_dict = item.model_dump() if hasattr(item, "model_dump") else item.dict()
            base_item = {
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
            # Include any custom dynamic column fields
            for k, v in item_dict.items():
                if k not in base_item and k not in ("amount", "hsn_code", "cgst_rate", "cgst_amount", "sgst_rate", "sgst_amount", "igst_rate", "igst_amount", "taxable_value"):
                    base_item[k] = v
            items.append(base_item)

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
                "poNumber": self.po_number or "",
                "category": self.category or "",
                "subcategory": self.subcategory or "",
                "date": self.invoice_date or "",
                "placeOfSupply": self.place_of_supply or "",
                "dueDate": self.due_date or "",
            },
            "columns": self.columns or [],
            "items": items,
            "totals": {
                "taxableAmount": self.subtotal or 0,
                "totalDiscount": self.discount or 0,
                "netTaxable": (self.subtotal or 0) - (self.discount or 0) - (self.global_discount or 0),
                "globalDiscount": self.global_discount or 0,
                "totalCgst": self.cgst or 0,
                "totalSgst": self.sgst or 0,
                "totalIgst": self.igst or 0,
                "globalCgstRate": self.global_cgst_rate or 0,
                "globalSgstRate": self.global_sgst_rate or 0,
                "globalIgstRate": self.global_igst_rate or 0,
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
                "paymentTerms": self.payment_terms or "",
            },
            "remarks": self.remarks or "",
            "certifiedRemarks": self.certified_remarks or [],
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
        columns = data.get("columns") or []
        certified_remarks = data.get("certifiedRemarks") or data.get("certified_remarks") or []

        line_items = []
        for it in raw_items:
            if isinstance(it, dict):
                item_data = {
                    "description": it.get("description") or "",
                    "hsn_code": it.get("hsnSac") or it.get("hsn_code"),
                    "quantity": float(it.get("quantity", 1) or 1),
                    "unit": str(it.get("unit") or "NOS"),
                    "rate": float(it.get("rate", 0) or 0),
                    "discount": float(it.get("discount", 0) or 0),
                    "taxable_value": float(it.get("taxableValue", it.get("taxable_value", 0)) or 0),
                    "amount": float(it.get("taxableValue", it.get("amount", 0)) or 0),
                    "cgst_rate": float(it.get("cgstRate", it.get("cgst_rate", 0)) or 0),
                    "cgst_amount": float(it.get("cgstAmount", it.get("cgst_amount", 0)) or 0),
                    "sgst_rate": float(it.get("sgstRate", it.get("sgst_rate", 0)) or 0),
                    "sgst_amount": float(it.get("sgstAmount", it.get("sgst_amount", 0)) or 0),
                    "igst_rate": float(it.get("igstRate", it.get("igst_rate", 0)) or 0),
                    "igst_amount": float(it.get("igstAmount", it.get("igst_amount", 0)) or 0),
                }
                # Preserve any custom dynamic column fields
                for k, v in it.items():
                    if k not in item_data and k not in ("hsnSac", "cgstRate", "cgstAmount", "sgstRate", "sgstAmount", "igstRate", "igstAmount", "taxableValue"):
                        item_data[k] = v
                line_items.append(LineItem(**item_data))

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
        global_discount = float(totals.get("globalDiscount", 0) or 0)
        global_cgst_rate = float(totals.get("globalCgstRate", 0) or 0)
        global_sgst_rate = float(totals.get("globalSgstRate", 0) or 0)
        global_igst_rate = float(totals.get("globalIgstRate", 0) or 0)
        round_off = float(totals.get("roundOff", 0) or 0)
        grand_total = float(totals.get("grandTotal", 0) or 0)

        return cls(
            invoice_number=meta.get("invoiceNo") or meta.get("invoice_number"),
            po_number=meta.get("poNumber") or meta.get("po_number"),
            category=meta.get("category"),
            subcategory=meta.get("subcategory"),
            invoice_date=meta.get("date") or meta.get("invoice_date"),
            place_of_supply=meta.get("placeOfSupply") or meta.get("place_of_supply"),
            due_date=meta.get("dueDate") or meta.get("due_date"),
            vendor_name=company.get("name") or company.get("vendor_name"),
            vendor_address=v_addr,
            vendor_address_line1=company.get("addressLine1"),
            vendor_address_line2=company.get("addressLine2"),
            vendor_email=company.get("email"),
            vendor_phone=company.get("phone"),
            vendor_gstin=company.get("gstin"),
            vendor_pan=company.get("pan"),
            buyer_name=client.get("name") or client.get("buyer_name"),
            buyer_address=b_addr,
            buyer_address_line1=client.get("addressLine1"),
            buyer_address_line2=client.get("addressLine2"),
            buyer_gstin=client.get("gstin"),
            buyer_phone=client.get("phone"),
            sls_code=client.get("slsCode") or client.get("sls_code"),
            line_items=line_items,
            columns=columns,
            subtotal=subtotal,
            cgst=cgst,
            sgst=sgst,
            igst=igst,
            tax_amount=tax_amount,
            discount=discount,
            global_discount=global_discount,
            global_cgst_rate=global_cgst_rate,
            global_sgst_rate=global_sgst_rate,
            global_igst_rate=global_igst_rate,
            round_off=round_off,
            grand_total=grand_total,
            amount_in_words=totals.get("amountInWords") or totals.get("amount_in_words"),
            bank_name=bank.get("bankName") or bank.get("bank_name"),
            branch_name=bank.get("branchName") or bank.get("branch_name"),
            account_name=bank.get("accountName") or bank.get("account_name"),
            account_number=bank.get("accountNumber") or bank.get("account_number"),
            ifsc_code=bank.get("ifsc") or bank.get("ifsc_code"),
            payment_terms=bank.get("paymentTerms") or data.get("paymentTerms") or data.get("payment_terms"),
            remarks=data.get("remarks"),
            certified_remarks=certified_remarks,
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
GST_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def verify_gstin_checksum(gstin: Optional[str]) -> bool:
    """
    Verifies 15-character Indian GSTIN checksum using Modulo 36 algorithm.
    Structure:
      - Positions 1-2: State code (01-37)
      - Positions 3-12: PAN (5 letters + 4 digits + 1 letter)
      - Position 13: Entity code (1-9, A-Z)
      - Position 14: Default 'Z'
      - Position 15: Check digit (0-9, A-Z)
    """
    if not gstin:
        return False
    clean = gstin.strip().upper().replace(" ", "")
    if len(clean) != 15 or not GSTIN_RE.match(clean):
        return False

    try:
        chars = list(clean)
        total = 0
        for i in range(14):
            c = chars[i]
            if c not in GST_CHARS:
                return False
            val = GST_CHARS.index(c)
            multiplier = 1 if (i % 2 == 0) else 2
            product = val * multiplier
            quotient = product // 36
            remainder = product % 36
            total += quotient + remainder

        rem = total % 36
        check_idx = (36 - rem) % 36
        expected_char = GST_CHARS[check_idx]
        return chars[14] == expected_char
    except Exception:
        return False

GST_STATE_CODES: dict[str, str] = {
    "01": "Jammu & Kashmir", "02": "Himachal Pradesh", "03": "Punjab", "04": "Chandigarh",
    "05": "Uttarakhand", "06": "Haryana", "07": "Delhi", "08": "Rajasthan", "09": "Uttar Pradesh",
    "10": "Bihar", "11": "Sikkim", "12": "Arunachal Pradesh", "13": "Nagaland", "14": "Manipur",
    "15": "Mizoram", "16": "Tripura", "17": "Meghalaya", "18": "Assam", "19": "West Bengal",
    "20": "Jharkhand", "21": "Odisha", "22": "Chhattisgarh", "23": "Madhya Pradesh",
    "24": "Gujarat", "27": "Maharashtra", "29": "Karnataka", "30": "Goa", "32": "Kerala",
    "33": "Tamil Nadu", "36": "Telangana", "37": "Andhra Pradesh"
}


def number_to_words_inr(num: Optional[float]) -> str:
    """Convert number to Indian currency words format."""
    if not num or num <= 0:
        return ""
    a = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
         "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
    b = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]

    def in_words(n):
        s = ""
        if n > 99:
            s += a[n // 100] + " Hundred "
            n %= 100
        if n > 19:
            s += b[n // 10] + (" " + a[n % 10] if n % 10 else "")
        elif n > 0:
            s += a[n]
        return s.strip()

    whole = int(num)
    paise = round((num - whole) * 100)
    res = ""
    crore = whole // 10000000
    whole %= 10000000
    lakh = whole // 100000
    whole %= 100000
    thousand = whole // 1000
    whole %= 1000
    hundred = whole

    if crore: res += in_words(crore) + " Crore "
    if lakh: res += in_words(lakh) + " Lakh "
    if thousand: res += in_words(thousand) + " Thousand "
    if hundred: res += in_words(hundred) + " "

    res = res.strip()
    if res: res = "Rupees " + res
    if paise > 0:
        res += (" and " if res else "Rupees ") + in_words(paise) + " Paise"
    return (res + " Only") if res else ""


class InvoiceValidator:
    """
    Applies business rules to an ExtractedInvoice and produces
    a canonicalised InvoiceSchema + ValidationReport.
    """

    TOLERANCE = 0.02   # 2% tolerance for floating-point rounding in OCR amounts

    def validate(
        self,
        invoice: ExtractedInvoice,
        doc_type: str = "PRINTED_SCAN",
        handwriting_level: str = "NONE",
        handwriting_penalty: float = 0.85,
    ) -> tuple[InvoiceSchema, ValidationReport]:
        report = ValidationReport()
        schema = self._to_schema(invoice)

        # Cross-field mathematical & consistency auto-repair
        schema = self._auto_repair(schema)

        self._check_required_fields(schema, report)
        self._check_formats(schema, report)
        self._check_math(schema, report)
        self._check_dates(schema, report)
        self._check_amounts(schema, report)

        # Multi-evidence field-level confidence and arithmetic consistency evaluation
        from validation.confidence_engine import FieldConfidenceEngine
        conf_eval = FieldConfidenceEngine().evaluate(
            schema.model_dump(),
            ocr_avg_conf=0.90,
            handwriting_level=handwriting_level,
            doc_type=doc_type,
            handwriting_penalty=handwriting_penalty,
        )

        schema.field_confidences = conf_eval["field_confidences"]
        schema.fields_needing_review = conf_eval["fields_needing_review"]
        schema.auto_accepted_fields = conf_eval["auto_accepted_fields"]
        schema.overall_confidence = conf_eval["overall_confidence"]
        schema.needs_review = report.needs_review or conf_eval["needs_review"]
        schema.review_reasons = list(dict.fromkeys(report.errors + report.warnings + conf_eval["review_reasons"]))

        logger.info(
            f"Validation: valid={report.is_valid}, "
            f"errors={len(report.errors)}, warnings={len(report.warnings)}, "
            f"review_fields={len(schema.fields_needing_review)}, auto_accepted={len(schema.auto_accepted_fields)}"
        )
        return schema, report

    def _auto_repair(self, s: InvoiceSchema) -> InvoiceSchema:
        """
        Cross-field consistency and auto-repair logic before validation.
        Fixes common arithmetic and field omissions using known relationships.
        """
        # 1. Compute subtotal from line items if subtotal is missing
        if s.line_items and (s.subtotal is None or s.subtotal == 0):
            item_sum = sum(item.amount for item in s.line_items if item.amount)
            if item_sum > 0:
                s.subtotal = round(item_sum, 2)

        # 2. Total tax calculation from CGST + SGST + IGST if tax_amount is missing
        taxes = [s.cgst, s.sgst, s.igst]
        valid_taxes = [t for t in taxes if t is not None]
        if valid_taxes and (s.tax_amount is None or s.tax_amount == 0):
            s.tax_amount = round(sum(valid_taxes), 2)

        # 3. Derive grand_total if subtotal + tax_amount are present
        if s.subtotal is not None and s.tax_amount is not None and (s.grand_total is None or s.grand_total == 0):
            computed = s.subtotal + s.tax_amount - (s.discount or 0.0) + (s.round_off or 0.0)
            if computed > 0:
                s.grand_total = round(computed, 2)

        # 4. Derive subtotal from grand_total if grand_total is present and tax_amount is known
        if s.grand_total is not None and s.tax_amount is not None and (s.subtotal is None or s.subtotal == 0):
            derived_subtotal = s.grand_total - s.tax_amount + (s.discount or 0.0) - (s.round_off or 0.0)
            if derived_subtotal > 0:
                s.subtotal = round(derived_subtotal, 2)

        # 5. If grand_total exists and no taxes were extracted or mentioned, subtotal == grand_total
        if s.grand_total is not None and (s.subtotal is None or s.subtotal == 0) and not valid_taxes:
            s.subtotal = s.grand_total

        # 6. If line items amount is missing but grand_total exists, sync single line item
        if s.grand_total is not None and s.line_items and len(s.line_items) == 1:
            if s.line_items[0].amount == 0 or s.line_items[0].taxable_value == 0:
                target_val = s.subtotal if s.subtotal else s.grand_total
                s.line_items[0].rate = target_val
                s.line_items[0].amount = target_val
                s.line_items[0].taxable_value = target_val

        # 7. Extract state code / place of supply from GSTIN if missing
        if not s.place_of_supply and s.vendor_gstin and len(s.vendor_gstin) >= 2:
            st_code = s.vendor_gstin[:2]
            if st_code in GST_STATE_CODES:
                s.place_of_supply = f"{st_code} - {GST_STATE_CODES[st_code]}"

        # 8. Amount in words derivation
        if not s.amount_in_words and s.grand_total and s.grand_total > 0:
            s.amount_in_words = number_to_words_inr(s.grand_total)

        # 9. Account Name fallback to Beneficiary or Vendor name
        if not s.account_name:
            if s.buyer_name:
                s.account_name = s.buyer_name
            elif s.vendor_name:
                s.account_name = s.vendor_name

        return s

    def reconcile_accounting_hypotheses(
        self,
        candidate_map: dict[str, list[tuple[str, float]]],
        subtotal_cands: Optional[list[float]] = None,
        tax_cands: Optional[list[float]] = None,
        total_cands: Optional[list[float]] = None,
    ) -> Optional[dict[str, tuple[float, float]]]:
        """
        Accounting Hypothesis Validator:
        Searches candidate combinations across financial totals to find arithmetic equilibrium:
        subtotal + tax == grand_total.
        Returns the winning combination with boosted confidence if verified.
        """
        sub_list = subtotal_cands or []
        tax_list = tax_cands or [0.0]
        tot_list = total_cands or []

        for sub in sub_list:
            for tax in tax_list:
                for tot in tot_list:
                    if abs((sub + tax) - tot) <= 0.05 and tot > 0:
                        logger.info(
                            f"[Accounting Validator] Arithmetic equilibrium verified: "
                            f"subtotal ({sub}) + tax ({tax}) == grand_total ({tot})"
                        )
                        return {
                            "subtotal": (sub, 0.98),
                            "tax_amount": (tax, 0.98),
                            "grand_total": (tot, 0.98),
                        }
        return None


    # ------------------------------------------------------------------

    def _get_val(self, field: Optional[ExtractedField]) -> Optional[str]:
        if not field or field.value is None:
            return None
        return str(field.value).strip()

    def _get_float(self, field: Optional[ExtractedField]) -> Optional[float]:
        if not field or field.value is None:
            return None
        if isinstance(field.value, (int, float)):
            return float(field.value)
        val_str = str(field.value).strip()
        if not val_str:
            return None
        try:
            is_neg = val_str.startswith("-")
            cleaned = re.sub(r"[^\d.]", "", val_str)
            if not cleaned:
                return None
            val = float(cleaned)
            return -val if is_neg else val
        except (ValueError, TypeError):
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
            category=self._get_val(getattr(inv, "category", None)),
            subcategory=self._get_val(getattr(inv, "subcategory", None)),
            vendor_name=self._get_val(inv.vendor_name),
            vendor_address=self._get_val(inv.vendor_address),
            vendor_address_line1=self._get_val(getattr(inv, "vendor_address_line1", None)),
            vendor_address_line2=self._get_val(getattr(inv, "vendor_address_line2", None)),
            vendor_gstin=self._get_val(inv.vendor_gstin),
            vendor_pan=self._get_val(inv.vendor_pan),
            vendor_email=self._get_val(inv.vendor_email),
            vendor_phone=self._get_val(inv.vendor_phone),
            buyer_name=self._get_val(inv.buyer_name),
            buyer_address=self._get_val(inv.buyer_address),
            buyer_address_line1=self._get_val(getattr(inv, "buyer_address_line1", None)),
            buyer_address_line2=self._get_val(getattr(inv, "buyer_address_line2", None)),
            buyer_gstin=self._get_val(inv.buyer_gstin),
            buyer_phone=self._get_val(getattr(inv, "buyer_phone", None)),
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
            field_provenance={
                fname: {
                    "value": str(getattr(inv, fname).value),
                    "confidence": float(getattr(getattr(inv, fname), "confidence", 0.90)),
                    "source": str(getattr(getattr(inv, fname), "source", "ocr")),
                    "page": int(getattr(getattr(inv, fname), "page", 1)),
                    "bbox": getattr(getattr(inv, fname), "bbox", None),
                    "ocr_confidence": getattr(getattr(inv, fname), "ocr_confidence", None),
                }
                for fname in [
                    "invoice_number", "invoice_date", "due_date", "po_number", "place_of_supply",
                    "vendor_name", "vendor_gstin", "vendor_pan", "vendor_email", "vendor_phone",
                    "buyer_name", "buyer_gstin", "buyer_phone",
                    "subtotal", "tax_amount", "grand_total", "cgst", "sgst", "igst",
                    "bank_name", "branch_name", "account_name", "account_number", "ifsc_code"
                ]
                if getattr(inv, fname, None) and getattr(getattr(inv, fname), "value", None)
            },
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
            if ok:
                chk_ok = verify_gstin_checksum(clean)
                r.add("gstin_checksum", chk_ok,
                      "Vendor GSTIN checksum valid" if chk_ok else f"Vendor GSTIN checksum mismatch: {s.vendor_gstin}",
                      "warning" if not chk_ok else "info")

        if s.buyer_gstin:
            clean = s.buyer_gstin.replace(" ", "").upper()
            ok = bool(GSTIN_RE.match(clean))
            r.add("buyer_gstin_format", ok,
                  "Buyer GSTIN format valid" if ok else f"Invalid buyer GSTIN: {s.buyer_gstin}",
                  "warning")
            if ok:
                chk_ok = verify_gstin_checksum(clean)
                r.add("buyer_gstin_checksum", chk_ok,
                      "Buyer GSTIN checksum valid" if chk_ok else f"Buyer GSTIN checksum mismatch: {s.buyer_gstin}",
                      "warning" if not chk_ok else "info")

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
        Holistic accounting consistency resolver & GST engine.
        Evaluates multiple mathematical hypotheses (line-item tax vs global tax,
        tax inclusive vs exclusive) to prevent false rejections.
        """
        tol = self.TOLERANCE

        # 1. Line items sum = subtotal (or grand_total if tax inclusive)
        if s.line_items and s.subtotal:
            items_sum = sum(item.amount for item in s.line_items if item.amount is not None)
            diff = abs(items_sum - s.subtotal)
            ok = diff <= max(s.subtotal * tol, 1.5)
            if not ok and s.grand_total and abs(items_sum - s.grand_total) <= max(s.grand_total * tol, 1.5):
                # Tax inclusive line item scheme
                r.add("line_items_sum", True, f"Line items match gross grand total (Tax Inclusive Pricing): {items_sum:.2f}", "info")
            else:
                r.add("line_items_sum",
                      ok,
                      f"Line items sum {items_sum:.2f} matches subtotal {s.subtotal:.2f}" if ok
                      else f"Line items sum {items_sum:.2f} ≠ subtotal {s.subtotal:.2f} (diff={diff:.2f})",
                      "error" if not ok else "info")

        # 2. GST Intra-State vs Inter-State Consistency Check (Line-Item & Global)
        global_cgst = s.global_cgst_rate or 0.0
        global_sgst = s.global_sgst_rate or 0.0
        global_igst = s.global_igst_rate or 0.0
        total_global_rate = global_cgst + global_sgst + global_igst

        # Intra-state check
        if (s.cgst is not None and s.cgst > 0) or (s.sgst is not None and s.sgst > 0) or (global_cgst > 0 or global_sgst > 0):
            if global_cgst > 0 and global_sgst > 0:
                rate_match = abs(global_cgst - global_sgst) < 0.01
                r.add("intra_state_rates", rate_match,
                      f"Intra-state global GST rates equal: CGST={global_cgst}%, SGST={global_sgst}%" if rate_match
                      else f"Intra-state GST rate mismatch: CGST={global_cgst}% ≠ SGST={global_sgst}%",
                      "warning" if not rate_match else "info")
            if s.cgst is not None and s.sgst is not None and (s.cgst > 0 or s.sgst > 0):
                amt_match = abs(s.cgst - s.sgst) <= max(s.cgst * tol, 1.5)
                r.add("intra_state_amounts", amt_match,
                      f"Intra-state amounts match: CGST={s.cgst:.2f}, SGST={s.sgst:.2f}" if amt_match
                      else f"Intra-state amount mismatch: CGST={s.cgst:.2f} ≠ SGST={s.sgst:.2f}",
                      "warning" if not amt_match else "info")
            if (s.igst and s.igst > 0) or global_igst > 0:
                r.add("gst_type_conflict", False, "Both Intra-state (CGST/SGST) and Inter-state (IGST) detected simultaneously", "warning")

        # Inter-state check
        elif (s.igst is not None and s.igst > 0) or global_igst > 0:
            r.add("inter_state_gst", True, f"Inter-state IGST active (Rate={global_igst}%, Amount={s.igst or 0.0})", "info")

        # 3. CGST + SGST = tax_amount (or Global GST calculation)
        tax_total = (s.cgst or 0.0) + (s.sgst or 0.0) + (s.igst or 0.0)
        if tax_total == 0.0 and s.subtotal and total_global_rate > 0:
            computed_tax = round(s.subtotal * (total_global_rate / 100.0), 2)
            if s.tax_amount and abs(computed_tax - s.tax_amount) <= max(s.tax_amount * tol, 1.5):
                r.add("global_gst_math", True, f"Global GST math verified ({s.subtotal} * {total_global_rate}% ≈ {s.tax_amount})", "info")
                tax_total = s.tax_amount
            else:
                tax_total = computed_tax

        if s.tax_amount and tax_total > 0:
            diff = abs(tax_total - s.tax_amount)
            ok = diff <= max(s.tax_amount * tol, 1.5)
            r.add("cgst_sgst_sum", ok,
                  f"Tax components ({tax_total:.2f}) match total tax ({s.tax_amount:.2f})" if ok
                  else f"Tax components ({tax_total:.2f}) ≠ total tax ({s.tax_amount:.2f})",
                  "error" if not ok else "info")

        # 4. subtotal + tax - discount + round_off = grand_total
        if s.subtotal and s.grand_total and (s.tax_amount is not None or tax_total > 0):
            effective_tax = s.tax_amount if s.tax_amount is not None else tax_total
            discount = (s.discount or 0.0) + (s.global_discount or 0.0)
            round_off = s.round_off or 0.0
            computed_total = s.subtotal + effective_tax - discount + round_off
            diff = abs(computed_total - s.grand_total)
            ok = diff <= max(s.grand_total * tol, 1.5)
            r.add("grand_total_math", ok,
                  f"Grand total math checks out ({computed_total:.2f} ≈ {s.grand_total:.2f})" if ok
                  else f"Grand total mismatch: {s.subtotal}+{effective_tax}-{discount}+{round_off}={computed_total:.2f} ≠ {s.grand_total:.2f}",
                  "error" if not ok else "info")

        # 5. Amount in words semantic consistency
        if s.amount_in_words and s.grand_total and s.grand_total > 0:
            expected_words = number_to_words_inr(s.grand_total)
            clean_extracted = re.sub(r"[^a-z0-9]", "", s.amount_in_words.lower())
            clean_computed = re.sub(r"[^a-z0-9]", "", expected_words.lower())
            # Check ratio
            import difflib
            sim = difflib.SequenceMatcher(None, clean_extracted, clean_computed).ratio()
            word_ok = sim >= 0.70 or clean_computed in clean_extracted or clean_extracted in clean_computed
            r.add("amount_in_words_match", word_ok,
                  f"Amount in words aligns with grand total ({s.amount_in_words})" if word_ok
                  else f"Amount in words mismatch: '{s.amount_in_words}' vs expected '{expected_words}'",
                  "warning" if not word_ok else "info")

    def _check_dates(self, s: InvoiceSchema, r: ValidationReport):
        date_formats = [
            "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d %b %Y", "%d %B %Y",
            "%d-%b-%Y", "%d-%b-%y", "%d/%m/%y", "%d-%m-%y", "%d %b %y", "%d %B %y",
            "%m/%d/%Y", "%m-%d-%Y", "%m/%d/%y", "%m-%d-%y"
        ]

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
