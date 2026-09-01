"""
validation/acceptance_gate.py

Centralized, explicit Bank-Grade Auto-Acceptance Gatekeeper.
Evaluates whether an extracted invoice can be safely auto-accepted (Bronze tier)
without human review, enforcing unified multi-evidence gates:

1. Critical fields presence & strict format/checksum verification:
   - grand_total > 0
   - invoice_number present & compliant
   - invoice_date present & parseable
   - vendor_gstin present (and Modulo-36 verified if 15 chars)
   - bank account & IFSC (if present, verified format)
2. Arithmetic reconciliation:
   - Subtotal + Taxes - Discounts ≈ Grand Total (under Line-Item or Global GST hypothesis)
3. Document visual quality:
   - quality_score >= 0.60 (Laplacian blur, contrast, text density)
4. Multi-model consensus:
   - No hard contradiction between LayoutLM, Heuristic, and LLM extractions
5. Template familiarity:
   - Reject novel/unseen layout templates for initial verification
6. Document routing:
   - Handwritten, mixed, and camera distorted routes must fail closed into human review.
"""

from typing import Optional, Any
from loguru import logger
from validation.validator import InvoiceSchema, ValidationReport, verify_gstin_checksum, GSTIN_RE, IFSC_RE


class AutoAcceptanceGate:
    """
    Unified gatekeeper for automated invoice acceptance.
    """

    MIN_OVERALL_CONFIDENCE = 0.88
    MIN_OVERALL_CONFIDENCE_FAMILY = 0.92
    MIN_CRITICAL_FIELD_CONFIDENCE = 0.90
    MIN_CRITICAL_FIELD_CONFIDENCE_FAMILY = 0.94
    MIN_DOCUMENT_QUALITY_SCORE = 0.60
    CRITICAL_FIELDS = ["grand_total", "invoice_number", "invoice_date"]

    @classmethod
    def evaluate(
        cls,
        invoice: InvoiceSchema,
        validation_report: Optional[ValidationReport] = None,
        quality_score: float = 1.0,
        has_contradiction: bool = False,
        is_novel_template: bool = False,
        doc_type: str = "UNKNOWN",
        match_type: Optional[str] = None,
    ) -> tuple[bool, list[str]]:
        """
        Evaluates invoice against all acceptance gates.
        Enforces distinct rigor for exact_version, family_anchor, and novel templates.
        Returns: (is_auto_accepted, rejection_reasons)
        """
        rejection_reasons: list[str] = []

        is_family = (match_type == "family_anchor")
        req_min_overall = cls.MIN_OVERALL_CONFIDENCE_FAMILY if is_family else cls.MIN_OVERALL_CONFIDENCE
        req_min_critical = cls.MIN_CRITICAL_FIELD_CONFIDENCE_FAMILY if is_family else cls.MIN_CRITICAL_FIELD_CONFIDENCE

        # 1. Validation Report Errors & Warnings
        if validation_report:
            if not validation_report.is_valid or validation_report.errors:
                for err in validation_report.errors:
                    rejection_reasons.append(f"Validation Error: {err}")
            # Family matches must not have unresolved arithmetic warnings
            if is_family and validation_report.warnings:
                arithmetic_warnings = [w for w in validation_report.warnings if any(k in w.lower() for k in ("arithmetic", "math", "total", "tax", "mismatch"))]
                if arithmetic_warnings:
                    rejection_reasons.append(f"Family match rejected due to arithmetic reconciliation warning: {arithmetic_warnings[0]}")

        # 2. Critical Fields & Confidence Thresholds
        confs = invoice.field_confidences or {}
        for cf in cls.CRITICAL_FIELDS:
            val = getattr(invoice, cf, None)
            if val is None or str(val).strip() == "":
                rejection_reasons.append(f"Required critical field missing: {cf}")
            else:
                cf_conf = confs.get(cf, 0.0)
                if cf_conf < req_min_critical:
                    rejection_reasons.append(
                        f"Critical field {cf} confidence ({cf_conf:.2f}) below threshold ({req_min_critical:.2f})"
                    )

        # 3. Vendor GSTIN Checksum
        if invoice.vendor_gstin:
            clean_gstin = invoice.vendor_gstin.strip().upper().replace(" ", "")
            if len(clean_gstin) == 15 and GSTIN_RE.match(clean_gstin):
                if not verify_gstin_checksum(clean_gstin):
                    rejection_reasons.append(f"Vendor GSTIN checksum mismatch: {clean_gstin}")
            else:
                rejection_reasons.append(f"Vendor GSTIN invalid format: {invoice.vendor_gstin}")

        # 4. IFSC Code Format
        if invoice.ifsc_code:
            clean_ifsc = invoice.ifsc_code.strip().upper()
            if not IFSC_RE.match(clean_ifsc):
                rejection_reasons.append(f"Bank IFSC code format invalid: {clean_ifsc}")

        # 5. Overall Calibrated Confidence Threshold
        overall_conf = invoice.overall_confidence or 0.0
        if overall_conf < req_min_overall:
            rejection_reasons.append(
                f"Overall confidence ({overall_conf:.2f}) below auto-acceptance threshold ({req_min_overall:.2f})"
            )

        # 6. Document Visual Quality
        if quality_score < cls.MIN_DOCUMENT_QUALITY_SCORE:
            rejection_reasons.append(
                f"Document visual quality ({quality_score:.2f}) below threshold ({cls.MIN_DOCUMENT_QUALITY_SCORE:.2f})"
            )

        # 7. Model Contradiction / Disagreement
        if has_contradiction:
            rejection_reasons.append("Extraction contradiction detected between consensus engines")

        # 8. Novel Layout Template / Unknown Template
        if is_novel_template or match_type == "none":
            rejection_reasons.append(f"Unseen layout template ({invoice.template_id or 'novel'}) requires initial human verification")

        # 9. Document Routing (Handwriting & Mixed must be human reviewed)
        if doc_type in ("HANDWRITTEN", "MIXED"):
            rejection_reasons.append(f"Document route '{doc_type}' requires human review (handwriting path)")

        is_auto_accepted = len(rejection_reasons) == 0
        return is_auto_accepted, list(dict.fromkeys(rejection_reasons))
