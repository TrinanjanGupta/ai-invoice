"""
ocr/candidate_fusion.py

Candidate Fusion Engine for Multi-Hypothesis OCR & Handwriting Recognition.
Weighs, reconciles, and selects between OCR candidates (PaddleOCR, TrOCR, NumericRecognizer,
and Modulo 36 / Character Confusion specialists) based on field semantics, syntax, and arithmetic consistency.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional, Any, Sequence
from loguru import logger

from ocr.candidate_generator import OCRCandidate
from validation.char_confusion import (
    correct_numeric_field,
    correct_gstin,
    correct_ifsc,
    correct_date,
    correct_phone,
    verify_gstin_checksum,
)


@dataclass
class FusedCandidateResult:
    selected_text: str
    confidence: float
    source: str
    field_name: str
    bbox: list[int]
    reasoning: str
    all_candidates: list[OCRCandidate] = field(default_factory=list)


class CandidateFusionEngine:
    """
    Centralized candidate selection engine that resolves ambiguities across OCR streams.
    """

    SOURCE_PRIORITIES = {
        "numeric_specializer": 0.92,
        "char_correction": 0.88,
        "trocr": 0.85,
        "paddleocr": 0.75,
        "ocr": 0.70,
    }

    @classmethod
    def fuse_candidates(
        cls,
        candidates: list[OCRCandidate],
        field_name: str,
        expected_type: str = "text",   # "numeric" | "gstin" | "ifsc" | "date" | "phone" | "text"
        context_text: str = "",
    ) -> FusedCandidateResult:
        """
        Selects the most statistically and semantically plausible candidate.
        """
        if not candidates:
            return FusedCandidateResult(
                selected_text="",
                confidence=0.0,
                source="none",
                field_name=field_name,
                bbox=[0, 0, 0, 0],
                reasoning="No candidate hypotheses provided",
                all_candidates=[],
            )

        scored: list[tuple[float, OCRCandidate, str]] = []

        for cand in candidates:
            text = str(cand.text).strip()
            if not text:
                continue

            base_weight = cls.SOURCE_PRIORITIES.get(cand.source, 0.70)
            score = cand.confidence * base_weight
            reasons = [f"base({cand.source}={cand.confidence:.2f})"]

            # 1. Type-specific syntax & checksum validation
            if expected_type == "numeric" or field_name in ("grand_total", "subtotal", "tax_amount", "cgst", "sgst", "igst", "discount", "round_off"):
                num_cands = correct_numeric_field(text)
                if num_cands:
                    text = num_cands[0][0]
                    score += 0.10
                    reasons.append(f"num_confusion_corrected({text})")

                # Validate float parseability
                cleaned_num = re.sub(r"[^\d.]", "", text)
                if cleaned_num and cleaned_num.count(".") <= 1:
                    score += 0.15
                    reasons.append("valid_float_syntax")
                else:
                    score -= 0.25
                    reasons.append("invalid_numeric_syntax")

            elif expected_type == "gstin" or field_name in ("vendor_gstin", "buyer_gstin"):
                gstin_cands = correct_gstin(text)
                if gstin_cands:
                    text = gstin_cands[0][0]
                if verify_gstin_checksum(text):
                    score += 0.30
                    reasons.append("modulo36_checksum_valid")
                elif re.match(r"^\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d]$", text):
                    score += 0.15
                    reasons.append("gstin_regex_valid")
                else:
                    score -= 0.20
                    reasons.append("invalid_gstin_structure")

            elif expected_type == "ifsc" or field_name == "ifsc_code":
                ifsc_cands = correct_ifsc(text)
                if ifsc_cands:
                    text = ifsc_cands[0][0]
                if re.match(r"^[A-Z]{4}0[A-Z0-9]{6}$", text):
                    score += 0.25
                    reasons.append("ifsc_regex_valid")

            elif expected_type == "date" or field_name in ("invoice_date", "due_date"):
                date_cands = correct_date(text)
                if date_cands:
                    text = date_cands[0][0]
                if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text):
                    score += 0.20
                    reasons.append("date_format_valid")

            elif expected_type == "phone" or field_name in ("vendor_phone", "buyer_phone"):
                phone_cands = correct_phone(text)
                if phone_cands:
                    text = phone_cands[0][0]
                digits = re.sub(r"\D", "", text)
                if len(digits) == 10:
                    score += 0.20
                    reasons.append("10_digit_phone_valid")

            elif field_name == "invoice_number":
                # Clean alphanumeric invoice number (e.g. resolve letter O in digit sequences)
                if re.search(r"\d[O|o]\d", text):
                    text = re.sub(r"(\d)[O|o](\d)", r"\g<1>0\g<2>", text)
                    score += 0.10
                    reasons.append("inv_num_O_to_0_resolved")

            # Final normalized composite confidence
            final_conf = max(0.10, min(0.99, score))
            cand.text = text
            scored.append((final_conf, cand, ", ".join(reasons)))

        if not scored:
            return FusedCandidateResult(
                selected_text=candidates[0].text,
                confidence=candidates[0].confidence,
                source=candidates[0].source,
                field_name=field_name,
                bbox=candidates[0].bbox,
                reasoning="fallback_first_candidate",
                all_candidates=candidates,
            )

        # Sort descending by final composite score
        scored.sort(key=lambda x: x[0], reverse=True)
        best_conf, best_cand, reasoning = scored[0]

        return FusedCandidateResult(
            selected_text=best_cand.text,
            confidence=round(best_conf, 2),
            source=best_cand.source,
            field_name=field_name,
            bbox=best_cand.bbox,
            reasoning=reasoning,
            all_candidates=candidates,
        )
