"""
Stage 4b: Ollama LLM fallback (Llama 3.1 8B / invoice-expert).

Called for fields that scored below the confidence threshold or are missing.
Uses local Ollama inference with candidate evidence and spatial document context.
"""

import re
import json
import httpx
from loguru import logger
from typing import Optional
from understanding.layoutlm import ExtractedField



FIELD_PROMPTS = {
    "invoice_number": (
        "Extract the invoice number from the text below. "
        "Return ONLY the invoice number (e.g. INV-2024-001). "
        "If not found, return UNKNOWN."
    ),
    "invoice_date": (
        "Extract the invoice date from the text below. "
        "Return ONLY the date in DD/MM/YYYY format. "
        "If not found, return UNKNOWN."
    ),
    "due_date": (
        "Extract the payment due date from the text below. "
        "Return ONLY the date in DD/MM/YYYY format. "
        "If not found, return UNKNOWN."
    ),
    "vendor_name": (
        "Extract the seller/vendor company name from the text below. "
        "Return ONLY the company name, nothing else. "
        "If not found, return UNKNOWN."
    ),
    "vendor_gstin": (
        "Extract the seller's GSTIN (GST Identification Number) from the text. "
        "Format: 2 digits + 5 letters + 4 digits + 1 letter + 1 letter + Z + 1 alphanumeric. "
        "Return ONLY the GSTIN. If not found, return UNKNOWN."
    ),
    "buyer_name": (
        "Extract the buyer/bill-to company or person name from the text. "
        "Return ONLY the name. If not found, return UNKNOWN."
    ),
    "grand_total": (
        "Extract the final grand total / total amount due from the text. "
        "Return ONLY the numeric value without currency symbols (e.g. 12500.00). "
        "If not found, return UNKNOWN."
    ),
    "subtotal": (
        "Extract the pre-tax subtotal / taxable amount from the text. "
        "Return ONLY the numeric value (e.g. 10593.22). "
        "If not found, return UNKNOWN."
    ),
    "tax_amount": (
        "Extract the total tax / GST / VAT amount from the text. "
        "Return ONLY the numeric value. If not found, return UNKNOWN."
    ),
    "cgst": (
        "Extract the CGST (Central GST) amount from the text. "
        "Return ONLY the numeric value. If not found, return UNKNOWN."
    ),
    "sgst": (
        "Extract the SGST (State GST) amount from the text. "
        "Return ONLY the numeric value. If not found, return UNKNOWN."
    ),
    "igst": (
        "Extract the IGST (Integrated GST) amount from the text. "
        "Return ONLY the numeric value. If not found, return UNKNOWN."
    ),
    "line_items": (
        "Extract all line items from the invoice text. "
        "Return a JSON array of objects, each with: "
        "description (string), quantity (number), rate (number), amount (number). "
        "Return ONLY the JSON array, no other text."
    ),
    "payment_terms": (
        "Extract the payment terms (e.g. Net 30, Due on receipt) from the text. "
        "Return ONLY the payment terms. If not found, return UNKNOWN."
    ),
    "bank_name": (
        "Extract the bank name from the payment details in the text. "
        "Return ONLY the bank name. If not found, return UNKNOWN."
    ),
    "place_of_supply": (
        "Extract the Place of Supply (state name or 2-digit GST state code + state) from the header text. "
        "Return ONLY the place of supply (e.g. 19-West Bengal or Maharashtra). "
        "If not found, return UNKNOWN."
    ),
    "branch_name": (
        "Extract the bank branch name from the payment/bank details in the text. "
        "Return ONLY the branch name. If not found, return UNKNOWN."
    ),
    "account_name": (
        "Extract the beneficiary / account holder name from the bank details in the text. "
        "Return ONLY the account name. If not found, return UNKNOWN."
    ),
    "amount_in_words": (
        "Extract the total amount written in words from the invoice (e.g. INR Ten Thousand Only). "
        "Return ONLY the words. If not found, return UNKNOWN."
    ),
    "round_off": (
        "Extract the round off / rounding adjustment amount from the totals. "
        "Return ONLY the numeric value (e.g. 0.45 or -0.20). If not found, return UNKNOWN."
    ),
}


class OllamaClient:
    """
    Ollama local LLM client for low-confidence field extraction.
    Connects to Ollama running at localhost:11434 by default.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "invoice-expert",
        timeout: float = 3000.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def is_available(self) -> bool:
        """Check if Ollama is running and the model is pulled."""
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                return False
            models = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
            return self.model.split(":")[0] in models
        except Exception:
            return False

    def extract_field(
        self,
        field_name: str,
        raw_text: str,
        context_texts: Optional[dict] = None,
    ) -> ExtractedField:
        """
        Ask the LLM to extract a single field from the OCR text.
        Returns an ExtractedField with source="llm".
        """
        system_prompt = FIELD_PROMPTS.get(
            field_name,
            f"Extract the {field_name} from the text. Return ONLY the value."
        )

        context_dict = context_texts if context_texts else {"full_page": raw_text}
        context = self._build_targeted_context([field_name], context_dict)
        user_message = f"{system_prompt}\n\n---\nTARGETED INVOICE CONTEXT:\n{context}\n---"

        try:
            resp = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": user_message,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,   # deterministic — we want facts not creativity
                        "num_predict": 200,
                    },
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            value = resp.json().get("response", "").strip()

            if not value or value.upper() == "UNKNOWN":
                return ExtractedField(value="", confidence=0.0, source="llm")

            # For line_items, try to parse JSON
            if field_name == "line_items":
                try:
                    parsed = json.loads(value)
                    return ExtractedField(
                        value=json.dumps(parsed),
                        confidence=0.82,
                        source="llm",
                    )
                except json.JSONDecodeError:
                    pass

            logger.debug(f"LLM extracted {field_name}: {value[:60]}")
            return ExtractedField(value=value, confidence=0.82, source="llm")

        except httpx.ConnectError:
            logger.warning("Ollama not reachable — LLM fallback skipped")
            return ExtractedField(value="", confidence=0.0, source="llm_unavailable")
        except Exception as e:
            logger.error(f"Ollama error for field {field_name}: {e}")
            return ExtractedField(value="", confidence=0.0, source="llm_error")

    def _get_dynamic_few_shot_context(self) -> str:
        """Fetches 1-2 verified invoices from the database as few-shot in-context learning demonstrations."""
        try:
            import asyncio
            import concurrent.futures
            from storage.db import DatabaseManager, InvoiceRecord
            from config.settings import get_settings
            from sqlalchemy import select

            async def _fetch():
                settings = get_settings()
                db = DatabaseManager(settings.database_url)
                async with db.session_factory() as session:
                    res = await session.execute(
                        select(InvoiceRecord)
                        .filter(
                            InvoiceRecord.status.in_(["reviewed", "partially_reviewed", "done"]),
                            InvoiceRecord.needs_review == False,
                            InvoiceRecord.output_json.isnot(None)
                        )
                        .limit(2)
                    )
                    return res.scalars().all()

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        records = pool.submit(asyncio.run, _fetch()).result()
                else:
                    records = loop.run_until_complete(_fetch())
            except Exception:
                records = asyncio.run(_fetch())

            if not records:
                return ""

            examples = []
            for i, rec in enumerate(records):
                clean_out = {
                    k: v for k, v in rec.output_json.items()
                    if k in ["invoice_number", "invoice_date", "vendor_name", "vendor_gstin", "buyer_name", "buyer_gstin", "subtotal", "tax_amount", "grand_total"]
                    and v is not None
                }

                if clean_out:
                    examples.append(
                        f"### Example {i+1} (Verified Ground Truth from your database):\n"
                        f"Extracted JSON:\n{json.dumps(clean_out, indent=2)}"
                    )

            if examples:
                return "\n\n" + "\n\n".join(examples) + "\n\n"
        except Exception as e:
            logger.debug(f"Dynamic few-shot loading skipped: {e}")
        return ""

    def _build_targeted_context(self, field_names: list, ocr_texts: dict) -> str:
        """
        Builds targeted, multi-page context tailored to the requested fields,
        avoiding blind document truncation.
        """
        context_parts = []
        full_page_text = ocr_texts.get("full_page", "")

        # Always include specific regional crops if available
        for region, text in ocr_texts.items():
            if region != "full_page" and text.strip():
                context_parts.append(f"[{region.upper()}]\n{text.strip()}")

        # If regional crops are sparse or full_page is available, extract targeted slices
        has_meta = any(f in field_names for f in ["invoice_number", "invoice_date", "due_date", "place_of_supply"])
        has_parties = any(f in field_names for f in ["vendor_name", "vendor_gstin", "buyer_name", "buyer_gstin"])
        has_totals = any(f in field_names for f in ["grand_total", "subtotal", "tax_amount", "cgst", "sgst", "igst", "round_off", "amount_in_words"])
        has_bank = any(f in field_names for f in ["bank_name", "branch_name", "account_name", "account_number", "ifsc_code", "payment_terms"])

        if full_page_text:
            lines = [l.strip() for l in full_page_text.split("\n") if l.strip()]
            if has_meta or has_parties:
                # Top slice (Header / Parties)
                top_slice = "\n".join(lines[:40])
                context_parts.append(f"[DOCUMENT_HEADER_AND_PARTIES]\n{top_slice}")
            if has_totals or has_bank:
                # Bottom slice (Totals / Bank / Payment)
                bot_slice = "\n".join(lines[-40:])
                context_parts.append(f"[DOCUMENT_TOTALS_AND_PAYMENT]\n{bot_slice}")

        return "\n\n".join(context_parts)

    def _extract_multiple_fields_batch(
        self,
        field_names: list,
        ocr_texts: dict,
        candidate_hints: Optional[list[str]] = None,
    ) -> dict:
        """
        Ask the LLM to extract or verify low-confidence fields using targeted context
        and rich spatial candidate evidence.
        """
        context = self._build_targeted_context(field_names, ocr_texts)
        fields_desc = "\n".join(f"- {name}" for name in field_names)

        candidates_sec = ""
        if candidate_hints:
            candidates_sec = "SPATIAL CANDIDATE EVIDENCE (Geometric coordinates & OCR confidence):\n" + "\n".join(candidate_hints) + "\n\n"

        few_shot_sec = self._get_dynamic_few_shot_context()

        prompt = (
            "You are an expert AI Invoice Digitization Engine specialized in Indian GST, multi-state commercial invoices, and bilingual Indic script extraction.\n"
            "Support bilingual documents with English headers and Indic scripts (Devanagari, Bengali, Tamil, Gujarati, Marathi, Kannada) or Romanized transliterations for company and customer names.\n"
            "From the targeted OCR context and spatial candidate evidence below, extract or verify the requested fields.\n"
            "Return your answer as a JSON object with field names as keys and extracted values as strings. If a field is not found, set its value to null.\n\n"
            f"Fields to extract:\n{fields_desc}\n\n"
            f"{candidates_sec}"
            "Strict Normalization Rules:\n"
            "- Dates MUST be formatted as DD/MM/YYYY (convert 2-digit years like '22-Dec-25' -> '22/12/2025')\n"
            "- Numbers MUST be plain numeric (no currency symbols)\n"
            "- GSTIN MUST be strictly 15 alphanumeric characters. Auto-correct obvious character confusions: 'O' <-> '0' in numeric positions, 'I'/'l' <-> '1', 'S' <-> '5'\n"
            "- Reconcile Math: Subtotal + CGST + SGST = Grand Total\n"
            f"{few_shot_sec}"
            f"---\nTARGETED INVOICE CONTEXT:\n{context}\n---\n\n"
            "Return ONLY the JSON object, no other text."
        )

        try:
            resp = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 500,
                    },
                },
                timeout=self.timeout,
            )

            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()

            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return {k: str(v) for k, v in parsed.items() if v is not None and str(v).strip()}
            return {}

        except json.JSONDecodeError:
            logger.warning(f"LLM batch response was not valid JSON: {raw[:200]}")
            return {}
        except httpx.ConnectError:
            logger.warning("Ollama not reachable — batch LLM fallback skipped")
            return {}
        except Exception as e:
            logger.error(f"Ollama batch extraction error: {e}")
            return {}

    def enhance_low_confidence_fields(
        self,
        invoice,   # ExtractedInvoice
        ocr_texts: dict,
        confidence_threshold: float = 0.65,
    ):
        """
        Scan all fields in an ExtractedInvoice.
        For any field below threshold, call Ollama with rich spatial candidate evidence.
        Calculates calibrated dynamic confidence grounded in arithmetic reconciliation and GSTIN checksums.
        """
        if not self.is_available():
            logger.warning("Ollama not available — skipping LLM enhancement")
            return

        fields_to_check = [
            "invoice_number", "invoice_date", "due_date", "place_of_supply",
            "vendor_name", "vendor_gstin",
            "buyer_name", "buyer_gstin",
            "grand_total", "subtotal", "tax_amount",
            "cgst", "sgst", "igst", "round_off",
            "bank_name", "branch_name", "account_name", "account_number", "ifsc_code",
            "amount_in_words",
        ]

        missing_fields = []
        candidate_hints = []
        spatial_cands = getattr(invoice, "spatial_candidates", [])

        for field_name in fields_to_check:
            current = getattr(invoice, field_name, None)
            if current is None or current.confidence < confidence_threshold:
                missing_fields.append(field_name)
                # Find matching spatial candidate evidence if available
                field_cands = [c for c in spatial_cands if c.field_name == field_name]
                if field_cands:
                    for fc in field_cands[:2]:
                        candidate_hints.append(
                            f"Field: {field_name} | Candidate: '{fc.value}' | Bbox: {fc.bbox} | "
                            f"Nearby Label: '{fc.nearby_label}' | Label Bbox: {fc.label_bbox} | "
                            f"Distance: {fc.distance_px:.0f}px | Conf: {fc.confidence:.2f} | Region: {fc.region}"
                        )
                elif current and current.value:
                    candidate_hints.append(
                        f"Field: {field_name} | Candidate: '{current.value}' | Source: {current.source} | Conf: {current.confidence:.2f}"
                    )

        if not missing_fields:
            logger.info("All fields above confidence threshold — no LLM fallback needed")
            return

        logger.info(
            f"LLM batch fallback for {len(missing_fields)} low-confidence fields: "
            f"{', '.join(missing_fields)}"
        )

        results = self._extract_multiple_fields_batch(missing_fields, ocr_texts, candidate_hints=candidate_hints)

        enhanced_count = 0
        gstin_pattern = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}$")
        from validation.validator import verify_gstin_checksum

        for field_name, value in results.items():
            if field_name in missing_fields and value:
                val_str = str(value).strip()
                if "gstin" in field_name and not gstin_pattern.match(val_str.upper()):
                    logger.debug(f"  Discarding LLM non-compliant GSTIN for {field_name}: {val_str}")
                    continue

                # Calibrate dynamic confidence
                base_conf = 0.55
                has_cand_match = any(cand.value.lower() in val_str.lower() for cand in spatial_cands)
                if has_cand_match:
                    base_conf += 0.15

                if "gstin" in field_name:
                    if verify_gstin_checksum(val_str):
                        base_conf += 0.15
                    else:
                        base_conf = min(base_conf, 0.45)
                elif "date" in field_name and re.match(r"^\d{2}/\d{2}/\d{4}$", val_str):
                    base_conf += 0.10
                elif "ifsc" in field_name and re.match(r"^[A-Z]{4}0[A-Z0-9]{6}$", val_str.upper()):
                    base_conf += 0.10

                calibrated_conf = round(min(0.85, base_conf), 2)
                setattr(invoice, field_name, ExtractedField(
                    value=val_str, confidence=calibrated_conf, source="llm"
                ))
                enhanced_count += 1
                logger.debug(f"  LLM extracted {field_name} (conf={calibrated_conf}): {val_str[:60]}")

        # Post-pass: Cross-field arithmetic consistency boost
        try:
            st = float(str(invoice.subtotal.value).replace(",", "")) if invoice.subtotal and invoice.subtotal.value else 0.0
            tx = float(str(invoice.tax_amount.value).replace(",", "")) if invoice.tax_amount and invoice.tax_amount.value else 0.0
            gt = float(str(invoice.grand_total.value).replace(",", "")) if invoice.grand_total and invoice.grand_total.value else 0.0
            if gt > 0 and st > 0 and abs((st + tx) - gt) <= max(1.0, gt * 0.02):
                if invoice.subtotal and invoice.subtotal.source == "llm":
                    invoice.subtotal.confidence = min(0.85, round(invoice.subtotal.confidence + 0.10, 2))
                if invoice.grand_total and invoice.grand_total.source == "llm":
                    invoice.grand_total.confidence = min(0.85, round(invoice.grand_total.confidence + 0.10, 2))
        except Exception:
            pass

        if enhanced_count:
            logger.info(f"LLM enhanced {enhanced_count}/{len(missing_fields)} low-confidence fields")
        else:
            logger.warning("LLM batch call returned no usable values")
