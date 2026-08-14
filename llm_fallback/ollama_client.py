"""
Stage 4b: Ollama LLM fallback.

Called ONLY for fields that scored below the confidence threshold.
Uses Mistral 7B (or any Ollama-hosted model) running locally — zero cost.

Prompt engineering is structured so the model returns only the requested
field value, nothing else, making parsing trivial and reliable.
"""

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
        model: str = "mistral",
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

        context = ""
        if context_texts:
            context = "\n".join(
                f"[{region}]\n{text}"
                for region, text in context_texts.items()
                if text.strip()
            )
        else:
            context = raw_text

        user_message = f"{system_prompt}\n\n---\nINVOICE TEXT:\n{context[:3000]}\n---"

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

    def _extract_multiple_fields_batch(
        self,
        field_names: list,
        ocr_texts: dict,
    ) -> dict:
        """
        Ask the LLM to extract ALL missing fields in a single prompt.
        Returns a dict of {field_name: value_string}.
        Much faster than one call per field on CPU-only Ollama.
        """
        context = "\n".join(
            f"[{region}]\n{text}"
            for region, text in ocr_texts.items()
            if text.strip()
        )

        fields_desc = "\n".join(
            f"- {name}" for name in field_names
        )

        prompt = (
            "You are an invoice data extraction assistant.\n"
            "From the OCR text below, extract the following fields.\n"
            "Return your answer as a JSON object with field names as keys "
            "and extracted values as strings. If a field is not found, set its value to null.\n\n"
            f"Fields to extract:\n{fields_desc}\n\n"
            "Formatting rules:\n"
            "- Dates should be DD/MM/YYYY\n"
            "- Numbers should be plain numeric (no currency symbols)\n"
            "- GSTIN format: 2 digits + 5 letters + 4 digits + 1 letter + 1 letter + Z + 1 alphanumeric\n\n"
            f"---\nINVOICE TEXT:\n{context[:4000]}\n---\n\n"
            "Return ONLY the JSON object, no other text."
        )

        try:
            resp = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 500,
                    },
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()

            # Try to parse JSON — the model may wrap it in markdown fences
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
        For any field below the threshold, call Ollama to improve it.
        Uses a SINGLE batched prompt for all missing fields (much faster on CPU).
        Modifies the invoice in-place.
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

        # Collect fields that need LLM help
        missing_fields = []
        for field_name in fields_to_check:
            current = getattr(invoice, field_name, None)
            if current is None or current.confidence < confidence_threshold:
                missing_fields.append(field_name)

        if not missing_fields:
            logger.info("All fields above confidence threshold — no LLM fallback needed")
            return

        logger.info(
            f"LLM batch fallback for {len(missing_fields)} low-confidence fields: "
            f"{', '.join(missing_fields)}"
        )

        # Single batched LLM call for all missing fields
        results = self._extract_multiple_fields_batch(missing_fields, ocr_texts)

        enhanced_count = 0
        for field_name, value in results.items():
            if field_name in missing_fields and value:
                setattr(invoice, field_name, ExtractedField(
                    value=value, confidence=0.82, source="llm"
                ))
                enhanced_count += 1
                logger.debug(f"  LLM extracted {field_name}: {value[:60]}")

        if enhanced_count:
            logger.info(f"LLM enhanced {enhanced_count}/{len(missing_fields)} low-confidence fields")
        else:
            logger.warning("LLM batch call returned no usable values")
