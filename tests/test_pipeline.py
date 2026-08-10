"""
tests/test_pipeline.py

Unit and integration tests for the invoice digitization pipeline.
Run with: pytest tests/ -v
"""

import pytest
import json
import io
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_white_image(w=794, h=1123) -> np.ndarray:
    """Return a blank white BGR image."""
    import cv2
    return np.ones((h, w, 3), dtype=np.uint8) * 255


def make_image_bytes(w=200, h=300) -> bytes:
    img = Image.new("RGB", (w, h), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_simple_pdf_bytes() -> bytes:
    """Minimal valid PDF bytes."""
    content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 595 842]/Parent 2 0 R/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 44>>stream
BT /F1 12 Tf 100 700 Td (INVOICE) Tj ET
endstream
endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000360 00000 n
trailer<</Size 6/Root 1 0 R>>
startxref
441
%%EOF"""
    return content


# ------------------------------------------------------------------
# Preprocessing tests
# ------------------------------------------------------------------

class TestPreprocessor:
    def test_process_image_bytes(self):
        from preprocessing.pipeline import InvoicePreprocessor
        p = InvoicePreprocessor()
        img_bytes = make_image_bytes()
        result = p.process(img_bytes)
        assert result.image is not None
        assert result.pil_image is not None
        assert result.original_size[0] > 0
        assert result.processed_size[0] > 0

    def test_process_numpy_array(self):
        from preprocessing.pipeline import InvoicePreprocessor
        p = InvoicePreprocessor()
        img = make_white_image()
        result = p.process(img)
        assert result.image.shape[2] == 3

    def test_deskew_zero_angle(self):
        from preprocessing.pipeline import InvoicePreprocessor
        p = InvoicePreprocessor()
        img = make_white_image()
        rotated, angle = p._deskew(img)
        assert abs(angle) < 45   # must not produce wild corrections

    def test_binarize_high_variance_not_binarized(self):
        """High variance (colour) images should NOT be binarized."""
        from preprocessing.pipeline import InvoicePreprocessor
        import cv2
        p = InvoicePreprocessor()
        # Colourful image = high variance
        img = np.random.randint(0, 255, (300, 200, 3), dtype=np.uint8)
        _, was_binarized = p._adaptive_binarize(img)
        assert not was_binarized

    def test_unsupported_file_raises(self):
        from preprocessing.pipeline import InvoicePreprocessor
        p = InvoicePreprocessor()
        with pytest.raises(FileNotFoundError):
            p.process("/nonexistent/path/invoice.png")


# ------------------------------------------------------------------
# Detection tests
# ------------------------------------------------------------------

class TestDetector:
    def test_heuristic_fallback_produces_regions(self):
        from detection.detector import InvoiceDetector
        d = InvoiceDetector(model_path=None)   # no model → heuristic
        img = make_white_image()
        result = d.detect(img)
        assert result.model_used == "heuristic"
        assert len(result.regions) > 0

    def test_region_labels_are_valid(self):
        from detection.detector import InvoiceDetector, REGION_LABELS
        d = InvoiceDetector(model_path=None)
        img = make_white_image()
        result = d.detect(img)
        valid_labels = set(REGION_LABELS.values())
        for region in result.regions:
            assert region.label in valid_labels

    def test_regions_have_nonzero_crops(self):
        from detection.detector import InvoiceDetector
        d = InvoiceDetector(model_path=None)
        img = make_white_image()
        result = d.detect(img)
        for region in result.regions:
            assert region.crop.size > 0

    def test_visualise_returns_image(self):
        from detection.detector import InvoiceDetector
        d = InvoiceDetector(model_path=None)
        img = make_white_image()
        result = d.detect(img)
        vis = d.visualise(img, result)
        assert vis.shape == img.shape


# ------------------------------------------------------------------
# Understanding / heuristic extraction tests
# ------------------------------------------------------------------

class TestLayoutLMExtractor:
    def _make_mock_ocr_results(self, texts: dict):
        from ocr.extractor import OCRResult, TextBlock
        results = {}
        for label, text in texts.items():
            results[label] = OCRResult(
                region_label=label,
                text_blocks=[TextBlock(text=text, confidence=0.9, bbox=[], region_label=label)],
                full_text=text,
                avg_confidence=0.9,
            )
        return results

    def test_extracts_gstin_from_text(self):
        from understanding.layoutlm import LayoutLMExtractor
        ex = LayoutLMExtractor(model_path=None)
        ocr = self._make_mock_ocr_results({
            "vendor_block": "Acme Ltd\nGSTIN: 27AADCA1234A1Z5\nPune",
        })
        result = ex.extract(ocr)
        assert result.vendor_gstin is not None
        assert "27AADCA1234A1Z5" in result.vendor_gstin.value

    def test_extracts_grand_total(self):
        from understanding.layoutlm import LayoutLMExtractor
        ex = LayoutLMExtractor(model_path=None)
        ocr = self._make_mock_ocr_results({
            "totals_block": "Subtotal: 10000.00\nGST: 1800.00\nGrand Total: Rs. 11800.00",
        })
        result = ex.extract(ocr)
        assert result.grand_total is not None
        assert "11800" in result.grand_total.value

    def test_extracts_email(self):
        from understanding.layoutlm import LayoutLMExtractor
        ex = LayoutLMExtractor(model_path=None)
        ocr = self._make_mock_ocr_results({
            "vendor_block": "Tech Corp\nbilling@techcorp.com\n+91 9876543210",
        })
        result = ex.extract(ocr)
        assert result.vendor_email is not None
        assert "billing@techcorp.com" in result.vendor_email.value

    def test_extracts_invoice_number(self):
        from understanding.layoutlm import LayoutLMExtractor
        ex = LayoutLMExtractor(model_path=None)
        ocr = self._make_mock_ocr_results({
            "header": "Invoice No: INV-2024-099\nDate: 01/06/2024",
        })
        result = ex.extract(ocr)
        assert result.invoice_number is not None
        assert "INV-2024-099" in result.invoice_number.value

    def test_empty_input_returns_empty_invoice(self):
        from understanding.layoutlm import LayoutLMExtractor
        ex = LayoutLMExtractor(model_path=None)
        result = ex.extract({})
        assert result.vendor_name is None
        assert result.grand_total is None


# ------------------------------------------------------------------
# Validation tests
# ------------------------------------------------------------------

class TestValidator:
    def _make_extracted(self, **kwargs):
        from understanding.layoutlm import ExtractedInvoice, ExtractedField
        inv = ExtractedInvoice()
        for k, v in kwargs.items():
            setattr(inv, k, ExtractedField(value=str(v), confidence=0.9, source="test"))
        return inv

    def test_valid_invoice_passes(self):
        from validation.validator import InvoiceValidator
        v = InvoiceValidator()
        inv = self._make_extracted(
            invoice_number="INV-001",
            vendor_name="Acme Ltd",
            vendor_gstin="27AADCA1234A1Z5",
            subtotal="10000",
            tax_amount="1800",
            grand_total="11800",
        )
        schema, report = v.validate(inv)
        assert schema.invoice_number == "INV-001"
        assert schema.vendor_name == "Acme Ltd"
        assert schema.grand_total == 11800.0

    def test_missing_required_fields_flagged(self):
        from validation.validator import InvoiceValidator
        v = InvoiceValidator()
        from understanding.layoutlm import ExtractedInvoice
        schema, report = v.validate(ExtractedInvoice())
        assert not report.is_valid
        assert len(report.errors) > 0

    def test_invalid_gstin_flagged(self):
        from validation.validator import InvoiceValidator
        v = InvoiceValidator()
        inv = self._make_extracted(
            invoice_number="INV-001",
            vendor_name="Acme",
            vendor_gstin="INVALID_GSTIN",
            grand_total="5000",
        )
        _, report = v.validate(inv)
        errors_and_warns = report.errors + report.warnings
        assert any("GSTIN" in msg or "gstin" in msg.lower() for msg in errors_and_warns)

    def test_math_mismatch_flagged(self):
        from validation.validator import InvoiceValidator
        v = InvoiceValidator()
        inv = self._make_extracted(
            invoice_number="INV-001",
            vendor_name="Acme",
            grand_total="5000",
            subtotal="3000",     # 3000 + 900 = 3900, not 5000 → mismatch
            tax_amount="900",
        )
        _, report = v.validate(inv)
        assert any("total" in msg.lower() or "mismatch" in msg.lower() for msg in report.errors)

    def test_negative_amount_flagged(self):
        from validation.validator import InvoiceValidator
        v = InvoiceValidator()
        inv = self._make_extracted(
            invoice_number="INV-001",
            vendor_name="Acme",
            grand_total="-500",
        )
        _, report = v.validate(inv)
        assert not report.is_valid

    def test_valid_ifsc_passes(self):
        from validation.validator import InvoiceValidator
        v = InvoiceValidator()
        inv = self._make_extracted(
            invoice_number="INV-001",
            vendor_name="Acme",
            grand_total="1000",
            ifsc_code="HDFC0001234",
        )
        _, report = v.validate(inv)
        ifsc_results = [r for r in report.results if "ifsc" in r.rule]
        assert all(r.passed for r in ifsc_results)


# ------------------------------------------------------------------
# Renderer tests
# ------------------------------------------------------------------

class TestRenderer:
    def _make_schema(self, **kwargs):
        from validation.validator import InvoiceSchema, LineItem
        defaults = dict(
            invoice_number="INV-TEST-001",
            invoice_date="01/06/2024",
            vendor_name="Test Corp",
            buyer_name="Client Ltd",
            grand_total=11800.0,
            subtotal=10000.0,
            tax_amount=1800.0,
            currency="INR",
            line_items=[LineItem(description="Service", quantity=1, rate=10000, amount=10000)],
            overall_confidence=0.88,
            needs_review=False,
            review_reasons=[],
        )
        defaults.update(kwargs)
        return InvoiceSchema(**defaults)

    def test_to_html_contains_invoice_number(self):
        from output.renderer import InvoiceRenderer
        r = InvoiceRenderer()
        schema = self._make_schema()
        html = r.to_html(schema)
        assert "INV-TEST-001" in html
        assert "Test Corp" in html
        assert "11800" in html

    def test_to_html_shows_review_warning(self):
        from output.renderer import InvoiceRenderer
        r = InvoiceRenderer()
        schema = self._make_schema(needs_review=True, review_reasons=["Grand total mismatch"])
        html = r.to_html(schema)
        assert "Grand total mismatch" in html

    def test_to_json_is_valid(self):
        from output.renderer import InvoiceRenderer
        r = InvoiceRenderer()
        schema = self._make_schema()
        j = r.to_json(schema)
        parsed = json.loads(j)
        assert parsed["invoice_number"] == "INV-TEST-001"
        assert parsed["grand_total"] == 11800.0

    def test_to_dict_round_trips(self):
        from output.renderer import InvoiceRenderer
        from validation.validator import InvoiceSchema
        r = InvoiceRenderer()
        schema = self._make_schema()
        d = r.to_dict(schema)
        schema2 = InvoiceSchema(**d)
        assert schema2.invoice_number == schema.invoice_number


# ------------------------------------------------------------------
# Ollama client tests (mocked)
# ------------------------------------------------------------------

class TestOllamaClient:
    def test_unavailable_when_not_running(self):
        from llm_fallback.ollama_client import OllamaClient
        client = OllamaClient(base_url="http://localhost:19999")  # wrong port
        assert not client.is_available()

    def test_extract_field_returns_empty_on_connection_error(self):
        from llm_fallback.ollama_client import OllamaClient
        client = OllamaClient(base_url="http://localhost:19999")
        result = client.extract_field("invoice_number", "some text")
        assert result.value == ""
        assert result.confidence == 0.0

    @patch("llm_fallback.ollama_client.httpx.post")
    def test_extract_field_parses_response(self, mock_post):
        from llm_fallback.ollama_client import OllamaClient
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "INV-2024-001"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = OllamaClient()
        result = client.extract_field("invoice_number", "Invoice No: INV-2024-001")
        assert result.value == "INV-2024-001"
        assert result.confidence > 0
        assert result.source == "llm"
