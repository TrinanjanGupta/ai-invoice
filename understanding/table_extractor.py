"""
PyMuPDF-based structured table and line item extractor.

Extracts line items, quantities, rates, amounts, and tax values directly
from native table grids in digital PDFs.
"""

import re
import pymupdf
from loguru import logger
from typing import Optional


class TableExtractor:
    """
    Extracts tabular line items from PDF pages using PyMuPDF table detection.
    """

    COLUMN_KEYWORDS = {
        "description": ["description", "particular", "item", "product", "service", "goods", "details"],
        "hsn_code":    ["hsn", "sac", "hsn/sac", "hsn_code"],
        "quantity":    ["qty", "quantity", "nos", "qnty"],
        "unit":        ["unit", "uom"],
        "rate":        ["rate", "price", "unit price", "mrp", "unit_rate"],
        "discount":    ["disc", "discount", "disc."],
        "amount":      ["taxable value", "taxable amount", "amount", "total", "value", "net amount"],
        "tax_rate":    ["gst %", "tax %", "rate %", "tax rate", "gst rate"],
        "cgst_amount": ["cgst", "cgst amt", "cgst amount"],
        "sgst_amount": ["sgst", "sgst amt", "sgst amount"],
        "igst_amount": ["igst", "igst amt", "igst amount"],
    }

    def extract_tables_from_page(self, page: pymupdf.Page) -> list[dict]:
        """
        Find and extract all structured line items from a single PDF page.
        Returns a list of line item dicts suitable for InvoiceSchema.line_items.
        """
        line_items = []
        try:
            tab_finder = page.find_tables()
            for tab in tab_finder:
                extracted = tab.extract()
                if not extracted or len(extracted) < 2:
                    continue
                items = self._parse_table_rows(extracted)
                line_items.extend(items)
        except Exception as e:
            logger.debug(f"Table extraction error on page: {e}")
        return line_items

    def extract_tables_from_doc(self, doc: pymupdf.Document) -> list[dict]:
        """Extract line items from all pages in a document."""
        all_items = []
        for page in doc:
            items = self.extract_tables_from_page(page)
            all_items.extend(items)
        return all_items

    def _detect_column_mapping(self, header_row: list[Optional[str]]) -> dict[str, int]:
        """
        Detect column index for each field name based on header text.
        """
        mapping = {}
        for idx, cell in enumerate(header_row):
            if not cell:
                continue
            cell_clean = str(cell).lower().replace("\n", " ").strip()
            if "taxable" in cell_clean:
                mapping["amount"] = idx
            for field_name, kws in self.COLUMN_KEYWORDS.items():
                if field_name not in mapping:
                    if any(kw in cell_clean for kw in kws):
                        mapping[field_name] = idx
                        break
        return mapping

    def _clean_number(self, val: Optional[str]) -> float:
        """Extract clean float from string like '3,000.00', 'Rs. 420.00', etc."""
        if not val:
            return 0.0
        s = str(val).replace(",", "").strip()
        m = re.search(r"[-+]?\d*\.?\d+", s)
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return 0.0
        return 0.0

    def _parse_table_rows(self, rows: list[list[Optional[str]]]) -> list[dict]:
        """
        Parses extracted rows (header + data) into standardized line item dicts.
        """
        # Find the header row (sometimes header is row 0 or row 1)
        header_idx = 0
        mapping = {}
        for i in range(min(3, len(rows))):
            curr_map = self._detect_column_mapping(rows[i])
            if "description" in curr_map or "amount" in curr_map:
                header_idx = i
                mapping = curr_map
                break

        if not mapping:
            return []

        line_items = []
        for row in rows[header_idx + 1:]:
            if not row or all(c is None or str(c).strip() == "" for c in row):
                continue

            # Check if this row is a summary / total / payment / tender / remark row
            row_text = " ".join(str(c or "").lower() for c in row)
            if any(term in row_text for term in [
                "sub total", "subtotal", "net total", "net bill", "total bill", "grand total",
                "+cgst", "+sgst", "+igst", "gst total", "tender", "tax summary",
                "coupon discount", "loyalty discount", "change due", "refund amt",
                "gift wrap", "delivery charges", "cod charges", "remarks"
            ]):
                continue
            if row[0] and any(term in str(row[0]).lower() for term in ["total", "subtotal", "taxable amount", "grand total", "gross"]):
                continue

            desc_idx = mapping.get("description")
            desc = str(row[desc_idx]).replace("\n", " ").strip() if desc_idx is not None and desc_idx < len(row) and row[desc_idx] else ""

            # If no description or only special chars, skip row
            if not desc or desc in {"-", "—", "None", ""}:
                # Fallback: check other text cells
                for cell in row:
                    if cell and len(str(cell).strip()) > 3 and not re.match(r"^[\d,.\s/:-]+$", str(cell)):
                        desc = str(cell).replace("\n", " ").strip()
                        break

            if not desc or len(desc) < 2:
                continue

            desc_clean = desc.lower().strip()
            if any(term == desc_clean or term in desc_clean for term in [
                "total", "sub total", "subtotal", "hsn code", "grand total", "tax summary", "gst total", "net total", "net bill"
            ]):
                continue

            # Parse numeric fields
            amount = 0.0
            if "amount" in mapping and mapping["amount"] < len(row):
                amount = self._clean_number(row[mapping["amount"]])

            rate = amount
            if "rate" in mapping and mapping["rate"] < len(row):
                rate = self._clean_number(row[mapping["rate"]])
                if rate == 0.0 and amount > 0:
                    rate = amount

            qty = 1.0
            if "quantity" in mapping and mapping["quantity"] < len(row):
                q = self._clean_number(row[mapping["quantity"]])
                if q > 0:
                    qty = q

            unit = "NOS"
            if "unit" in mapping and mapping["unit"] < len(row) and row[mapping["unit"]]:
                u = str(row[mapping["unit"]]).strip()
                if u and len(u) < 10:
                    unit = u

            hsn = None
            if "hsn_code" in mapping and mapping["hsn_code"] < len(row) and row[mapping["hsn_code"]]:
                h = str(row[mapping["hsn_code"]]).strip()
                if h and h != "None":
                    hsn = h

            item = {
                "description": desc,
                "quantity": qty,
                "unit": unit,
                "rate": rate,
                "amount": amount if amount > 0 else (rate * qty),
                "hsn_code": hsn,
            }
            line_items.append(item)

        return line_items

    def extract_tables_from_spatial_ocr(self, ocr_result) -> list[dict]:
        """
        Reconstructs table line items from scanned OCR text blocks & words using
        spatial 2D bounding box column alignment.
        """
        if not ocr_result or not getattr(ocr_result, "text_blocks", None):
            return []

        blocks = ocr_result.text_blocks
        if not blocks:
            return []

        # 1. Group blocks/lines by vertical Y-coordinate bands
        sorted_blocks = sorted(blocks, key=lambda b: (b.to_xyxy()[1] if hasattr(b, "to_xyxy") else 0.0))

        # Detect potential header row by keyword match
        header_idx = -1
        header_cols: dict[str, tuple[float, float]] = {}  # col_name -> (x_min, x_max)

        for idx, block in enumerate(sorted_blocks):
            line_words = block.words if getattr(block, "words", None) else []
            line_text = block.text.lower()
            matched_cols = {}
            for col_name, kws in self.COLUMN_KEYWORDS.items():
                for kw in kws:
                    if kw in line_text:
                        # Find word bounding box for this keyword if available
                        col_box = block.to_xyxy() if hasattr(block, "to_xyxy") else [0, 0, 100, 20]
                        for w in line_words:
                            if kw in w.text.lower():
                                col_box = w.to_xyxy() if hasattr(w, "to_xyxy") else col_box
                                break
                        matched_cols[col_name] = (col_box[0], col_box[2])
                        break
            if len(matched_cols) >= 2 and ("description" in matched_cols or "amount" in matched_cols):
                header_idx = idx
                header_cols = matched_cols
                break

        if header_idx == -1 or not header_cols:
            return []

        # Establish column X boundaries
        items = []
        for block in sorted_blocks[header_idx + 1:]:
            line_text = block.text.strip()
            if not line_text:
                continue

            # Skip summary / total lines
            if any(term in line_text.lower() for term in ["grand total", "subtotal", "total amount", "tax amount"]):
                break

            words = block.words if getattr(block, "words", None) else []
            if not words:
                continue

            row_data: dict[str, list[str]] = {k: [] for k in self.COLUMN_KEYWORDS}
            unmatched_words = []

            for w in words:
                w_box = w.to_xyxy() if hasattr(w, "to_xyxy") else [0, 0, 0, 0]
                w_cx = (w_box[0] + w_box[2]) / 2.0
                assigned = False
                for col_name, (c_x1, c_x2) in header_cols.items():
                    # Tolerant column width matching
                    tol = max(20.0, (c_x2 - c_x1) * 0.5)
                    if (c_x1 - tol) <= w_cx <= (c_x2 + tol):
                        row_data[col_name].append(w.text)
                        assigned = True
                        break
                if not assigned:
                    unmatched_words.append(w.text)

            desc = " ".join(row_data["description"])
            if not desc and unmatched_words:
                desc = " ".join(unmatched_words)

            if not desc or len(desc) < 2:
                continue

            amount = self._clean_number(" ".join(row_data["amount"]))
            rate = self._clean_number(" ".join(row_data["rate"])) if row_data["rate"] else amount
            qty = self._clean_number(" ".join(row_data["quantity"])) if row_data["quantity"] else 1.0
            if qty <= 0:
                qty = 1.0
            if rate <= 0 and amount > 0:
                rate = amount

            items.append({
                "description": desc,
                "quantity": qty,
                "unit": " ".join(row_data["unit"]) or "NOS",
                "rate": rate if rate > 0 else amount,
                "amount": amount if amount > 0 else (rate * qty),
                "hsn_code": " ".join(row_data["hsn_code"]) or None,
            })

        return items
