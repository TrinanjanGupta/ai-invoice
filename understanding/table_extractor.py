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

            # Check if this row is a summary / total row
            first_cell = str(row[0] or "").lower()
            if any(term in first_cell for term in ["total", "subtotal", "taxable amount", "grand total", "gross"]):
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
