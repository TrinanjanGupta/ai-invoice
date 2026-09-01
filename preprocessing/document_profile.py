"""
preprocessing/document_profile.py

Normalized internal document representation for invoice analysis, template retrieval,
and field-level extraction.

Combines OCR words (from PaddleOCR or digital PDF native text), YOLO visual regions,
page geometry, and deterministic text/layout/anchor fingerprints into a structured
DocumentProfile object.
"""

from __future__ import annotations
import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional, Any, Union, Sequence


# Canonical invoice anchor keywords for anchor signature computation
CANONICAL_ANCHORS = [
    "invoice", "tax invoice", "bill to", "ship to", "gstin", "pan",
    "invoice no", "inv no", "bill no", "invoice date", "date", "due date",
    "po no", "order no", "place of supply", "subtotal", "taxable value",
    "taxable amount", "cgst", "sgst", "igst", "tax amount", "total tax",
    "discount", "round off", "grand total", "net amount", "total payable",
    "amount in words", "bank", "account no", "ac no", "ifsc", "branch",
    "sls code", "remarks", "certified remarks", "authorized signatory"
]

GSTIN_PATTERN = re.compile(r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}\b")
CURRENCY_PATTERNS = [
    ("INR", re.compile(r"(?:₹|rs\.?|inr)\s*", re.IGNORECASE)),
    ("USD", re.compile(r"(?:\$|usd)\s*", re.IGNORECASE)),
    ("EUR", re.compile(r"(?:€|eur)\s*", re.IGNORECASE)),
    ("GBP", re.compile(r"(?:£|gbp)\s*", re.IGNORECASE)),
]


def normalize_box(box: Sequence[Union[int, float]], width: int, height: int) -> list[int]:
    """
    Normalizes a bounding box to standard 0-1000 coordinate space.
    Strictly guarantees 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000.
    """
    if len(box) == 4 and isinstance(box[0], (int, float)):
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    else:
        # Polygon points [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x1, y1, x2, y2 = float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))

    w_safe = max(1.0, float(width))
    h_safe = max(1.0, float(height))

    nx1 = max(0, min(1000, int(1000.0 * x1 / w_safe)))
    ny1 = max(0, min(1000, int(1000.0 * y1 / h_safe)))
    nx2 = max(0, min(1000, int(1000.0 * x2 / w_safe)))
    ny2 = max(0, min(1000, int(1000.0 * y2 / h_safe)))

    if nx2 <= nx1:
        nx2 = min(1000, nx1 + 2)
        if nx2 <= nx1:
            nx1 = max(0, nx2 - 2)

    if ny2 <= ny1:
        ny2 = min(1000, ny1 + 2)
        if ny2 <= ny1:
            ny1 = max(0, ny2 - 2)

    return [int(nx1), int(ny1), int(nx2), int(ny2)]


@dataclass
class WordToken:
    text: str
    bbox_norm: list[int]          # [x1, y1, x2, y2] in 0-1000
    bbox_raw: list[float]         # [x1, y1, x2, y2] in original pixels / points
    confidence: float = 0.99
    page: int = 1
    block_no: int = 0
    line_no: int = 0
    word_no: int = 0
    source: str = "ocr"           # "native_pdf" | "paddleocr" | "easyocr"

    @property
    def clean_text(self) -> str:
        return self.text.strip()

    @property
    def center_norm(self) -> tuple[float, float]:
        return ((self.bbox_norm[0] + self.bbox_norm[2]) / 2.0, (self.bbox_norm[1] + self.bbox_norm[3]) / 2.0)


@dataclass
class RegionBlock:
    label: str                    # e.g. "header", "vendor_block", "buyer_block", "line_items", "totals"
    bbox_norm: list[int]          # [x1, y1, x2, y2] in 0-1000
    bbox_raw: list[float]
    confidence: float = 0.90
    page: int = 1

    @property
    def center_norm(self) -> tuple[float, float]:
        return ((self.bbox_norm[0] + self.bbox_norm[2]) / 2.0, (self.bbox_norm[1] + self.bbox_norm[3]) / 2.0)


@dataclass
class DocumentProfile:
    """
    Unified, normalized document representation for invoice processing.
    Stores both exact identity hashes (for O(1) lookup) and rich structured
    features (for mathematical similarity: Jaccard, spatial graph, region topology).
    """
    page_count: int
    width: int
    height: int
    aspect_ratio: float
    is_digital_native: bool = False
    quality_score: float = 1.0

    words: list[WordToken] = field(default_factory=list)
    regions: list[RegionBlock] = field(default_factory=list)

    # ── Structured Mathematical Features (For Real Similarity) ────────────────
    anchor_set: set[str] = field(default_factory=set)
    anchor_positions: dict[str, tuple[float, float]] = field(default_factory=dict)
    region_topology: list[dict[str, Any]] = field(default_factory=list)
    aspect_bucket: int = 14

    # ── Identity & Invariant Fingerprints (For O(1) Exact/Family Indexing) ────
    exact_fingerprint: str = ""
    family_fingerprint: str = ""
    text_signature: str = ""
    layout_signature: str = ""
    anchor_signature: str = ""
    region_signature: str = ""

    # Inferred metadata
    vendor_gstin: Optional[str] = None
    buyer_gstin: Optional[str] = None
    vendor_name_candidate: Optional[str] = None
    detected_language: str = "en"
    currency_symbol: str = "INR"
    text_density: float = 0.0

    def __post_init__(self):
        self.aspect_bucket = int(round(float(self.aspect_ratio) * 10))

        if self.regions and not self.region_topology:
            for r in self.regions:
                self.region_topology.append({
                    "label": r.label,
                    "bbox_norm": list(r.bbox_norm),
                    "center": list(r.center_norm),
                    "page": r.page,
                })

        if not self.words:
            if not self.layout_signature:
                self.layout_signature = f"ar:{self.aspect_bucket}|pg:{self.page_count}"
            if not self.family_fingerprint:
                self.family_fingerprint = hashlib.sha256(self.layout_signature.encode("utf-8")).hexdigest()[:16]
            if not self.exact_fingerprint:
                self.exact_fingerprint = self.family_fingerprint
            return

        full_text = " ".join(w.text for w in self.words if w.text)
        lower_full_text = full_text.lower()

        # Extract structured anchor set & spatial centroids
        if not self.anchor_set:
            for anc in CANONICAL_ANCHORS:
                if anc in lower_full_text:
                    clean_anc = anc.replace(" ", "_").replace(":", "").strip("_")
                    matched_seqs = self.find_anchor_tokens(anc)
                    if matched_seqs:
                        self.anchor_set.add(clean_anc)
                        # Compute centroid across all occurrences
                        first_seq = matched_seqs[0]
                        cx = sum(w.center_norm[0] for w in first_seq) / len(first_seq)
                        cy = sum(w.center_norm[1] for w in first_seq) / len(first_seq)
                        self.anchor_positions[clean_anc] = (round(cx, 1), round(cy, 1))

        # Build Invariant Structural Family Fingerprint (Layout + Anchors + Regions, NO variable text)
        sorted_anchors = sorted(list(self.anchor_set))
        reg_labels = [r.get("label", "") for r in self.region_topology]
        family_seed = f"ar:{self.aspect_bucket}|pg:{self.page_count}|anc:{'|'.join(sorted_anchors[:20])}|reg:{'|'.join(reg_labels[:10])}"
        self.family_fingerprint = hashlib.sha256(family_seed.encode("utf-8")).hexdigest()[:16]

        # Build Exact Geometric Fingerprint
        geom_parts = [f"{anc}:{pos[0]},{pos[1]}" for anc, pos in sorted(self.anchor_positions.items())]
        exact_seed = f"{self.family_fingerprint}|geom:{'|'.join(geom_parts)}"
        self.exact_fingerprint = hashlib.sha256(exact_seed.encode("utf-8")).hexdigest()[:16]

        # Backward-compatible signatures
        if not self.text_signature:
            clean_words = [re.sub(r"[^a-zA-Z0-9]", "", w.text.lower()) for w in self.words if w.text]
            clean_words = [w for w in clean_words if len(w) > 2][:30]
            text_sig_raw = " ".join(clean_words)
            self.text_signature = hashlib.sha256(text_sig_raw.encode("utf-8")).hexdigest()[:16] if text_sig_raw else "empty"

        if not self.anchor_signature:
            anchor_sig_raw = "|".join(sorted_anchors)
            self.anchor_signature = hashlib.sha256(anchor_sig_raw.encode("utf-8")).hexdigest()[:16] if anchor_sig_raw else "none"

        if not self.region_signature:
            sorted_regions = sorted(self.regions, key=lambda r: (r.bbox_norm[1], r.bbox_norm[0]))
            reg_parts = []
            for r in sorted_regions[:15]:
                gx = r.bbox_norm[0] // 100
                gy = r.bbox_norm[1] // 100
                reg_parts.append(f"{r.label}_{gx}_{gy}")
            reg_sig_raw = "|".join(reg_parts)
            self.region_signature = hashlib.sha256(reg_sig_raw.encode("utf-8")).hexdigest()[:16] if reg_sig_raw else "no_regions"

        if not self.layout_signature:
            self.layout_signature = self.exact_fingerprint

        if not self.vendor_gstin:
            gstin_matches = GSTIN_PATTERN.findall(full_text)
            if gstin_matches:
                self.vendor_gstin = gstin_matches[0]

    def get_full_text(self, page: Optional[int] = None) -> str:
        """Return full text reconstructed from word tokens."""
        target_words = self.words if page is None else [w for w in self.words if w.page == page]
        if not target_words:
            return ""

        # Group words by page and line
        lines_dict: dict[tuple[int, int, int], list[WordToken]] = {}
        for w in target_words:
            key = (w.page, w.block_no, w.line_no)
            if key not in lines_dict:
                lines_dict[key] = []
            lines_dict[key].append(w)

        sorted_keys = sorted(lines_dict.keys(), key=lambda k: (k[0], min(w.bbox_norm[1] for w in lines_dict[k]), min(w.bbox_norm[0] for w in lines_dict[k])))
        rendered_lines = []
        for k in sorted_keys:
            ws = sorted(lines_dict[k], key=lambda w: w.bbox_norm[0])
            rendered_lines.append(" ".join(w.text for w in ws if w.text))

        return "\n".join(rendered_lines)

    def find_words_in_box(self, box_norm: list[int], page: int = 1) -> list[WordToken]:
        """
        Finds all word tokens whose centers fall within the normalized bounding box.
        box_norm: [x1, y1, x2, y2] in 0-1000
        """
        x1, y1, x2, y2 = box_norm
        matched = []
        for w in self.words:
            if w.page != page:
                continue
            cx, cy = w.center_norm
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                matched.append(w)
        return sorted(matched, key=lambda w: (w.bbox_norm[1], w.bbox_norm[0]))

    def find_anchor_tokens(self, anchor_phrase: str, page: int = 1) -> list[list[WordToken]]:
        """
        Finds sequences of word tokens matching the given anchor phrase (case-insensitive).
        Returns matching token sequences.
        """
        clean_target = anchor_phrase.strip().lower()
        target_tokens = [re.sub(r"[^a-z0-9]", "", t) for t in re.split(r"\s+", clean_target) if t]
        target_tokens = [t for t in target_tokens if t]
        if not target_tokens:
            return []

        page_words = [w for w in self.words if w.page == page]
        matches: list[list[WordToken]] = []

        if len(target_tokens) == 1:
            t = target_tokens[0]
            for w in page_words:
                w_clean = re.sub(r"[^a-z0-9]", "", w.text.lower())
                if w_clean == t or t in w_clean:
                    matches.append([w])
            return matches

        # Multi-word sequence match
        for i in range(len(page_words) - len(target_tokens) + 1):
            window = page_words[i : i + len(target_tokens)]
            matched_all = True
            for w, t in zip(window, target_tokens):
                w_clean = re.sub(r"[^a-z0-9]", "", w.text.lower())
                if w_clean != t and t not in w_clean:
                    matched_all = False
                    break
            if matched_all:
                matches.append(window)

        return matches

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_count": self.page_count,
            "width": self.width,
            "height": self.height,
            "aspect_ratio": self.aspect_ratio,
            "is_digital_native": self.is_digital_native,
            "quality_score": self.quality_score,
            "text_signature": self.text_signature,
            "layout_signature": self.layout_signature,
            "anchor_signature": self.anchor_signature,
            "region_signature": self.region_signature,
            "vendor_gstin": self.vendor_gstin,
            "buyer_gstin": self.buyer_gstin,
            "vendor_name_candidate": self.vendor_name_candidate,
            "detected_language": self.detected_language,
            "currency_symbol": self.currency_symbol,
            "text_density": self.text_density,
            "words_count": len(self.words),
            "regions_count": len(self.regions),
        }

    @classmethod
    def from_ocr_and_regions(
        cls,
        ocr_results: dict[str, Any],
        regions: list[Any],
        width: int,
        height: int,
        page_count: int = 1,
        is_digital_native: bool = False,
        quality_score: float = 1.0,
        page_num: int = 1,
    ) -> DocumentProfile:
        """
        Builds a DocumentProfile from OCR results and YOLO detected regions.
        """
        w_safe = max(1, width)
        h_safe = max(1, height)
        aspect_ratio = round(float(h_safe) / float(w_safe), 2)

        # 1. Convert words to normalized WordTokens
        word_tokens: list[WordToken] = []
        seen_word_keys = set()

        # Prioritize full_page OCR if present, or iterate across all region results
        priority_keys = [k for k in ocr_results.keys() if "full_page" in k]
        other_keys = [k for k in ocr_results.keys() if "full_page" not in k]
        iter_keys = priority_keys + other_keys if priority_keys else list(ocr_results.keys())

        for k in iter_keys:
            res = ocr_results[k]
            if not res or not hasattr(res, "text_blocks"):
                continue

            for b_idx, block in enumerate(res.text_blocks):
                # Extract words from TextBlock
                words_list = getattr(block, "words", None) or []
                if not words_list:
                    # Decompose line into words if not already done
                    from ocr.extractor import decompose_line_into_words
                    words_list = decompose_line_into_words(block)

                for w_idx, w in enumerate(words_list):
                    w_text = getattr(w, "text", "")
                    if not w_text or not str(w_text).strip():
                        continue

                    raw_box = getattr(w, "bbox", [0, 0, 0, 0])
                    if hasattr(w, "to_xyxy"):
                        xyxy_box = w.to_xyxy()
                    elif len(raw_box) == 4 and isinstance(raw_box[0], (int, float)):
                        xyxy_box = [float(b) for b in raw_box]
                    else:
                        xs = [p[0] for p in raw_box]
                        ys = [p[1] for p in raw_box]
                        xyxy_box = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]

                    norm_box = normalize_box(xyxy_box, w_safe, h_safe)
                    dedup_key = (norm_box[0] // 5, norm_box[1] // 5, w_text.strip().lower())
                    if dedup_key in seen_word_keys:
                        continue
                    seen_word_keys.add(dedup_key)

                    conf = float(getattr(w, "confidence", 0.95))
                    source = getattr(w, "source", "paddleocr") if not is_digital_native else "native_pdf"

                    word_tokens.append(
                        WordToken(
                            text=str(w_text).strip(),
                            bbox_norm=norm_box,
                            bbox_raw=xyxy_box,
                            confidence=conf,
                            page=page_num,
                            block_no=b_idx,
                            line_no=w_idx // 8,
                            word_no=w_idx,
                            source=source,
                        )
                    )

        # 2. Convert YOLO regions to RegionBlocks
        region_blocks: list[RegionBlock] = []
        for r in regions:
            r_label = getattr(r, "label", "region") if not isinstance(r, dict) else r.get("label", "region")
            r_bbox = getattr(r, "bbox", [0, 0, 0, 0]) if not isinstance(r, dict) else r.get("bbox", [0, 0, 0, 0])
            r_conf = getattr(r, "confidence", 0.85) if not isinstance(r, dict) else r.get("confidence", 0.85)
            norm_rbox = normalize_box(r_bbox, w_safe, h_safe)
            region_blocks.append(
                RegionBlock(
                    label=str(r_label),
                    bbox_norm=norm_rbox,
                    bbox_raw=[float(b) for b in r_bbox],
                    confidence=float(r_conf),
                    page=page_num,
                )
            )

        # 3. Compute Signatures
        full_text = " ".join(w.text for w in word_tokens)

        # A. Text Signature: Hash of first 20 clean words and prominent vendor/header terms
        clean_words = [re.sub(r"[^a-zA-Z0-9]", "", w.text.lower()) for w in word_tokens]
        clean_words = [w for w in clean_words if len(w) > 2][:30]
        text_sig_raw = " ".join(clean_words)
        text_signature = hashlib.sha256(text_sig_raw.encode("utf-8")).hexdigest()[:16] if text_sig_raw else "empty"

        # B. Anchor Signature: Presence hash of canonical invoice anchors
        present_anchors = []
        lower_full_text = full_text.lower()
        for anc in CANONICAL_ANCHORS:
            if anc in lower_full_text:
                present_anchors.append(anc.replace(" ", "_"))
        anchor_sig_raw = "|".join(sorted(present_anchors))
        anchor_signature = hashlib.sha256(anchor_sig_raw.encode("utf-8")).hexdigest()[:16] if anchor_sig_raw else "none"

        # C. Region Signature: YOLO macro-region topology (sorted top-to-bottom)
        sorted_regions = sorted(region_blocks, key=lambda r: (r.bbox_norm[1], r.bbox_norm[0]))
        reg_parts = []
        for r in sorted_regions[:15]:
            gx = r.bbox_norm[0] // 100
            gy = r.bbox_norm[1] // 100
            reg_parts.append(f"{r.label}_{gx}_{gy}")
        reg_sig_raw = "|".join(reg_parts)
        region_signature = hashlib.sha256(reg_sig_raw.encode("utf-8")).hexdigest()[:16] if reg_sig_raw else "no_regions"

        # D. Layout Signature: Hybrid spatial hash
        layout_sig_raw = f"ar:{aspect_ratio}|pg:{page_count}|reg:{region_signature}|anc:{anchor_signature}"
        layout_signature = hashlib.sha256(layout_sig_raw.encode("utf-8")).hexdigest()[:16]

        # 4. Extract Deterministic Metadata
        vendor_gstin = None
        gstin_matches = GSTIN_PATTERN.findall(full_text)
        if gstin_matches:
            vendor_gstin = gstin_matches[0]

        currency = "INR"
        for curr_name, curr_pat in CURRENCY_PATTERNS:
            if curr_pat.search(full_text):
                currency = curr_name
                break

        text_density = len(word_tokens) / 1000.0

        return cls(
            page_count=page_count,
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
            is_digital_native=is_digital_native,
            quality_score=quality_score,
            words=word_tokens,
            regions=region_blocks,
            text_signature=text_signature,
            layout_signature=layout_signature,
            anchor_signature=anchor_signature,
            region_signature=region_signature,
            vendor_gstin=vendor_gstin,
            currency_symbol=currency,
            text_density=text_density,
        )

