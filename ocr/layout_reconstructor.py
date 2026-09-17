"""
ocr/layout_reconstructor.py

OCR -> Lines -> Blocks Layout Reconstruction Engine.

Transforms unstructured raw OCR words/tokens (from PaddleOCR, EasyOCR, TrOCR)
into geometrically coherent lines, detects 2-column gutters to prevent
interleaving, clusters lines into semantic blocks, and classifies macro zones.

Provides structured intermediate evidence for TIE and LayoutLM on scanned invoices:
1. Horizontal Line Merging: Merges fragmented tokens with vertical IoU overlap into unified lines.
2. Column Detection & Reading Order: Detects left/right gutters so 'Bill To' and 'Invoice Date' don't interleave.
3. Block Clustering: Clusters lines with small line spacing into cohesive paragraphs/blocks.
4. Zone Classification: Reconstructs vendor, buyer, header, totals, and footer zones when YOLO regions fail.
"""

import re
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Any
from loguru import logger

from ocr.extractor import OCRWord, TextBlock, OCRResult
from preprocessing.document_profile import WordToken, RegionBlock, normalize_box


@dataclass
class ReconstructedLine:
    words: list[OCRWord]
    text: str
    bbox: list[float]          # [x1, y1, x2, y2]
    bbox_norm: list[int]       # [0..1000]
    confidence: float
    line_no: int
    column_id: int = 0         # 0 = full/left, 1 = right column

    @property
    def height(self) -> float:
        return max(1.0, self.bbox[3] - self.bbox[1])

    @property
    def width(self) -> float:
        return max(1.0, self.bbox[2] - self.bbox[0])

    @property
    def center_y(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0


@dataclass
class ReconstructedBlock:
    lines: list[ReconstructedLine]
    text: str
    bbox: list[float]
    bbox_norm: list[int]
    confidence: float
    block_id: int
    zone_label: str = "general"


class LayoutReconstructor:
    """
    Reconstructs reading order, lines, and semantic blocks from raw OCR tokens.
    """

    def __init__(self, page_width: int = 1200, page_height: int = 1600):
        self.page_width = max(1, page_width)
        self.page_height = max(1, page_height)

    def to_norm(self, bbox: list[float]) -> list[int]:
        w = max(1.0, float(self.page_width))
        h = max(1.0, float(self.page_height))
        return [
            max(0, min(1000, int(round(1000.0 * bbox[0] / w)))),
            max(0, min(1000, int(round(1000.0 * bbox[1] / h)))),
            max(0, min(1000, int(round(1000.0 * bbox[2] / w)))),
            max(0, min(1000, int(round(1000.0 * bbox[3] / h)))),
        ]

    def reconstruct(
        self,
        raw_blocks_or_words: list[Any],
        page_num: int = 1,
        yolo_regions: Optional[list[Any]] = None,
    ) -> tuple[OCRResult, list[RegionBlock]]:
        """
        Main entry point:
        Takes raw PaddleOCR TextBlocks or OCRWords, reconstructs lines, resolves
        column reading order, clusters blocks, and classifies zones.
        Returns:
          (OCRResult containing structured TextBlocks in true reading order,
           list of reconstructed RegionBlocks)
        """
        # 1. Flatten all words
        words: list[OCRWord] = []
        for item in raw_blocks_or_words:
            if isinstance(item, TextBlock):
                if item.words:
                    words.extend(item.words)
                else:
                    # Decompose line
                    from ocr.extractor import decompose_line_into_words
                    xyxy = item.to_xyxy()
                    w_list = decompose_line_into_words(item.text, xyxy, item.confidence)
                    words.extend(w_list)
            elif isinstance(item, OCRWord):
                words.append(item)

        if not words:
            return (
                OCRResult(region_label="full_page", text_blocks=[], full_text="", avg_confidence=0.0),
                []
            )

        # 2. Merge words into horizontal lines
        lines = self._merge_words_into_lines(words)

        # 3. Sort lines by 2-column gutter and natural reading order
        sorted_lines = self._sort_reading_order(lines)

        # 4. Cluster lines into cohesive blocks
        blocks = self._cluster_lines_into_blocks(sorted_lines)

        # 5. Classify macro zones (vendor, buyer, header, totals, footer)
        classified_blocks, macro_regions = self._classify_zones(blocks, page_num, yolo_regions)

        # 6. Build final TextBlock list and OCRResult
        final_text_blocks: list[TextBlock] = []
        for blk in classified_blocks:
            for line in blk.lines:
                poly_bbox = [
                    [line.bbox[0], line.bbox[1]],
                    [line.bbox[2], line.bbox[1]],
                    [line.bbox[2], line.bbox[3]],
                    [line.bbox[0], line.bbox[3]],
                ]
                final_text_blocks.append(
                    TextBlock(
                        text=line.text,
                        confidence=line.confidence,
                        bbox=poly_bbox,
                        region_label=blk.zone_label,
                        words=line.words,
                    )
                )

        full_text = "\n".join(b.text for b in final_text_blocks)
        avg_conf = sum(b.confidence for b in final_text_blocks) / len(final_text_blocks) if final_text_blocks else 0.0

        ocr_result = OCRResult(
            region_label="full_page",
            text_blocks=final_text_blocks,
            full_text=full_text,
            avg_confidence=round(avg_conf, 3),
            engine="reconstructed_layout",
        )

        return ocr_result, macro_regions

    def _merge_words_into_lines(self, words: list[OCRWord]) -> list[ReconstructedLine]:
        """
        Clusters individual word tokens into horizontal text lines.
        Words on the same baseline with vertical overlap belong to the same line.
        """
        # Calculate median word height
        heights = [w.to_xyxy()[3] - w.to_xyxy()[1] for w in words]
        med_height = float(np.median(heights)) if heights else 15.0
        med_height = max(8.0, min(80.0, med_height))

        # Sort words primarily top-to-bottom, then left-to-right
        sorted_words = sorted(
            words,
            key=lambda w: (w.to_xyxy()[1], w.to_xyxy()[0])
        )

        lines: list[list[OCRWord]] = []
        for w in sorted_words:
            wx1, wy1, wx2, wy2 = w.to_xyxy()
            w_cy = (wy1 + wy2) / 2.0
            w_h = max(1.0, wy2 - wy1)

            placed = False
            for line_words in lines:
                l_y1 = min(lw.to_xyxy()[1] for lw in line_words)
                l_y2 = max(lw.to_xyxy()[3] for lw in line_words)
                l_cy = (l_y1 + l_y2) / 2.0
                l_h = max(1.0, l_y2 - l_y1)

                # Vertical IoU overlap or centroid proximity
                v_overlap = max(0.0, min(wy2, l_y2) - max(wy1, l_y1))
                overlap_ratio = v_overlap / min(w_h, l_h)

                if overlap_ratio >= 0.45 or abs(w_cy - l_cy) <= min(w_h, l_h) * 0.45:
                    line_words.append(w)
                    placed = True
                    break

            if not placed:
                lines.append([w])

        reconstructed: list[ReconstructedLine] = []
        for idx, line_words in enumerate(lines):
            # Sort words within line left-to-right
            line_words.sort(key=lambda w: w.to_xyxy()[0])

            # Check if there is a massive horizontal gap within the line (e.g. 2 columns on the same baseline)
            # Split line if gap > 120 pixels or 20% of page width
            split_sublines: list[list[OCRWord]] = [[]]
            for w in line_words:
                if not split_sublines[-1]:
                    split_sublines[-1].append(w)
                else:
                    prev_w = split_sublines[-1][-1]
                    gap = w.to_xyxy()[0] - prev_w.to_xyxy()[2]
                    if gap > max(80.0, self.page_width * 0.12):
                        # Big horizontal gap: split into two separate column segments
                        split_sublines.append([w])
                    else:
                        split_sublines[-1].append(w)

            for sub_idx, sub_words in enumerate(split_sublines):
                if not sub_words:
                    continue
                min_x = min(w.to_xyxy()[0] for w in sub_words)
                min_y = min(w.to_xyxy()[1] for w in sub_words)
                max_x = max(w.to_xyxy()[2] for w in sub_words)
                max_y = max(w.to_xyxy()[3] for w in sub_words)
                line_text = " ".join(w.text for w in sub_words)
                line_conf = sum(w.confidence for w in sub_words) / len(sub_words)
                bbox_raw = [min_x, min_y, max_x, max_y]
                bbox_norm = self.to_norm(bbox_raw)

                # Column determination: left (0) vs right (1)
                col_id = 1 if min_x >= self.page_width * 0.45 else 0

                reconstructed.append(
                    ReconstructedLine(
                        words=sub_words,
                        text=line_text,
                        bbox=bbox_raw,
                        bbox_norm=bbox_norm,
                        confidence=round(line_conf, 3),
                        line_no=idx * 10 + sub_idx,
                        column_id=col_id,
                    )
                )

        return reconstructed

    def _sort_reading_order(self, lines: list[ReconstructedLine]) -> list[ReconstructedLine]:
        """
        Sorts lines according to true reading order:
        In two-column zones (like Bill To vs Invoice Details), left column lines
        are read together, followed by right column lines, rather than interleaving.
        """
        # Divide into vertical zones:
        # 1. Header (y_norm <= 380) -> potentially 2 columns
        # 2. Body / Table (380 < y_norm <= 700) -> 1 column reading order
        # 3. Totals & Footer (y_norm > 700) -> right-aligned totals, left-aligned bank
        top_lines = [l for l in lines if l.bbox_norm[3] <= 380]
        mid_lines = [l for l in lines if 380 < l.bbox_norm[3] <= 700]
        bot_lines = [l for l in lines if l.bbox_norm[3] > 700]

        # Top section: sort left column top-to-bottom, then right column top-to-bottom
        left_top = sorted([l for l in top_lines if l.column_id == 0], key=lambda l: l.bbox[1])
        right_top = sorted([l for l in top_lines if l.column_id == 1], key=lambda l: l.bbox[1])
        sorted_top = left_top + right_top

        # Mid section: sort top-to-bottom, left-to-right
        sorted_mid = sorted(mid_lines, key=lambda l: (l.bbox[1], l.bbox[0]))

        # Bottom section: sort left-to-right, top-to-bottom
        sorted_bot = sorted(bot_lines, key=lambda l: (l.bbox[1], l.bbox[0]))

        return sorted_top + sorted_mid + sorted_bot

    def _cluster_lines_into_blocks(self, lines: list[ReconstructedLine]) -> list[ReconstructedBlock]:
        """
        Clusters lines into coherent blocks based on line spacing and horizontal alignment.
        """
        if not lines:
            return []

        blocks: list[list[ReconstructedLine]] = []
        for line in lines:
            if not blocks:
                blocks.append([line])
                continue

            prev_line = blocks[-1][-1]
            y_gap = line.bbox[1] - prev_line.bbox[3]
            med_h = prev_line.height

            # Conditions to stay in same block:
            # 1. Same column
            # 2. Small vertical gap (<= 1.8 * line height)
            # 3. Not moving backwards vertically
            if (
                line.column_id == prev_line.column_id
                and 0 <= y_gap <= med_h * 1.8
                and line.bbox[1] >= prev_line.bbox[1]
            ):
                blocks[-1].append(line)
            else:
                blocks.append([line])

        reconstructed_blocks: list[ReconstructedBlock] = []
        for b_idx, blk_lines in enumerate(blocks):
            min_x = min(l.bbox[0] for l in blk_lines)
            min_y = min(l.bbox[1] for l in blk_lines)
            max_x = max(l.bbox[2] for l in blk_lines)
            max_y = max(l.bbox[3] for l in blk_lines)
            blk_text = "\n".join(l.text for l in blk_lines)
            blk_conf = sum(l.confidence for l in blk_lines) / len(blk_lines)
            bbox_raw = [min_x, min_y, max_x, max_y]

            reconstructed_blocks.append(
                ReconstructedBlock(
                    lines=blk_lines,
                    text=blk_text,
                    bbox=bbox_raw,
                    bbox_norm=self.to_norm(bbox_raw),
                    confidence=round(blk_conf, 3),
                    block_id=b_idx,
                )
            )

        return reconstructed_blocks

    def _classify_zones(
        self,
        blocks: list[ReconstructedBlock],
        page_num: int,
        yolo_regions: Optional[list[Any]] = None,
    ) -> tuple[list[ReconstructedBlock], list[RegionBlock]]:
        """
        Assigns macro zone labels (vendor_block, header, buyer_block, totals, footer)
        to each reconstructed block and emits authoritative RegionBlocks.
        """
        macro_regions: list[RegionBlock] = []

        for blk in blocks:
            bn = blk.bbox_norm
            t_lower = blk.text.lower()

            # 1. Header / Metadata
            if any(k in t_lower for k in ["invoice no", "inv no", "invoice date", "due date", "po no", "place of supply"]):
                blk.zone_label = "header"
            # 2. Buyer / Bill To
            elif any(k in t_lower for k in ["bill to", "billed to", "buyer", "consignee", "customer:"]):
                blk.zone_label = "buyer_block"
            # 3. Totals
            elif any(k in t_lower for k in ["subtotal", "sub total", "cgst", "sgst", "igst", "grand total", "net payable", "round off"]):
                blk.zone_label = "totals"
            # 4. Bank / Footer
            elif any(k in t_lower for k in ["bank", "ifsc", "a/c no", "account no", "authorized signatory", "terms & conditions"]):
                blk.zone_label = "footer"
            # 5. Position-based heuristics
            elif bn[3] <= 280:
                if bn[0] < 500:
                    blk.zone_label = "vendor_block"
                else:
                    blk.zone_label = "header"
            elif bn[1] < 450:
                if bn[0] < 550:
                    blk.zone_label = "buyer_block"
                else:
                    blk.zone_label = "header"
            elif bn[1] >= 650:
                if bn[0] >= 400:
                    blk.zone_label = "totals"
                else:
                    blk.zone_label = "footer"
            else:
                blk.zone_label = "line_items"

        # Emit unified macro RegionBlocks by grouping blocks of same label
        from collections import defaultdict
        grouped: dict[str, list[ReconstructedBlock]] = defaultdict(list)
        for blk in blocks:
            grouped[blk.zone_label].append(blk)

        for label, blist in grouped.items():
            min_x = min(b.bbox_norm[0] for b in blist)
            min_y = min(b.bbox_norm[1] for b in blist)
            max_x = max(b.bbox_norm[2] for b in blist)
            max_y = max(b.bbox_norm[3] for b in blist)
            raw_min_x = min(b.bbox[0] for b in blist)
            raw_min_y = min(b.bbox[1] for b in blist)
            raw_max_x = max(b.bbox[2] for b in blist)
            raw_max_y = max(b.bbox[3] for b in blist)

            macro_regions.append(
                RegionBlock(
                    label=label,
                    bbox_norm=[min_x, min_y, max_x, max_y],
                    bbox_raw=[raw_min_x, raw_min_y, raw_max_x, raw_max_y],
                    confidence=0.92,
                    page=page_num,
                )
            )

        # Merge with visual stamps/signatures from YOLO if available
        if yolo_regions:
            for yr in yolo_regions:
                label = getattr(yr, "label", "region")
                if label in ("stamp", "signature", "handwriting") or getattr(yr, "is_handwritten", False):
                    yr_norm = normalize_box(getattr(yr, "bbox", [0,0,0,0]), self.page_width, self.page_height)
                    macro_regions.append(
                        RegionBlock(
                            label=label,
                            bbox_norm=yr_norm,
                            bbox_raw=getattr(yr, "bbox", [0,0,0,0]),
                            confidence=getattr(yr, "confidence", 0.90),
                            page=page_num,
                        )
                    )

        return blocks, macro_regions
