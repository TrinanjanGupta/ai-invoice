"""
preprocessing/handwriting_cropper.py

Field and line-level crop extraction pipeline.
Crops targeted visual zones from full page images:
1. YOLO-detected handwriting, signature, and stamp regions.
2. TIE anchor-adjacent value regions.
3. Sub-segmentation of multi-line handwriting crops into individual text lines.
"""

from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Any
from loguru import logger

from detection.detector import DetectedRegion
from preprocessing.document_profile import DocumentProfile


@dataclass
class FieldCrop:
    field_name: str             # "invoice_number", "grand_total", etc.
    crop_image: np.ndarray      # Original crop image (BGR)
    enhanced_crop: np.ndarray   # Enhanced crop image (BGR)
    bbox_page: list[int]        # [x1, y1, x2, y2] in page coordinates
    label_text: str = ""        # Anchor / region label text
    crop_source: str = "yolo_region"  # "yolo_region" | "tie_anchor" | "spatial_zone"
    page: int = 1


def expand_crop_margin(
    bbox: list[int] | tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    margin_pct: float = 0.05,
) -> list[int]:
    """Expands bounding box slightly to avoid clipping cursive strokes or descenders."""
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    dx = int(bw * margin_pct)
    dy = int(bh * margin_pct)

    nx1 = max(0, x1 - dx)
    ny1 = max(0, y1 - dy)
    nx2 = min(img_w, x2 + dx)
    ny2 = min(img_h, y2 + dy)
    return [nx1, ny1, nx2, ny2]


def crop_from_yolo_regions(
    image: np.ndarray,
    regions: list[DetectedRegion],
    enhanced_image: Optional[np.ndarray] = None,
    page: int = 1,
) -> list[FieldCrop]:
    """Crops all detected handwriting/custom regions from full-page raster."""
    h, w = image.shape[:2]
    enh = enhanced_image if enhanced_image is not None else image
    crops: list[FieldCrop] = []

    for r in regions:
        exp_bbox = expand_crop_margin(r.bbox, w, h, margin_pct=0.04)
        x1, y1, x2, y2 = exp_bbox
        orig_crop = image[y1:y2, x1:x2]
        enh_crop = enh[y1:y2, x1:x2]

        if orig_crop.size > 0:
            crops.append(FieldCrop(
                field_name=r.label,
                crop_image=orig_crop,
                enhanced_crop=enh_crop if enh_crop.size > 0 else orig_crop,
                bbox_page=exp_bbox,
                label_text=r.label,
                crop_source="yolo_region",
                page=getattr(r, "page", page),
            ))

    return crops


def crop_from_tie_anchors(
    image: np.ndarray,
    profile: DocumentProfile,
    field_rules: list[dict],
    enhanced_image: Optional[np.ndarray] = None,
    page: int = 1,
) -> list[FieldCrop]:
    """
    Uses TIE anchor relative boxes to crop the value regions adjacent to detected anchor labels.
    """
    h, w = image.shape[:2]
    enh = enhanced_image if enhanced_image is not None else image
    crops: list[FieldCrop] = []

    for rule in field_rules:
        target_field = rule.get("target_field")
        strategy = rule.get("strategy")
        if strategy != "anchor_relative" or not target_field:
            continue

        anchor_pattern = rule.get("anchor_pattern", "").lower()
        offset_box = rule.get("offset_box")  # [rel_x1, rel_y1, rel_x2, rel_y2] in normalized coords (0-1000)
        if not offset_box:
            continue

        # Find matching anchor token in profile
        token_list = getattr(profile, "tokens", None) or getattr(profile, "words", [])
        for tok in token_list:
            if anchor_pattern in tok.text.lower():
                # Compute absolute value box from anchor position + relative offset
                tok_box = getattr(tok, "bbox_norm", getattr(tok, "bbox", [0, 0, 0, 0]))
                if len(tok_box) == 4:
                    ax1, ay1, ax2, ay2 = tok_box
                    # Token bbox is in 0-1000 space, convert offset
                    rx1, ry1, rx2, ry2 = offset_box
                    vx1 = int(max(0, min(1000, ax1 + rx1)) * (w / 1000.0))
                    vy1 = int(max(0, min(1000, ay1 + ry1)) * (h / 1000.0))
                    vx2 = int(max(0, min(1000, ax2 + rx2)) * (w / 1000.0))
                    vy2 = int(max(0, min(1000, ay2 + ry2)) * (h / 1000.0))

                if vx2 > vx1 and vy2 > vy1:
                    exp_bbox = expand_crop_margin([vx1, vy1, vx2, vy2], w, h, margin_pct=0.05)
                    bx1, by1, bx2, by2 = exp_bbox
                    orig_crop = image[by1:by2, bx1:bx2]
                    enh_crop = enh[by1:by2, bx1:bx2]

                    if orig_crop.size > 0:
                        crops.append(FieldCrop(
                            field_name=target_field,
                            crop_image=orig_crop,
                            enhanced_crop=enh_crop if enh_crop.size > 0 else orig_crop,
                            bbox_page=exp_bbox,
                            label_text=tok.text,
                            crop_source="tie_anchor",
                            page=page,
                        ))
                        break  # Found anchor match for this field rule

    return crops


def crop_text_lines(region_crop: np.ndarray) -> list[np.ndarray]:
    """
    Sub-segments a multi-line handwriting region crop into individual horizontal text lines
    using horizontal projection profile analysis.
    """
    if region_crop is None or region_crop.size == 0:
        return []

    h, w = region_crop.shape[:2]
    if h < 20:
        return [region_crop]

    gray = cv2.cvtColor(region_crop, cv2.COLOR_BGR2GRAY) if len(region_crop.shape) == 3 else region_crop
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Compute horizontal projection profile (sum of foreground pixels per row)
    proj = np.sum(thresh > 0, axis=1)

    # Find line boundaries (transitions from low/zero projection to active text rows)
    threshold_val = max(2, int(w * 0.015))
    in_line = False
    start_y = 0
    line_slices: list[tuple[int, int]] = []

    for y in range(h):
        if proj[y] > threshold_val and not in_line:
            in_line = True
            start_y = max(0, y - 2)
        elif proj[y] <= threshold_val and in_line:
            in_line = False
            end_y = min(h, y + 2)
            if (end_y - start_y) >= 8:  # Minimum line height
                line_slices.append((start_y, end_y))

    if in_line and (h - start_y) >= 8:
        line_slices.append((start_y, h))

    if not line_slices:
        return [region_crop]

    # Return line image crops
    line_crops = []
    for sy, ey in line_slices:
        crop = region_crop[sy:ey, :]
        if crop.size > 0:
            line_crops.append(crop)

    return line_crops
